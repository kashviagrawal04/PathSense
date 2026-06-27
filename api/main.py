# -*- coding: utf-8 -*-
"""
PathSense v2 — Main API
Serves predictions, model stats, GPS-based inference, and audio alerts.
Uses Redis for caching and supports both v1 (manual fields) and v2 (GPS) paths.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import redis as redis_lib
from fastapi import FastAPI, HTTPException, BackgroundTasks, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "ml"
ARTIFACTS = ML_DIR / "artifacts"

load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ML_DIR))

from predictor_v2 import RiskPredictor, _risk_bucket  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

API_KEY = os.getenv('API_KEY')
api_key_header = APIKeyHeader(name='X-API-Key', auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if API_KEY and api_key_header != API_KEY:
        raise HTTPException(status_code=403, detail='Could not validate API KEY')
    return api_key_header

app = FastAPI(
    title="PathSense API v2",
    description="Real-time pedestrian risk prediction for safer navigation.",
    version="2.0.0",
)
allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware, allow_origins=allowed_origins, allow_methods=["*"], allow_headers=["*"]
)

predictor = RiskPredictor(ARTIFACTS)

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_redis: Optional[redis_lib.Redis] = None


def get_redis():
    global _redis
    if _redis is None:
        try:
            _redis = redis_lib.from_url(REDIS_URL, decode_responses=True, socket_timeout=1)
            _redis.ping()
        except Exception:
            _redis = None
    return _redis


RISK_COLORS = {"LOW": "#22c55e", "MODERATE": "#f59e0b", "VERY_HIGH": "#ef4444"}
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


# ── V1 request schema (backward compat) ──────────────────────────────────────

class PredictRequest(BaseModel):
    weather: str = Field(..., example="Clear")
    lighting: str = Field(..., example="Daylight")
    road_type: str = Field(..., example="Urban Road")
    road_condition: str = Field(..., example="Dry")
    speed_limit: int = Field(..., ge=0, le=200, example=40)
    time_category: str = Field(..., example="Morning")
    day_of_week: str = Field(..., example="Tuesday")
    num_vehicles: int = Field(..., ge=1, le=1000, example=1)
    traffic_control: str = Field(..., example="Signals")


# ── V2 GPS request schema ─────────────────────────────────────────────────────

class GPSPredictRequest(BaseModel):
    user_id: str = Field(..., example="user_abc123")
    lat: float = Field(..., ge=-90, le=90, example=28.6139)
    lon: float = Field(..., ge=-180, le=180, example=77.2090)
    speed_kmh: float = Field(default=0.0, ge=0, example=3.5)
    gps_accuracy_m: float = Field(default=5.0, ge=0, example=4.2)
    heading_change_deg: float = Field(default=0.0, example=15.0)
    num_vehicles: int = Field(default=1, ge=0, example=2)
    road_condition: str = Field(default="Dry", example="Wet")
    traffic_control: str = Field(default="Signals", example="Signs")


class PredictResponse(BaseModel):
    probability: float
    risk_level: Literal["LOW", "MODERATE", "VERY_HIGH"]
    message: str
    color: str
    model_version: str = "2.0"
    from_cache: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _map_v1_request(req: PredictRequest) -> dict:
    return {
        "Weather Conditions": req.weather,
        "Lighting Conditions": req.lighting,
        "Road Type": req.road_type,
        "Road Condition": req.road_condition,
        "Speed Limit (km/h)": req.speed_limit,
        "Time_Category": req.time_category,
        "Day of Week": req.day_of_week,
        "Number of Vehicles Involved": req.num_vehicles,
        "Traffic Control Presence": req.traffic_control,
    }


def _gps_cache_key(lat: float, lon: float) -> str:
    return f"pred:{round(lat, 3)}:{round(lon, 3)}"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {
        "status": "ok",
        "service": "PathSense API v2",
        "model_version": predictor.version,
        "redis_connected": get_redis() is not None,
    }


@app.post("/predict", response_model=PredictResponse, tags=["prediction"], dependencies=[Depends(get_api_key)])
def predict(req: PredictRequest):
    """V1-compatible prediction endpoint. Accepts explicit feature values."""
    try:
        prob = predictor.predict_proba_high_risk(_map_v1_request(req))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    level = _risk_bucket(prob, predictor.optimal_threshold)
    return PredictResponse(
        probability=round(prob, 4),
        risk_level=level,
        message=predictor.alert_message(prob),
        color=RISK_COLORS[level],
        model_version=predictor.version,
    )


@app.post("/predict/gps", response_model=PredictResponse, tags=["prediction"], dependencies=[Depends(get_api_key)])
def predict_gps(req: GPSPredictRequest):
    """
    V2 GPS-based prediction. Auto-enriches features from OpenWeatherMap + OSM.
    Results cached in Redis for 2 minutes per GPS tile.
    """
    ck = _gps_cache_key(req.lat, req.lon)
    r = get_redis()

    # Check cache
    if r:
        cached = r.get(ck)
        if cached:
            data = json.loads(cached)
            data["from_cache"] = True
            return PredictResponse(**data)

    try:
        result = predictor.predict_from_gps(
            lat=req.lat, lon=req.lon,
            speed_kmh=req.speed_kmh,
            gps_accuracy_m=req.gps_accuracy_m,
            heading_change_deg=req.heading_change_deg,
            num_vehicles=req.num_vehicles,
            road_condition=req.road_condition,
            traffic_control=req.traffic_control,
        )
    except Exception as exc:
        logger.error("GPS prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    response_data = {
        "probability": result["probability"],
        "risk_level": result["risk_level"],
        "message": result["message"],
        "color": result["color"],
        "model_version": result.get("model_version", predictor.version),
        "from_cache": False,
    }

    # Cache for 120 seconds
    if r:
        try:
            r.setex(ck, 120, json.dumps(response_data))
        except Exception:
            pass

    return PredictResponse(**response_data)


@app.post("/predict/audio", tags=["prediction"], dependencies=[Depends(get_api_key)])
def predict_audio(req: GPSPredictRequest, background_tasks: BackgroundTasks):
    """
    Returns a WAV file with a spoken risk assessment using TTS.
    Uses pyttsx3 (offline) or a pitched beep fallback.
    """
    try:
        result = predictor.predict_from_gps(
            lat=req.lat, lon=req.lon,
            speed_kmh=req.speed_kmh,
            gps_accuracy_m=req.gps_accuracy_m,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    prob = result["probability"]
    enriched = result.get("enriched_features", {})
    spoken_text = predictor.spoken_alert(prob, enriched)

    tmp_dir = Path(tempfile.mkdtemp())
    wav_path = tmp_dir / "alert.wav"

    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        engine.setProperty("volume", 0.9)
        engine.save_to_file(spoken_text, str(wav_path))
        engine.runAndWait()
    except Exception:
        from predictor_v2 import write_alert_wav
        write_alert_wav(wav_path, prob)

    def cleanup():
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    background_tasks.add_task(cleanup)

    return FileResponse(
        path=str(wav_path),
        media_type="audio/wav",
        filename="pathsense_alert.wav",
        headers={
            "X-Risk-Level": result["risk_level"],
            "X-Risk-Probability": str(result["probability"]),
            "X-Spoken-Text": spoken_text[:300],
        },
    )


@app.get("/model/stats", tags=["model"])
def model_stats():
    """Return current model performance report."""
    v2_path = ARTIFACTS / "model_report_v2.json"
    v1_path = ARTIFACTS / "model_report.json"
    path = v2_path if v2_path.exists() else v1_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="No model report found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/model/meta", tags=["model"])
def model_meta():
    v2_path = ARTIFACTS / "model_meta_v2.json"
    v1_path = ARTIFACTS / "model_meta.json"
    path = v2_path if v2_path.exists() else v1_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="No model meta found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/options", tags=["prediction"])
def get_options():
    """Return valid categorical values (from label encoders)."""
    import joblib
    enc_path = ARTIFACTS / "label_encoders_v2.pkl"
    if not enc_path.exists():
        enc_path = ARTIFACTS / "label_encoders.pkl"
    le_dict = joblib.load(enc_path)
    return {col: list(le.classes_) for col, le in le_dict.items()}


@app.get("/cache/stats", tags=["cache"])
def cache_stats():
    """Redis cache diagnostics."""
    r = get_redis()
    if not r:
        return {"redis": "unavailable"}
    try:
        info = r.info("stats")
        return {
            "redis": "connected",
            "keyspace_hits": info.get("keyspace_hits"),
            "keyspace_misses": info.get("keyspace_misses"),
            "hit_rate": (
                info.get("keyspace_hits", 0) /
                max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0))
            ),
        }
    except Exception as e:
        return {"redis": "error", "detail": str(e)}


class SOSRequest(BaseModel):
    phone_numbers: list[str] = Field(..., example=["+919999999999"])
    message: str = Field(..., example="SOS! I need help.")
    lat: Optional[float] = None
    lon: Optional[float] = None


@app.post("/send-sos", tags=["notification"])
def send_sos(req: SOSRequest):
    """Send SOS SMS via Twilio to emergency contacts."""
    try:
        from twilio.rest import Client
    except ImportError:
        raise HTTPException(status_code=500, detail="Twilio not installed")

    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")

    if not all([account_sid, auth_token, from_number]):
        raise HTTPException(status_code=500, detail="Missing Twilio configuration")

    location_str = ""
    if req.lat and req.lon:
        location_str = f"\nLocation: https://maps.google.com/?q={req.lat},{req.lon}"

    client = Client(account_sid, auth_token)
    results = []
    for to_number in req.phone_numbers:
        if not to_number.startswith("+"):
            raise HTTPException(status_code=400, detail=f"Invalid number: {to_number}")
        try:
            msg = client.messages.create(
                body=req.message + location_str,
                from_=from_number,
                to=to_number,
            )
            results.append({"to": to_number, "sid": msg.sid, "status": msg.status})
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    return {"status": "sent", "sent": len(results), "results": results}
