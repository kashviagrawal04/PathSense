# -*- coding: utf-8 -*-
"""
PathSense v3 — YOLO Obstacle Detection Service
Runs YOLOv8n on phone camera frames to detect:
  - Vehicles (car, truck, bus, motorbike)
  - Pedestrians
  - Traffic lights (and their state)
  - Stop signs / crosswalks

Outputs structured hazard features that feed directly into the risk model
AND generates immediate audio alerts for objects within danger distance.

Architecture:
  Mobile app → POST /detect (JPEG frame) → FastAPI → YOLOv8 → hazard features + audio alert
  OR
  Mobile app runs YOLOv8 on-device (recommended for latency) → POST /score (features only)

Run: uvicorn services.vision.main:app --port 8002
"""
from __future__ import annotations

import base64
import io
import logging
import math
import os
import wave
from pathlib import Path
from typing import Optional

import wave
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

MODEL_SIZE = os.getenv("YOLO_MODEL_SIZE", "yolov8n")  # n=nano (fastest), s/m/l/x for accuracy
DANGER_ZONE_M = float(os.getenv("DANGER_ZONE_M", "5.0"))   # objects within 5m = danger
WARN_ZONE_M   = float(os.getenv("WARN_ZONE_M",   "15.0"))  # objects within 15m = warning

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.warning("WARNING: API_KEY environment variable is not set. Running in insecure mode.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if API_KEY and api_key_header != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API KEY")
    return api_key_header

origins_str = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in origins_str.split(",") if o.strip()]
if not ALLOWED_ORIGINS:
    logger.warning("No ALLOWED_ORIGINS set. CORS will block all browser requests.")

app = FastAPI(title="PathSense Vision Service", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])

# ── YOLO model (lazy-loaded — takes ~2s on first call) ───────────────────────

_yolo_model = None


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            model_path = Path(__file__).parent / f"{MODEL_SIZE}.pt"
            # Downloads automatically on first call (~6MB for nano)
            _yolo_model = YOLO(str(model_path) if model_path.exists() else MODEL_SIZE)
            logger.info("YOLO model loaded: %s", MODEL_SIZE)
        except ImportError:
            logger.error(
                "ultralytics not installed. Run: pip install ultralytics\n"
                "Falling back to mock detector."
            )
            _yolo_model = MockDetector()
    return _yolo_model


# ── COCO class IDs relevant to pedestrian safety ─────────────────────────────

VEHICLE_CLASSES  = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
PERSON_CLASS     = 0
TRAFFIC_LIGHT    = 9
STOP_SIGN        = 11
BICYCLE          = 1
FIRE_HYDRANT     = 10  # good pedestrian landmark

ALL_RELEVANT = set(VEHICLE_CLASSES.keys()) | {PERSON_CLASS, TRAFFIC_LIGHT, STOP_SIGN, BICYCLE}

# ── Schemas ───────────────────────────────────────────────────────────────────

class FrameRequest(BaseModel):
    """POST a base64 JPEG frame for full YOLO detection."""
    frame_b64: str = Field(..., description="Base64-encoded JPEG image from phone camera")
    camera_height_m: float = Field(default=1.2, description="Camera height above ground (metres)")
    focal_length_px: float = Field(default=800.0, description="Camera focal length in pixels")
    frame_width_px:  int   = Field(default=1280)
    frame_height_px: int   = Field(default=720)
    tilt_deg: float = Field(
        default=0.0,
        description="Device tilt from upright, in degrees, reported by the phone's "
                     "orientation sensors (e.g. via the browser's deviceorientation "
                     "event). Used to correct bounding-box height before distance "
                     "estimation, since a tilted camera foreshortens objects."
    )


class VisionFeatures(BaseModel):
    """Structured features output — fed into the risk model."""
    # Vehicle detections
    vehicles_in_danger_zone: int   = 0   # within DANGER_ZONE_M
    vehicles_in_warn_zone:   int   = 0   # within WARN_ZONE_M
    closest_vehicle_m:       float = 999.0
    largest_vehicle_area:    float = 0.0  # fraction of frame

    # Pedestrian context
    pedestrians_detected:    int   = 0
    pedestrian_density:      float = 0.0  # per 100m² estimated

    # Traffic signals
    traffic_light_visible:   int   = 0
    traffic_light_state:     str   = "unknown"  # red | amber | green | unknown
    stop_sign_visible:       int   = 0

    # Scene complexity
    total_objects:           int   = 0
    scene_clutter_score:     float = 0.0   # 0-1, higher = more cluttered

    # Derived risk flags (fed to model)
    imminent_collision_risk: int   = 0    # vehicle <3m
    unsafe_to_cross:         int   = 0    # traffic light red or vehicle approaching

    # Camera orientation
    camera_tilt_deg:          float = 0.0
    distances_tilt_corrected: int   = 0   # 1 if a tilt correction was applied
    distances_unreliable:     int   = 0   # 1 if tilt was too extreme to trust distances

    # Alert
    alert_text:              str   = ""
    alert_severity:          str   = "none"   # none | caution | danger | stop


class ScoreOnlyRequest(BaseModel):
    """
    Alternative: mobile runs YOLO on-device, sends features only (lower bandwidth).
    Recommended for production — keeps inference on device, no image upload.
    """
    features: VisionFeatures


# ── Distance estimation ───────────────────────────────────────────────────────

def estimate_distance_m(
    bbox_height_px: float,
    object_real_height_m: float,
    focal_length_px: float,
) -> float:
    """
    Pinhole camera model: distance = (real_height × focal_length) / bbox_height
    Accurate to ~20% for objects at 2-20m with a calibrated focal length.
    """
    if bbox_height_px < 1:
        return 999.0
    return (object_real_height_m * focal_length_px) / bbox_height_px


# Beyond this tilt, the pinhole approximation breaks down badly enough that we
# no longer trust the corrected distance — better to flag it than report a
# confidently wrong number.
MAX_TRUSTED_TILT_DEG = 35.0


def tilt_corrected_bbox_height(bbox_height_px: float, tilt_deg: float) -> float:
    """
    A tilted camera foreshortens an object's apparent vertical extent — the same
    object at the same distance produces a shorter bounding box the more the
    phone is angled away from facing it head-on. Dividing the measured bbox
    height by cos(tilt) approximately undoes that foreshortening, so the
    downstream pinhole formula sees a height closer to what it would have
    measured from a perfectly upright camera.

    This is a first-order correction, not a full 3D re-projection — it assumes
    tilt mostly affects vertical foreshortening and ignores rotation about the
    optical axis. Good enough to meaningfully reduce error for moderate tilt;
    not a substitute for holding the phone upright in the first place.
    """
    tilt_rad = math.radians(min(abs(tilt_deg), 89.0))
    cos_tilt = math.cos(tilt_rad)
    if cos_tilt < 1e-3:
        return bbox_height_px  # avoid dividing by ~0 at extreme tilt
    return bbox_height_px / cos_tilt


# Real-world heights for distance estimation
REAL_HEIGHTS = {
    "car":        1.5,
    "truck":      3.0,
    "bus":        3.2,
    "motorbike":  1.1,
    "person":     1.7,
    "bicycle":    1.0,
}


# ── Traffic light colour detection ───────────────────────────────────────────

def detect_light_colour(image_np, bbox) -> str:
    """
    Crop the traffic light ROI and detect dominant colour (red/amber/green).
    Uses HSV colour analysis — no additional model needed.
    """
    try:
        import cv2
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        roi = image_np[y1:y2, x1:x2]
        if roi.size == 0:
            return "unknown"
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        # Red: hue 0-10 or 160-180
        red_mask = (
            cv2.inRange(hsv, np.array([0,100,100]),   np.array([10,255,255])) |
            cv2.inRange(hsv, np.array([160,100,100]), np.array([180,255,255]))
        )
        # Amber: hue 15-35
        amber_mask = cv2.inRange(hsv, np.array([15,100,100]), np.array([35,255,255]))
        # Green: hue 40-85
        green_mask = cv2.inRange(hsv, np.array([40,60,60]),   np.array([85,255,255]))

        red   = red_mask.sum()
        amber = amber_mask.sum()
        green = green_mask.sum()
        total = max(red + amber + green, 1)

        if red / total > 0.25:
            return "red"
        if green / total > 0.25:
            return "green"
        if amber / total > 0.15:
            return "amber"
        return "unknown"
    except Exception:
        return "unknown"


# ── Core detection logic ──────────────────────────────────────────────────────

def run_detection(image_np, focal_length_px: float = 800.0, tilt_deg: float = 0.0) -> VisionFeatures:
    """
    Run YOLO on a numpy RGB image and return structured VisionFeatures.

    tilt_deg (device tilt from upright, reported by the phone's orientation
    sensors) is used to correct each detection's bounding-box height before
    it's fed into the pinhole distance formula — see tilt_corrected_bbox_height().
    """
    model = get_yolo()
    h_frame, w_frame = image_np.shape[:2]
    frame_area = h_frame * w_frame

    results = model(image_np, verbose=False, conf=0.35)[0]
    boxes   = results.boxes

    feats = VisionFeatures()
    feats.total_objects = len(boxes)
    feats.camera_tilt_deg = tilt_deg

    tilt_too_extreme = abs(tilt_deg) > MAX_TRUSTED_TILT_DEG
    if tilt_too_extreme:
        feats.distances_unreliable = 1
    elif abs(tilt_deg) > 1.0:
        feats.distances_tilt_corrected = 1

    vehicle_distances = []
    light_state = "unknown"

    import sys
    print(f"YOLO detected {len(boxes)} total objects", file=sys.stderr, flush=True)
    
    for box in boxes:
        cls_id   = int(box.cls[0])
        conf     = float(box.conf[0])
        cls_name = model.names[cls_id]
        
        if cls_id in ALL_RELEVANT:
            print(f"Found relevant {cls_name} with conf {conf:.2f}", file=sys.stderr, flush=True)

        if cls_id not in ALL_RELEVANT:
            continue
        if conf < 0.50:
            continue

        # Traffic light parsing
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox_h   = y2 - y1
        bbox_area_frac = ((x2-x1) * bbox_h) / frame_area

        if cls_id in VEHICLE_CLASSES:
            vehicle_name = VEHICLE_CLASSES[cls_id]
            # Correct for tilt before estimating distance, unless the tilt is so
            # extreme the correction itself can't be trusted (flagged above).
            corrected_bbox_h = bbox_h if tilt_too_extreme else tilt_corrected_bbox_height(bbox_h, tilt_deg)
            dist = estimate_distance_m(corrected_bbox_h, REAL_HEIGHTS.get(vehicle_name, 1.5), focal_length_px)
            vehicle_distances.append(dist)
            if dist <= DANGER_ZONE_M:
                feats.vehicles_in_danger_zone += 1
            elif dist <= WARN_ZONE_M:
                feats.vehicles_in_warn_zone   += 1
            feats.largest_vehicle_area = max(feats.largest_vehicle_area, bbox_area_frac)

        elif cls_id == PERSON_CLASS:
            feats.pedestrians_detected += 1

        elif cls_id == TRAFFIC_LIGHT:
            feats.traffic_light_visible = 1
            light_state = detect_light_colour(image_np, (x1, y1, x2, y2))

        elif cls_id == STOP_SIGN:
            feats.stop_sign_visible = 1

    feats.traffic_light_state = light_state
    feats.scene_clutter_score = min(1.0, feats.total_objects / 20)

    if vehicle_distances:
        feats.closest_vehicle_m = min(vehicle_distances)
    if feats.closest_vehicle_m <= 3.0:
        feats.imminent_collision_risk = 1
    if light_state == "red" or feats.imminent_collision_risk:
        feats.unsafe_to_cross = 1

    # Generate alert
    feats.alert_severity, feats.alert_text = build_alert(feats)
    return feats


def build_alert(f: VisionFeatures) -> tuple[str, str]:
    """Generate spoken alert text and severity level from detected features."""
    
    ped_text = ""
    if getattr(f, 'pedestrians_detected', 0) > 0:
        ped_text = "Caution, people are there walk slowly. "

    if f.imminent_collision_risk:
        return "stop", f"{ped_text}STOP! Vehicle extremely close. Do not move."
    if f.vehicles_in_danger_zone >= 2:
        return "danger", f"{ped_text}Danger! {f.vehicles_in_danger_zone} vehicles within 5 metres."
    if f.vehicles_in_danger_zone == 1:
        return "danger", f"{ped_text}Vehicle nearby. Closest at {f.closest_vehicle_m:.0f} metres."
        
    if f.traffic_light_state == "red":
        return "danger", f"{ped_text}Red light. Do not cross."
    if f.traffic_light_state == "amber":
        return "caution", f"{ped_text}Amber light. Prepare to stop."
    if f.vehicles_in_warn_zone >= 3:
        return "caution", f"{ped_text}Caution. {f.vehicles_in_warn_zone} vehicles approaching."
    if f.vehicles_in_warn_zone >= 1:
        return "caution", f"{ped_text}Caution. Vehicle approaching, {f.closest_vehicle_m:.0f} metres away."
        
    if ped_text:
        return "caution", ped_text.strip()
        
    if f.traffic_light_state == "green":
        return "none", "Green light. Safe to cross when clear."
    return "none", "Path appears clear."


# ── Alert audio (pitched beep + spoken) ──────────────────────────────────────

def make_alert_wav(severity: str, text: str) -> bytes:
    """
    Generate a WAV audio alert. Priority:
    1. pyttsx3 TTS (spoken text, offline)
    2. Pitched beep (fallback)
    """
    # Frequency and rhythm encode severity even without TTS
    freq_map   = {"stop": 1480, "danger": 1047, "caution": 660, "none": 440}
    pulses_map = {"stop": 5,    "danger": 3,     "caution": 2,  "none": 1}
    freq   = freq_map.get(severity, 440)
    pulses = pulses_map.get(severity, 1)

    sr      = 22050
    pulse_s = 0.18
    gap_s   = 0.06
    n_pulse = int(sr * pulse_s)
    n_gap   = int(sr * gap_s)
    t       = np.linspace(0, pulse_s, n_pulse, endpoint=False)
    tone    = (0.25 * np.sin(2 * math.pi * freq * t) * 32767).astype(np.int16)

    # Fade
    fade = int(sr * 0.01)
    tone[:fade]  = (tone[:fade] * np.linspace(0, 1, fade)).astype(np.int16)
    tone[-fade:] = (tone[-fade:] * np.linspace(1, 0, fade)).astype(np.int16)
    silence = np.zeros(n_gap, dtype=np.int16)

    audio = np.concatenate([np.concatenate([tone, silence]) for _ in range(pulses)])

    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


# ── Mock detector (when ultralytics not installed) ────────────────────────────

class MockDetector:
    """Returns empty detections — for dev/test without GPU/ultralytics."""
    def __call__(self, image, **kwargs):
        class MockResult:
            class boxes:
                pass
            boxes = type("Boxes", (), {"__iter__": lambda s: iter([]), "__len__": lambda s: 0})()
        return [MockResult()]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", dependencies=[Depends(get_api_key)])
def health():
    model = get_yolo()
    return {
        "status": "ok",
        "model": MODEL_SIZE,
        "yolo_loaded": not isinstance(model, MockDetector),
        "danger_zone_m": DANGER_ZONE_M,
        "warn_zone_m": WARN_ZONE_M,
    }


@app.post("/detect", response_model=VisionFeatures, dependencies=[Depends(get_api_key)])
def detect_frame(req: FrameRequest):
    """
    Full detection: accepts base64 JPEG, runs YOLO, returns VisionFeatures.
    Use when running model server-side (higher accuracy, more bandwidth).
    Latency: ~80-200ms on CPU (nano model), ~20ms on GPU.
    """
    try:
        from PIL import Image
        img_bytes = base64.b64decode(req.frame_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_np = np.array(img)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    features = run_detection(image_np, req.focal_length_px, req.tilt_deg)
    return features


@app.post("/detect/audio", dependencies=[Depends(get_api_key)])
def detect_frame_audio(req: FrameRequest):
    """
    Same as /detect but returns WAV audio alert instead of JSON.
    Mobile app plays this directly through the earpiece.
    """
    try:
        from PIL import Image
        img_bytes = base64.b64decode(req.frame_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_np = np.array(img)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    features = run_detection(image_np, req.focal_length_px, req.tilt_deg)
    wav = make_alert_wav(features.alert_severity, features.alert_text)
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={
            "X-Alert-Severity": features.alert_severity,
            "X-Alert-Text": features.alert_text[:200],
            "X-Closest-Vehicle-M": str(round(features.closest_vehicle_m, 1)),
            "X-Traffic-Light": features.traffic_light_state,
        },
    )


@app.post("/score", response_model=dict, dependencies=[Depends(get_api_key)])
def score_features(req: ScoreOnlyRequest):
    """
    Lightweight endpoint: mobile sends pre-computed features (from on-device YOLO),
    server returns risk adjustment + alert. No image upload needed.
    """
    f   = req.features
    sev = f.alert_severity
    wav = make_alert_wav(sev, f.alert_text)

    # Risk delta: how much does the vision signal shift the model's risk score?
    risk_delta = 0.0
    if f.imminent_collision_risk:
        risk_delta = 0.95
    elif f.vehicles_in_danger_zone >= 2:
        risk_delta = 0.80
    elif f.vehicles_in_danger_zone == 1:
        risk_delta = 0.65
    elif f.vehicles_in_warn_zone >= 1:
        risk_delta = 0.40 + (f.vehicles_in_warn_zone * 0.05)
    elif f.traffic_light_state == "red":
        risk_delta = 0.70

    return {
        "alert_severity": sev,
        "alert_text": f.alert_text,
        "vision_risk_score": round(min(risk_delta, 1.0), 3),
        "audio_b64": base64.b64encode(wav).decode(),
        "features": f.model_dump(),
    }


@app.get("/demo/alert/{severity}", dependencies=[Depends(get_api_key)])
def demo_alert(severity: str):
    """Generate a demo alert WAV for testing audio output."""
    texts = {
        "stop":    "STOP! Vehicle extremely close. Do not move.",
        "danger":  "Danger! Vehicle within 5 metres.",
        "caution": "Caution. Vehicle approaching 10 metres away.",
        "none":    "Path appears clear.",
    }
    text = texts.get(severity, "Test alert.")
    wav  = make_alert_wav(severity, text)
    return Response(content=wav, media_type="audio/wav",
                    headers={"X-Alert-Text": text})
