# -*- coding: utf-8 -*-
"""
PathSense v2 — Predictor
Wraps the calibrated LightGBM model.
Supports both:
  - GPS-based inference (lat/lon → auto-enrich → predict)
  - Manual inference (explicit feature dict, backward-compat with v1)
"""
from __future__ import annotations

import json
import logging
import math
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def _risk_bucket(prob: float, threshold: float = 0.45) -> str:
    """
    v2 uses a recall-optimized threshold (not fixed 0.5).
    Calibrated probabilities mean VERY_HIGH is genuinely very high.
    """
    if prob >= 0.70:
        return "VERY_HIGH"
    if prob >= threshold:
        return "MODERATE"
    return "LOW"


class RiskPredictor:
    """
    Unified predictor that:
      1. Loads the v2 LightGBM model if available
      2. Falls back to v1 XGBoost model if v2 not yet trained
      3. Accepts GPS coords for full auto-enrichment
      4. Accepts manual feature dict for backward compatibility
    """

    def __init__(self, artifacts_dir: Optional[Path] = None):
        d = artifacts_dir or ARTIFACTS

        # Try to load v3 model first
        v3_model_path = d / "lgbm_v3_calibrated.pkl"
        v3_meta_path = d / "model_meta_v3.json"
        
        v2_model_path = d / "lgbm_calibrated.pkl"
        v2_meta_path = d / "model_meta_v2.json"

        if v3_model_path.exists() and v3_meta_path.exists():
            self.model = joblib.load(v3_model_path)
            meta = json.loads(v3_meta_path.read_text(encoding="utf-8"))
            self.version = "3.0"
            self.label_encoders = joblib.load(d / meta.get("label_encoders_file", "label_encoders_v3.pkl"))
            self.optimal_threshold = meta.get("optimal_threshold", 0.5)
            logger.info("Loaded PathSense v3 model (LightGBM)")
        elif v2_model_path.exists() and v2_meta_path.exists():
            self.model = joblib.load(v2_model_path)
            meta = json.loads(v2_meta_path.read_text(encoding="utf-8"))
            self.version = "2.0"
            self.label_encoders = joblib.load(d / meta.get("label_encoders_file", "label_encoders_v2.pkl"))
            self.optimal_threshold = meta.get("optimal_threshold", 0.45)
            logger.info("Loaded PathSense v2 model (LightGBM)")
        else:
            # Fallback to v1. The trained v1 artifacts (xgboost_risk.pkl,
            # random_forest.pkl, label_encoders.pkl, model_meta.json) live
            # directly under ml/, not ml/artifacts/ — that folder is only
            # ever created once a v2 model has been trained. Resolve to
            # wherever model_meta.json actually is so this doesn't throw
            # FileNotFoundError when v2 hasn't been trained yet.
            v1_dir = d if (d / "model_meta.json").exists() else ROOT
            meta = json.loads((v1_dir / "model_meta.json").read_text(encoding="utf-8"))

            # Respect whichever model model_meta.json actually names as
            # primary, instead of hardcoding one file. model_report.json
            # (from the same training run) showed random_forest.pkl beating
            # xgboost_risk.pkl by 13+ points of holdout accuracy (64.3% vs
            # 51%, near-chance), so model_meta.json's primary_model_file has
            # been updated to random_forest.pkl — this line just makes the
            # code actually honor that instead of silently ignoring it.
            primary_file = meta.get("primary_model_file", "xgboost_risk.pkl")
            model_path = v1_dir / primary_file
            if not model_path.exists():
                # Defensive fallback if the named file is ever missing.
                model_path = v1_dir / "xgboost_risk.pkl"
                logger.warning("%s not found — falling back to xgboost_risk.pkl", primary_file)

            self.model = joblib.load(model_path)
            self.version = f"1.0 ({model_path.stem})"
            self.label_encoders = joblib.load(v1_dir / "label_encoders.pkl")
            self.optimal_threshold = 0.5
            logger.warning(
                "v2 model not found — using v1 %s (run train_v2.py to upgrade)",
                model_path.name,
            )

        self.feature_columns: list[str] = meta["feature_columns"]
        self.cat_cols: list[str] = meta["categorical_columns"]

    # ── GPS-based prediction (v2 recommended path) ───────────────────────────

    def predict_from_gps(
        self,
        lat: float,
        lon: float,
        dt: Optional[datetime] = None,
        speed_kmh: float = 0.0,
        gps_accuracy_m: float = 5.0,
        heading_change_deg: float = 0.0,
        num_vehicles: int = 1,
        road_condition: str = "Dry",
        traffic_control: str = "Signals",
    ) -> dict:
        """
        Full GPS-based prediction with real-time feature enrichment.
        Returns prediction + enriched features for debugging.
        """
        from .feature_engineering import enrich_request
        features = enrich_request(
            lat=lat, lon=lon, dt=dt,
            speed_kmh=speed_kmh, gps_accuracy_m=gps_accuracy_m,
            heading_change_deg=heading_change_deg,
            num_vehicles=num_vehicles,
            road_condition=road_condition,
            traffic_control=traffic_control,
        )
        # Strip metadata keys
        feature_input = {k: v for k, v in features.items() if not k.startswith("_")}
        prob = self._score(feature_input)
        level = _risk_bucket(prob, self.optimal_threshold)
        return {
            "probability": round(prob, 4),
            "risk_level": level,
            "message": self.alert_message(prob),
            "color": {"LOW": "#22c55e", "MODERATE": "#f59e0b", "VERY_HIGH": "#ef4444"}[level],
            "enriched_features": features,
            "model_version": self.version,
        }

    # ── Manual dict-based prediction (v1 backward-compat) ───────────────────

    def predict_proba_high_risk(self, input_data: dict) -> float:
        """
        Drop-in replacement for v1 API. Accepts explicit feature dict.
        """
        return self._score(input_data)

    def _score(self, input_data: dict) -> float:
        """
        Core scoring: encode categoricals, align to feature columns, infer.
        Unknown features are filled with 0 (safe default).
        """
        row = dict(input_data)

        # Encode categorical columns
        for col in self.cat_cols:
            if col not in row:
                continue
            le = self.label_encoders.get(col)
            if le is None:
                continue
            val = str(row[col])
            if val not in le.classes_:
                val = le.classes_[0]  # fallback to first class
            row[col] = int(le.transform([val])[0])

        # Build DataFrame aligned to model's feature columns
        # Missing columns → 0
        aligned = {col: row.get(col, 0) for col in self.feature_columns}
        df = pd.DataFrame([aligned])

        return float(self.model.predict_proba(df)[0, 1])

    def alert_message(self, prob: float) -> str:
        level = _risk_bucket(prob, self.optimal_threshold)
        if level == "VERY_HIGH":
            return (
                "Very high risk detected. Stop if safe. "
                "Consider waiting or replanning your route."
            )
        if level == "MODERATE":
            return "Moderate risk. Proceed with extra caution. Stay on footpaths."
        return "Low risk. Safe to proceed. Stay alert to surroundings."

    def spoken_alert(self, prob: float, context: dict) -> str:
        """
        Generates a natural-language spoken alert for TTS.
        More informative than v1 — includes specific hazard reasons.
        """
        level = _risk_bucket(prob, self.optimal_threshold)
        pct = round(prob * 100)
        reasons = []

        if context.get("is_dark"):
            reasons.append("poor lighting")
        if context.get("is_bad_weather") or context.get("precipitation_mm", 0) > 0:
            reasons.append("adverse weather")
        if context.get("Speed Limit (km/h)", 0) > 70:
            reasons.append("high speed traffic zone")
        if context.get("Road Type") in ("National Highway", "State Highway"):
            reasons.append("high-speed road")
        if context.get("crosswalk_nearby") == 0:
            reasons.append("no crosswalk detected")
        if context.get("sudden_direction_change"):
            reasons.append("sudden movement detected")

        reason_str = ", ".join(reasons) if reasons else "current conditions"
        base = (
            f"Risk assessment: {level.replace('_', ' ').lower()}. "
            f"{pct} percent probability of serious accident. "
        )
        if reasons:
            base += f"Main hazards: {reason_str}. "
        base += self.alert_message(prob)
        return base


def write_alert_wav(
    out_path: Path,
    prob: float,
    sample_rate: int = 22050,
    duration_s: float = 0.5,
) -> None:
    """
    v2 improvement: dual-tone alert for VERY_HIGH, single tone for MODERATE.
    Rhythm also encodes risk: rapid pulses = very high, single pulse = moderate.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    level = _risk_bucket(prob)
    n = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)

    if level == "VERY_HIGH":
        # Dual tone — 880Hz + 1320Hz = alarm-like
        wave_data = (
            0.15 * np.sin(2 * math.pi * 880 * t)
            + 0.15 * np.sin(2 * math.pi * 1320 * t)
        )
    elif level == "MODERATE":
        wave_data = 0.2 * np.sin(2 * math.pi * 660 * t)
    else:
        wave_data = 0.15 * np.sin(2 * math.pi * 440 * t)

    # Fade in/out to avoid clicks
    fade = int(sample_rate * 0.02)
    wave_data[:fade] *= np.linspace(0, 1, fade)
    wave_data[-fade:] *= np.linspace(1, 0, fade)

    pcm = (wave_data * 32767).astype(np.int16)
    with wave.open(str(out_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
