# -*- coding: utf-8 -*-
"""
PathSense v3 — Fused Risk Scorer
Combines the LightGBM tabular model (weather, road, time)
with the YOLO vision signal (vehicles, traffic lights, proximity)
into a single calibrated risk score.

Fusion strategy:
  - Vision score overrides model when imminent collision detected (safety-critical)
  - Otherwise: weighted average (model 60% + vision 40%)
  - Final output: probability + risk level + spoken alert
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

ML_DIR    = Path(__file__).resolve().parent.parent / "ml"
ARTIFACTS = ML_DIR / "artifacts"
sys.path.insert(0, str(ML_DIR))


def _risk_bucket(prob: float, threshold: float = 0.45) -> str:
    if prob >= 0.75:
        return "VERY_HIGH"
    if prob >= threshold:
        return "MODERATE"
    return "LOW"


RISK_COLORS = {"LOW": "#22c55e", "MODERATE": "#f59e0b", "VERY_HIGH": "#ef4444"}


# ── Model loader ──────────────────────────────────────────────────────────────

class ModelLoader:
    """
    Loads the best available model version:
    v3 (LightGBM, real data) > v2 (LightGBM, v2 features) > v1 (XGBoost)
    """
    def __init__(self, artifacts_dir: Path = ARTIFACTS):
        import joblib
        self.threshold = 0.45

        for version, model_file, meta_file, enc_file in [
            ("3.0", "lgbm_v3_calibrated.pkl", "model_meta_v3.json", "label_encoders_v3.pkl"),
            ("2.0", "lgbm_calibrated.pkl",    "model_meta_v2.json", "label_encoders_v2.pkl"),
            ("1.0", "xgboost_risk.pkl",        "model_meta.json",    "label_encoders.pkl"),
        ]:
            mp = artifacts_dir / model_file
            mm = artifacts_dir / meta_file
            ep = artifacts_dir / enc_file
            if mp.exists() and mm.exists() and ep.exists():
                self.model   = joblib.load(mp)
                self.meta    = json.loads(mm.read_text())
                self.encoders= joblib.load(ep)
                self.version = version
                self.feature_cols = self.meta["feature_columns"]
                self.threshold = self.meta.get("optimal_threshold", 0.45)
                logger.info("Loaded model v%s from %s", version, mp)
                return

        raise RuntimeError("No model found in artifacts dir. Run train_v3.py first.")

    def predict(self, features: dict) -> float:
        """Score a feature dict. Unknown features default to 0."""
        import pandas as pd
        row = dict(features)
        for col in self.meta.get("categorical_columns", []):
            if col not in row:
                continue
            le = self.encoders.get(col)
            if le is None:
                continue
            val = str(row[col])
            if val not in le.classes_:
                val = le.classes_[0]
            row[col] = int(le.transform([val])[0])
        aligned = {c: row.get(c, 0) for c in self.feature_cols}
        df = pd.DataFrame([aligned])
        return float(self.model.predict_proba(df)[0, 1])


# ── Feature engineering for v3 schema ────────────────────────────────────────

def build_features_v3(
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
    Build model features from GPS + sensors.
    Pulls live weather from OpenWeatherMap and road type from OSM.
    Mirrors the unified schema columns + engineered features from train_v3.py.
    """
    if dt is None:
        dt = datetime.now()

    h    = dt.hour + dt.minute / 60.0
    month = dt.month

    # Try live enrichment; fall back to estimates
    weather   = "Clear"
    lighting  = "Daylight"
    road_type = "Urban Road"
    speed_kmh_road = 50

    try:
        sys.path.insert(0, str(ML_DIR))
        from feature_engineering import get_weather_features, get_road_features, estimate_lighting
        w = get_weather_features(lat, lon)
        r = get_road_features(lat, lon)
        l = estimate_lighting(lat, lon, dt)
        weather        = w["Weather Conditions"]
        road_type      = r["Road Type"]
        speed_kmh_road = r["Speed Limit (km/h)"]
        lighting       = l["Lighting Conditions"]
    except Exception:
        pass

    # Cyclical encoding
    hour_sin = math.sin(2 * math.pi * h / 24)
    hour_cos = math.cos(2 * math.pi * h / 24)

    # Risk scores
    lighting_risk_map = {"Dark":4,"Dusk":3,"Dawn":2,"Daylight":0}
    weather_risk_map  = {"Stormy":5,"Foggy":4,"Snowy":5,"Rainy":3,"Hazy":2,"Clear":0}
    road_risk_map     = {"National Highway":4,"State Highway":3,"Urban Road":2,"Village Road":1}
    cond_risk_map     = {"Icy":5,"Under Construction":3,"Wet":3,"Dry":0}
    ctrl_risk_map     = {"None":4,"Signs":2,"Guard":1,"Signals":0}

    lr = lighting_risk_map.get(lighting, 1)
    wr = weather_risk_map.get(weather, 1)
    rr = road_risk_map.get(road_type, 2)
    cr = cond_risk_map.get(road_condition, 0)
    tr = ctrl_risk_map.get(traffic_control, 2)

    composite = lr*0.25 + wr*0.20 + rr*0.20 + cr*0.20 + tr*0.15

    return {
        # Core tabular features (v3 schema)
        "hour_sin":          hour_sin,
        "hour_cos":          hour_cos,
        "month_sin":         math.sin(2 * math.pi * month / 12),
        "month_cos":         math.cos(2 * math.pi * month / 12),
        "is_night":          int((h >= 22) or (h <= 5)),
        "is_rush_hour":      int(7 <= h <= 9 or 16 <= h <= 19),
        "is_weekend":        int(dt.weekday() >= 5),
        "is_friday_night":   int(dt.weekday() == 4 and h >= 21),
        "lighting_risk":     float(lr),
        "weather_risk":      float(wr),
        "road_risk":         float(rr),
        "cond_risk":         float(cr),
        "ctrl_risk":         float(tr),
        "speed_limit_kmh":   float(speed_kmh_road),
        "speed_bin":         float(min(4, speed_kmh_road // 20)),
        "is_high_speed":     int(speed_kmh_road >= 80),
        "num_vehicles":      float(num_vehicles),
        "log_vehicles":      math.log1p(num_vehicles),
        "multi_vehicle":     int(num_vehicles >= 3),
        "composite_risk":    composite,
        # Interaction terms
        "dark_wet":          int(lr >= 3 and cr >= 3),
        "night_highway":     int((h >= 22 or h <= 5) and road_type == "National Highway"),
        "fog_dark":          int(weather == "Foggy" and (h >= 22 or h <= 5)),
        "storm_wet":         int(weather == "Stormy" and road_condition == "Wet"),
        "speed_dark":        int(speed_kmh_road >= 80 and lr >= 3),
        "no_ctrl_highway":   int(traffic_control == "None" and rr >= 3),
        "rush_multi_vehicle":int((7<=h<=9 or 16<=h<=19) and num_vehicles >= 3),
        "ped_dark":          0,  # updated if pedestrian_involved passed
        "icy_road":          int(road_condition == "Icy" and speed_kmh_road >= 60),
        # GPS
        "has_gps":           int(lat != 0 and lon != 0),
        "lat_rounded":       round(lat, 1),
        "lon_rounded":       round(lon, 1),
        # V1-compat keys (for fallback to v1 model)
        "Weather Conditions":  weather,
        "Lighting Conditions": lighting,
        "Road Type":           road_type,
        "Road Condition":      road_condition,
        "Speed Limit (km/h)": speed_kmh_road,
        "Time_Category": (
            "Morning"   if 5<=h<12 else
            "Afternoon" if 12<=h<17 else
            "Evening"   if 17<=h<21 else "Night"
        ),
        "Day of Week": dt.strftime("%A"),
        "Number of Vehicles Involved": num_vehicles,
        "Traffic Control Presence": traffic_control,
        # Metadata (stripped before scoring)
        "_weather": weather,
        "_lighting": lighting,
        "_road_type": road_type,
        "_lat": lat,
        "_lon": lon,
    }


# ── Fusion engine ─────────────────────────────────────────────────────────────

class FusedRiskScorer:
    """
    Combines tabular ML model + vision risk score into one final probability.
    """

    # How much weight to give vision vs tabular model
    VISION_WEIGHT  = 0.45
    TABULAR_WEIGHT = 0.55

    def __init__(self, artifacts_dir: Path = ARTIFACTS):
        self.loader = ModelLoader(artifacts_dir)

    def score(
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
        vision_features: Optional[dict] = None,
    ) -> dict:
        """
        Full fused prediction.
        vision_features: dict from VisionFeatures.model_dump() (optional)
        """
        # 1. Tabular model
        raw_features = build_features_v3(
            lat=lat, lon=lon, dt=dt,
            speed_kmh=speed_kmh, gps_accuracy_m=gps_accuracy_m,
            heading_change_deg=heading_change_deg,
            num_vehicles=num_vehicles,
            road_condition=road_condition,
            traffic_control=traffic_control,
        )
        clean = {k: v for k, v in raw_features.items() if not k.startswith("_")}
        tabular_prob = self.loader.predict(clean)

        # 2. Vision signal
        vision_prob  = None
        vision_override = False
        if vision_features:
            vf = vision_features
            if vf.get("imminent_collision_risk"):
                vision_prob     = 0.97
                vision_override = True   # safety-critical: override tabular
            elif vf.get("vehicles_in_danger_zone", 0) >= 2:
                vision_prob = 0.88
            elif vf.get("vehicles_in_danger_zone", 0) == 1:
                vision_prob = 0.75
            elif vf.get("traffic_light_state") == "red":
                vision_prob = 0.72
            elif vf.get("vehicles_in_warn_zone", 0) >= 1:
                vision_prob = 0.50 + vf["vehicles_in_warn_zone"] * 0.05
            else:
                vision_prob = max(tabular_prob * 0.8, 0.05)

        # 3. Fusion
        if vision_override:
            final_prob = vision_prob
        elif vision_prob is not None:
            final_prob = self.TABULAR_WEIGHT * tabular_prob + self.VISION_WEIGHT * vision_prob
        else:
            final_prob = tabular_prob

        final_prob = float(np.clip(final_prob, 0.01, 0.99))
        level      = _risk_bucket(final_prob, self.loader.threshold)

        # 4. Spoken alert
        spoken = self._spoken_alert(final_prob, level, raw_features, vision_features)

        return {
            "probability":      round(final_prob, 4),
            "risk_level":       level,
            "color":            RISK_COLORS[level],
            "message":          self._short_message(level),
            "spoken_alert":     spoken,
            "components": {
                "tabular_prob": round(tabular_prob, 4),
                "vision_prob":  round(vision_prob, 4) if vision_prob is not None else None,
                "vision_override": vision_override,
                "model_version": self.loader.version,
            },
        }

    def _short_message(self, level: str) -> str:
        return {
            "VERY_HIGH": "Very high risk. Stop or replan your route.",
            "MODERATE":  "Moderate risk. Proceed with caution.",
            "LOW":       "Low risk. Stay alert.",
        }[level]

    def _spoken_alert(self, prob: float, level: str, raw: dict, vision: Optional[dict]) -> str:
        pct  = round(prob * 100)
        base = f"{level.replace('_',' ').lower()} risk, {pct} percent. "

        reasons = []
        if raw.get("is_night"):
            reasons.append("night-time")
        if raw.get("_lighting") == "Dark":
            reasons.append("poor lighting")
        w = raw.get("_weather","")
        if w in ("Stormy","Foggy","Snowy","Rainy"):
            reasons.append(f"{w.lower()} weather")
        if raw.get("is_high_speed"):
            reasons.append(f"speed limit {int(raw.get('speed_limit_kmh',0))} kilometres per hour")
        if raw.get("_road_type") == "National Highway":
            reasons.append("national highway")

        if vision:
            vf = vision
            if vf.get("imminent_collision_risk"):
                return "STOP. Vehicle extremely close. Do not move."
            if vf.get("vehicles_in_danger_zone", 0):
                reasons.insert(0, f"{vf['vehicles_in_danger_zone']} vehicle close ahead")
            if vf.get("traffic_light_state") == "red":
                reasons.insert(0, "red traffic light")

        r_str = ", ".join(reasons) if reasons else "current conditions"
        return base + f"Hazards: {r_str}. " + self._short_message(level)
