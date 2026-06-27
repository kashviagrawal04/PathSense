# -*- coding: utf-8 -*-
"""
PathSense v2 — Feature Engineering
Replaces static CSV labels with real-world enriched features.

Sources:
  - OpenWeatherMap API  (live weather keyed to GPS)
  - Nominatim / OSM     (road type, speed limit from lat/lon)
  - Computed temporal   (continuous hour, is_weekend, is_rush_hour)
  - Device sensors      (GPS accuracy, movement speed from phone)
"""
from __future__ import annotations

import os
import time
import math
import hashlib
import logging
import requests
from datetime import datetime
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OWM_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
OWM_URL = "https://api.openweathermap.org/data/2.5/weather"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Cache TTL: weather changes slowly, road data almost never
_weather_cache: dict[str, tuple[dict, float]] = {}
_road_cache: dict[str, dict] = {}
WEATHER_TTL = 600  # 10 minutes


# ── Weather enrichment ────────────────────────────────────────────────────────

def _cache_key(lat: float, lon: float, precision: int = 2) -> str:
    return f"{round(lat, precision)},{round(lon, precision)}"


def get_weather_features(lat: float, lon: float) -> dict:
    """
    Fetch live weather from OpenWeatherMap and return structured features.
    Falls back to neutral defaults if API is unavailable (no key / offline).
    """
    key = _cache_key(lat, lon)
    now = time.time()

    # Return cached result if fresh
    if key in _weather_cache:
        data, ts = _weather_cache[key]
        if now - ts < WEATHER_TTL:
            return data

    if not OWM_API_KEY:
        logger.warning("No OPENWEATHERMAP_API_KEY set — using default weather features")
        return _default_weather()

    try:
        resp = requests.get(
            OWM_URL,
            params={"lat": lat, "lon": lon, "appid": OWM_API_KEY, "units": "metric"},
            timeout=3,
        )
        resp.raise_for_status()
        raw = resp.json()
        features = _parse_owm(raw)
        _weather_cache[key] = (features, now)
        return features
    except Exception as exc:
        logger.error("OWM fetch failed: %s", exc)
        return _default_weather()


def _parse_owm(raw: dict) -> dict:
    weather_id = raw.get("weather", [{}])[0].get("id", 800)
    wind_speed = raw.get("wind", {}).get("speed", 0)
    visibility = raw.get("visibility", 10000)
    rain_1h = raw.get("rain", {}).get("1h", 0)
    snow_1h = raw.get("snow", {}).get("1h", 0)
    temp = raw.get("main", {}).get("temp", 20)

    # Map OWM weather ID to our risk-relevant categories
    if weather_id >= 200 and weather_id < 300:
        condition = "Stormy"
    elif weather_id >= 300 and weather_id < 600:
        condition = "Rainy"
    elif weather_id >= 600 and weather_id < 700:
        condition = "Snowy"
    elif weather_id in (741,):  # fog
        condition = "Foggy"
    elif weather_id in (721, 731, 751, 761):  # haze/dust
        condition = "Hazy"
    else:
        condition = "Clear"

    # Quantitative risk signals (new features v2 adds)
    low_visibility = int(visibility < 1000)
    high_wind = int(wind_speed > 10)
    precipitation_mm = rain_1h + snow_1h

    return {
        "Weather Conditions": condition,
        "visibility_m": visibility,
        "wind_speed_ms": wind_speed,
        "precipitation_mm": precipitation_mm,
        "temperature_c": temp,
        "low_visibility_flag": low_visibility,
        "high_wind_flag": high_wind,
    }


def _default_weather() -> dict:
    return {
        "Weather Conditions": "Clear",
        "visibility_m": 10000,
        "wind_speed_ms": 0.0,
        "precipitation_mm": 0.0,
        "temperature_c": 20.0,
        "low_visibility_flag": 0,
        "high_wind_flag": 0,
    }


# ── Road enrichment via OSM Overpass ─────────────────────────────────────────

def get_road_features(lat: float, lon: float, radius_m: int = 30) -> dict:
    """
    Query Overpass API for road features within radius_m of the point.
    Returns road type, speed limit, crosswalk proximity, and lane count.
    Falls back to defaults if offline.
    """
    key = f"{round(lat, 4)},{round(lon, 4)}"
    if key in _road_cache:
        return _road_cache[key]

    query = f"""
    [out:json][timeout:5];
    (
      way(around:{radius_m},{lat},{lon})[highway];
      node(around:20,{lat},{lon})[highway=crossing];
    );
    out tags;
    """

    try:
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=6,
            headers={"User-Agent": "PathSense/2.0 (pedestrian safety app)"},
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        features = _parse_osm(elements)
        _road_cache[key] = features
        return features
    except Exception as exc:
        logger.warning("OSM Overpass failed: %s", exc)
        return _default_road()


def _parse_osm(elements: list) -> dict:
    road_type = "Urban Road"
    speed_limit = 50
    crosswalk_nearby = 0
    lanes = 2

    highway_map = {
        "motorway": ("National Highway", 100),
        "trunk": ("National Highway", 80),
        "primary": ("State Highway", 70),
        "secondary": ("State Highway", 60),
        "tertiary": ("Urban Road", 50),
        "residential": ("Urban Road", 30),
        "living_street": ("Urban Road", 20),
        "unclassified": ("Village Road", 40),
        "service": ("Village Road", 20),
        "path": ("Village Road", 10),
        "footway": ("Village Road", 10),
    }

    for el in elements:
        tags = el.get("tags", {})
        hw = tags.get("highway", "")

        if hw == "crossing":
            crosswalk_nearby = 1
            continue

        if hw in highway_map:
            road_type, default_speed = highway_map[hw]
            try:
                speed_limit = int(str(tags.get("maxspeed", default_speed)).split()[0])
            except (ValueError, TypeError):
                speed_limit = default_speed
            try:
                lanes = int(tags.get("lanes", 2))
            except (ValueError, TypeError):
                lanes = 2
            break

    return {
        "Road Type": road_type,
        "Speed Limit (km/h)": speed_limit,
        "crosswalk_nearby": crosswalk_nearby,
        "road_lanes": lanes,
    }


def _default_road() -> dict:
    return {
        "Road Type": "Urban Road",
        "Speed Limit (km/h)": 50,
        "crosswalk_nearby": 0,
        "road_lanes": 2,
    }


# ── Temporal features ─────────────────────────────────────────────────────────

def get_temporal_features(dt: Optional[datetime] = None) -> dict:
    """
    Extract rich temporal features from a datetime.
    Continuous hour-of-day is much more informative than 'Morning/Evening' buckets.
    """
    if dt is None:
        dt = datetime.now()

    hour = dt.hour
    minute = dt.minute
    hour_continuous = hour + minute / 60.0

    # Cyclical encoding — hour 23 should be close to hour 0
    hour_sin = math.sin(2 * math.pi * hour_continuous / 24)
    hour_cos = math.cos(2 * math.pi * hour_continuous / 24)

    # Coarse bucket (kept for backward compat with v1 model)
    if 5 <= hour < 12:
        time_cat = "Morning"
    elif 12 <= hour < 17:
        time_cat = "Afternoon"
    elif 17 <= hour < 21:
        time_cat = "Evening"
    else:
        time_cat = "Night"

    is_weekend = int(dt.weekday() >= 5)
    is_rush_hour = int((7 <= hour <= 9) or (17 <= hour <= 19))
    day_of_week = dt.strftime("%A")

    # Month seasonality (monsoon months are higher risk in South Asia)
    month = dt.month
    is_monsoon = int(6 <= month <= 9)

    return {
        "Time_Category": time_cat,
        "Day of Week": day_of_week,
        "hour_of_day": hour_continuous,
        "hour_sin": round(hour_sin, 4),
        "hour_cos": round(hour_cos, 4),
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour,
        "is_monsoon_month": is_monsoon,
        "month": month,
    }


# ── Movement / device features ────────────────────────────────────────────────

def get_movement_features(
    speed_kmh: float = 0.0,
    gps_accuracy_m: float = 5.0,
    heading_change_deg: float = 0.0,
) -> dict:
    """
    Features derived from the user's phone sensors.
    speed_kmh: pedestrian walking speed (0-10 normal, >10 unusual)
    gps_accuracy_m: GPS fix quality — worse = more uncertainty
    heading_change_deg: sudden direction change = possible hazard avoidance
    """
    is_stationary = int(speed_kmh < 1.0)
    is_running = int(speed_kmh > 7.0)
    poor_gps = int(gps_accuracy_m > 20.0)
    sudden_turn = int(abs(heading_change_deg) > 45)

    return {
        "pedestrian_speed_kmh": speed_kmh,
        "gps_accuracy_m": gps_accuracy_m,
        "is_stationary": is_stationary,
        "is_running": is_running,
        "poor_gps_signal": poor_gps,
        "sudden_direction_change": sudden_turn,
    }


# ── Lighting estimation (sun position) ───────────────────────────────────────

def estimate_lighting(lat: float, lon: float, dt: Optional[datetime] = None) -> dict:
    """
    Estimate lighting condition from sun elevation angle.
    Much more accurate than the string 'Daylight'/'Dark' from v1.
    Uses a simplified solar position calculation.
    """
    if dt is None:
        dt = datetime.now()

    # Day of year
    doy = dt.timetuple().tm_yday
    hour_utc = dt.hour + dt.minute / 60.0

    # Solar declination
    decl = math.radians(23.45 * math.sin(math.radians(360 / 365 * (doy - 81))))
    lat_r = math.radians(lat)

    # Hour angle (rough — ignores timezone properly, use for estimation only)
    ha = math.radians((hour_utc - 12) * 15 + lon)
    sin_alt = (
        math.sin(lat_r) * math.sin(decl)
        + math.cos(lat_r) * math.cos(decl) * math.cos(ha)
    )
    sun_altitude_deg = math.degrees(math.asin(max(-1, min(1, sin_alt))))

    if sun_altitude_deg > 6:
        lighting = "Daylight"
    elif sun_altitude_deg > 0:
        lighting = "Dusk" if dt.hour > 12 else "Dawn"
    elif sun_altitude_deg > -6:
        lighting = "Dusk" if dt.hour > 12 else "Dawn"
    else:
        lighting = "Dark"

    return {
        "Lighting Conditions": lighting,
        "sun_altitude_deg": round(sun_altitude_deg, 2),
        "is_dark": int(lighting == "Dark"),
    }


# ── Master enrichment function ────────────────────────────────────────────────

def enrich_request(
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
    Full enrichment pipeline. Takes GPS + sensor inputs, returns a
    feature dict ready for the v2 model.
    """
    weather = get_weather_features(lat, lon)
    road = get_road_features(lat, lon)
    temporal = get_temporal_features(dt)
    lighting = estimate_lighting(lat, lon, dt)
    movement = get_movement_features(speed_kmh, gps_accuracy_m, heading_change_deg)

    features = {
        # V1 compatible keys (for backward compat during transition)
        "Weather Conditions": weather["Weather Conditions"],
        "Lighting Conditions": lighting["Lighting Conditions"],
        "Road Type": road["Road Type"],
        "Road Condition": road_condition,
        "Speed Limit (km/h)": road["Speed Limit (km/h)"],
        "Time_Category": temporal["Time_Category"],
        "Day of Week": temporal["Day of Week"],
        "Number of Vehicles Involved": num_vehicles,
        "Traffic Control Presence": traffic_control,

        # V2 new features (used by the enhanced model)
        "visibility_m": weather["visibility_m"],
        "wind_speed_ms": weather["wind_speed_ms"],
        "precipitation_mm": weather["precipitation_mm"],
        "temperature_c": weather["temperature_c"],
        "low_visibility_flag": weather["low_visibility_flag"],
        "high_wind_flag": weather["high_wind_flag"],
        "hour_of_day": temporal["hour_of_day"],
        "hour_sin": temporal["hour_sin"],
        "hour_cos": temporal["hour_cos"],
        "is_weekend": temporal["is_weekend"],
        "is_rush_hour": temporal["is_rush_hour"],
        "is_monsoon_month": temporal["is_monsoon_month"],
        "month": temporal["month"],
        "sun_altitude_deg": lighting["sun_altitude_deg"],
        "is_dark": lighting["is_dark"],
        "crosswalk_nearby": road["crosswalk_nearby"],
        "road_lanes": road["road_lanes"],
        "pedestrian_speed_kmh": movement["pedestrian_speed_kmh"],
        "gps_accuracy_m": movement["gps_accuracy_m"],
        "is_stationary": movement["is_stationary"],
        "is_running": movement["is_running"],
        "poor_gps_signal": movement["poor_gps_signal"],
        "sudden_direction_change": movement["sudden_direction_change"],

        # Metadata
        "_lat": lat,
        "_lon": lon,
    }

    return features
