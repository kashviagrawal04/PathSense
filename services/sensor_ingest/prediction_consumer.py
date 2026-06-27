# -*- coding: utf-8 -*-
"""
PathSense v2 — Prediction Consumer (Kafka Streams worker)
Consumes `sensor-events`, runs the v2 LightGBM model,
publishes to `risk-predictions`, and triggers alerts when threshold crossed.

Runs as a long-lived background worker (separate Docker container).
Can be scaled horizontally — Kafka partitions handle load distribution.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
TOPIC_SENSOR = "sensor-events"
TOPIC_PREDICTIONS = "risk-predictions"
TOPIC_ALERTS = "alert-triggers"
CONSUMER_GROUP = "prediction-service-group"

# Alert thresholds
ALERT_PROB_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.70"))
ALERT_COOLDOWN_SEC = int(os.getenv("ALERT_COOLDOWN_SEC", "60"))

# Path to ML artifacts
ML_ROOT = Path(__file__).resolve().parent.parent.parent / "ml"
sys.path.insert(0, str(ML_ROOT))


# ── Redis client (for prediction caching and alert cooldowns) ─────────────────

_redis = None


def get_redis():
    global _redis
    if _redis is None:
        try:
            import redis
            _redis = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
            _redis.ping()
            logger.info("Redis connected at %s", REDIS_URL)
        except Exception as e:
            logger.warning("Redis unavailable: %s", e)
            _redis = None
    return _redis


def redis_get(key: str):
    r = get_redis()
    return r.get(key) if r else None


def redis_set(key: str, value: str, ttl: int = 300):
    r = get_redis()
    if r:
        try:
            r.setex(key, ttl, value)
        except Exception:
            pass


def redis_getset_nx(key: str, value: str, ttl: int = 60) -> bool:
    """Set key if not exists. Returns True if we set it (not in cooldown)."""
    r = get_redis()
    if r:
        result = r.set(key, value, nx=True, ex=ttl)
        return result is True
    return True  # if no Redis, never block alerts


# ── Prediction cache key ──────────────────────────────────────────────────────

def cache_key(lat: float, lon: float, precision: int = 3) -> str:
    """
    Cache predictions at ~111m grid cells (precision=3).
    Same GPS tile = same prediction, avoid redundant inference.
    """
    return f"pred:{round(lat, precision)}:{round(lon, precision)}"


# ── Main consumer loop ────────────────────────────────────────────────────────

def run():
    # Lazy-load predictor (big model, load once)
    from predictor_v2 import RiskPredictor
    predictor = RiskPredictor(ML_ROOT / "artifacts")
    logger.info("PathSense Prediction Consumer ready (model v%s)", predictor.version)

    from kafka import KafkaConsumer, KafkaProducer

    consumer = KafkaConsumer(
        TOPIC_SENSOR,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=CONSUMER_GROUP,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=1000,
        max_poll_records=50,
        fetch_max_wait_ms=100,  # low latency
    )

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks=1,
        compression_type="gzip",
        linger_ms=2,
    )

    logger.info("Consuming from '%s' → publishing to '%s'", TOPIC_SENSOR, TOPIC_PREDICTIONS)
    processed = 0

    try:
        while True:
            try:
                records = consumer.poll(timeout_ms=500)
            except Exception as e:
                logger.error("Poll error: %s", e)
                time.sleep(2)
                continue

            for tp, messages in records.items():
                for msg in messages:
                    event = msg.value
                    user_id = event.get("user_id", "unknown")
                    lat = event.get("lat")
                    lon = event.get("lon")

                    if lat is None or lon is None:
                        continue

                    # ── Check Redis cache first ───────────────────────────────
                    ck = cache_key(lat, lon)
                    cached = redis_get(ck)

                    if cached:
                        result = json.loads(cached)
                        result["from_cache"] = True
                    else:
                        try:
                            result = predictor.predict_from_gps(
                                lat=lat,
                                lon=lon,
                                speed_kmh=event.get("speed_kmh", 0),
                                gps_accuracy_m=event.get("gps_accuracy_m", 5),
                                heading_change_deg=event.get("heading_change_deg", 0),
                                num_vehicles=event.get("num_vehicles_observed", 1),
                                road_condition=event.get("road_condition", "Dry"),
                                traffic_control=event.get("traffic_control", "Signals"),
                            )
                            # Remove enriched_features from cache (too large)
                            cacheable = {k: v for k, v in result.items() if k != "enriched_features"}
                            redis_set(ck, json.dumps(cacheable), ttl=120)
                            result["from_cache"] = False
                        except Exception as e:
                            logger.error("Prediction failed for user %s: %s", user_id, e)
                            continue

                    # ── Publish prediction ────────────────────────────────────
                    prediction_event = {
                        "user_id": user_id,
                        "session_id": event.get("session_id"),
                        "lat": lat,
                        "lon": lon,
                        "probability": result["probability"],
                        "risk_level": result["risk_level"],
                        "message": result["message"],
                        "color": result["color"],
                        "from_cache": result["from_cache"],
                        "model_version": result.get("model_version", "?"),
                        "predicted_at": datetime.utcnow().isoformat(),
                        "sensor_timestamp": event.get("timestamp"),
                    }
                    producer.send(TOPIC_PREDICTIONS, key=user_id, value=prediction_event)

                    # ── Trigger alert if above threshold ──────────────────────
                    prob = result["probability"]
                    if prob >= ALERT_PROB_THRESHOLD:
                        # Cooldown prevents alert spam (once per minute per user)
                        cooldown_key = f"alert_cooldown:{user_id}"
                        if redis_getset_nx(cooldown_key, "1", ttl=ALERT_COOLDOWN_SEC):
                            alert_event = {
                                "user_id": user_id,
                                "session_id": event.get("session_id"),
                                "risk_level": result["risk_level"],
                                "probability": prob,
                                "lat": lat,
                                "lon": lon,
                                "message": result["message"],
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                            producer.send(TOPIC_ALERTS, key=user_id, value=alert_event)
                            logger.warning(
                                "ALERT triggered for %s | prob=%.3f | (%s, %s)",
                                user_id, prob, lat, lon,
                            )

                    processed += 1
                    if processed % 100 == 0:
                        logger.info("Processed %d events", processed)

    except KeyboardInterrupt:
        logger.info("Shutting down prediction consumer...")
    finally:
        consumer.close()
        producer.flush()
        producer.close()


if __name__ == "__main__":
    # Retry loop: wait for Kafka to be ready on startup
    for attempt in range(10):
        try:
            run()
            break
        except Exception as e:
            logger.error("Startup attempt %d failed: %s", attempt + 1, e)
            time.sleep(5 * (attempt + 1))
