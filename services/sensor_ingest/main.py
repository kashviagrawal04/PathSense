# -*- coding: utf-8 -*-
"""
PathSense v2 — Sensor Ingest Service
Kafka producer that receives GPS/sensor events from the mobile app
and publishes them to the `sensor-events` topic.

Runs as a FastAPI microservice (separate Docker container).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC_SENSOR = "sensor-events"
TOPIC_ALERTS = "alert-triggers"

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.error("CRITICAL: API_KEY environment variable is not set. Refusing to start in insecure mode.")
    sys.exit(1)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API KEY")
    return api_key_header

origins_str = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in origins_str.split(",") if o.strip()]
if not ALLOWED_ORIGINS:
    logger.warning("No ALLOWED_ORIGINS set. CORS will block all browser requests.")

app = FastAPI(title="PathSense Sensor Ingest", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])

# ── Kafka producer (lazy-init so service starts even if Kafka is warming up) ──

_producer = None


def get_producer():
    global _producer
    if _producer is None:
        try:
            from kafka import KafkaProducer
            _producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",              # wait for all replicas to ack
                retries=3,
                compression_type="gzip",
                linger_ms=5,             # small batching window
                request_timeout_ms=5000,
            )
            logger.info("Kafka producer connected to %s", KAFKA_BOOTSTRAP)
        except Exception as e:
            logger.error("Kafka unavailable: %s — running in degraded mode", e)
            _producer = None
    return _producer


# ── Schemas ───────────────────────────────────────────────────────────────────

class SensorEvent(BaseModel):
    user_id: str = Field(..., description="Anonymized user identifier")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    gps_accuracy_m: float = Field(default=5.0, ge=0)
    speed_kmh: float = Field(default=0.0, ge=0)
    heading_deg: float = Field(default=0.0, ge=0, le=360)
    heading_change_deg: float = Field(default=0.0)
    road_condition: str = Field(default="Dry")
    traffic_control: str = Field(default="Signals")
    num_vehicles_observed: int = Field(default=1, ge=0)
    timestamp: Optional[str] = None  # ISO8601, defaults to now


class AlertEvent(BaseModel):
    user_id: str
    session_id: str
    risk_level: str
    probability: float
    lat: float
    lon: float
    message: str
    timestamp: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", dependencies=[Depends(get_api_key)])
def health():
    producer = get_producer()
    return {
        "status": "ok",
        "kafka_connected": producer is not None,
        "kafka_bootstrap": KAFKA_BOOTSTRAP,
    }


@app.post("/ingest/sensor", status_code=202, dependencies=[Depends(get_api_key)])
def ingest_sensor(event: SensorEvent):
    """
    Receive a sensor update from the mobile app.
    Publishes to Kafka `sensor-events` topic keyed by user_id
    (same user always goes to same partition → ordered processing).
    """
    payload = event.model_dump()
    payload["timestamp"] = payload["timestamp"] or datetime.utcnow().isoformat()
    payload["ingested_at"] = datetime.utcnow().isoformat()

    producer = get_producer()
    if producer:
        try:
            future = producer.send(TOPIC_SENSOR, key=event.user_id, value=payload)
            future.get(timeout=3)  # confirm delivery
        except Exception as e:
            logger.error("Failed to publish to Kafka: %s", e)
            raise HTTPException(status_code=503, detail="Event queue unavailable")
    else:
        # Degraded mode: log event but don't fail the request
        logger.warning("Kafka unavailable — event dropped: %s", payload)

    return {"status": "queued", "session_id": event.session_id}


@app.post("/ingest/alert", status_code=202, dependencies=[Depends(get_api_key)])
def ingest_alert(event: AlertEvent):
    """
    Publish a risk alert event to `alert-triggers` topic.
    The Alert Service consumes this to fire Twilio SMS / push notifications.
    """
    payload = event.model_dump()
    producer = get_producer()
    if producer:
        try:
            producer.send(TOPIC_ALERTS, key=event.user_id, value=payload)
        except Exception as e:
            logger.error("Failed to publish alert: %s", e)

    return {"status": "alert_queued"}


# ── WebSocket for real-time mobile streaming ──────────────────────────────────

class ConnectionManager:
    """Manages active WebSocket connections per user."""

    def __init__(self):
        self.connections: dict[str, WebSocket] = {}

    async def connect(self, user_id: str, ws: WebSocket):
        await ws.accept()
        self.connections[user_id] = ws
        logger.info("WS connected: %s (total: %d)", user_id, len(self.connections))

    def disconnect(self, user_id: str):
        self.connections.pop(user_id, None)

    async def send_risk_update(self, user_id: str, data: dict):
        ws = self.connections.get(user_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(user_id)


manager = ConnectionManager()


@app.websocket("/ws/{user_id}")
async def websocket_sensor_stream(ws: WebSocket, user_id: str):
    """
    Mobile app connects here and sends sensor readings continuously.
    Each message is published to Kafka.
    Risk predictions from the Prediction Service are pushed back via this socket.

    Message format from mobile:
      {"lat": 28.6, "lon": 77.2, "speed_kmh": 3.5, "gps_accuracy_m": 4.2, ...}

    Risk update pushed to mobile:
      {"risk_level": "MODERATE", "probability": 0.62, "message": "..."}
    """
    client_api_key = ws.headers.get("X-API-Key") or ws.query_params.get("api_key")
    if client_api_key != API_KEY:
        await ws.close(code=1008)
        return

    await manager.connect(user_id, ws)
    session_id = str(uuid.uuid4())
    try:
        while True:
            data = await ws.receive_json()
            event = SensorEvent(
                user_id=user_id,
                session_id=session_id,
                **data,
            )
            payload = event.model_dump()
            payload["timestamp"] = datetime.utcnow().isoformat()

            producer = get_producer()
            if producer:
                producer.send(TOPIC_SENSOR, key=user_id, value=payload)

    except WebSocketDisconnect:
        manager.disconnect(user_id)
        logger.info("WS disconnected: %s", user_id)
