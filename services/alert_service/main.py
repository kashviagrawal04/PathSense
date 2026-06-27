# -*- coding: utf-8 -*-
"""
PathSense v2 — Alert Service
Consumes `alert-triggers` Kafka topic and dispatches:
  - Twilio SMS to emergency contacts
  - Push notifications (FCM)
  - Caretaker web dashboard updates (WebSocket broadcast)

Runs as a standalone Kafka consumer worker.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC_ALERTS = "alert-triggers"
CONSUMER_GROUP = "alert-service-group"

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")

import sys

# DB connection for looking up user emergency contacts
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.error("CRITICAL: DATABASE_URL environment variable is not set. Refusing to start in insecure mode.")
    sys.exit(1)


def get_emergency_contacts(user_id: str) -> list[str]:
    """
    Fetch registered emergency contact numbers for a user from PostgreSQL.
    Returns phone numbers in E.164 format.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT phone_number FROM emergency_contacts WHERE user_id = %s AND active = TRUE",
                (user_id,)
            )
            rows = cur.fetchall()
        conn.close()
        return [row[0] for row in rows]
    except Exception as e:
        logger.warning("DB contact lookup failed for %s: %s", user_id, e)
        return []


def send_sms(to_number: str, message: str) -> bool:
    """Send SMS via Twilio. Returns True on success."""
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER]):
        logger.warning("Twilio not configured — SMS skipped")
        return False
    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=TWILIO_FROM_NUMBER,
            to=to_number,
        )
        logger.info("SMS sent to %s | SID: %s", to_number[:8] + "XXXX", msg.sid)
        return True
    except Exception as e:
        logger.error("SMS failed to %s: %s", to_number[:8] + "XXXX", e)
        return False


def format_sos_message(event: dict) -> str:
    lat = event.get("lat")
    lon = event.get("lon")
    prob = round(event.get("probability", 0) * 100)
    maps_url = f"https://maps.google.com/?q={lat},{lon}" if lat and lon else ""
    return (
        f"PathSense ALERT: Your contact may be in danger!\n"
        f"Risk level: {event.get('risk_level', 'HIGH')} ({prob}%)\n"
        f"Message: {event.get('message', '')}\n"
        f"{('Location: ' + maps_url) if maps_url else ''}\n"
        f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
    )


def process_alert(event: dict):
    user_id = event.get("user_id")
    risk_level = event.get("risk_level", "UNKNOWN")
    prob = event.get("probability", 0)

    logger.warning(
        "Processing alert | user=%s | level=%s | prob=%.3f",
        user_id, risk_level, prob,
    )

    # Only send notifications for VERY_HIGH risk
    if risk_level != "VERY_HIGH":
        logger.info("Skipping notification for level=%s", risk_level)
        return

    contacts = get_emergency_contacts(user_id)
    if not contacts:
        logger.info("No emergency contacts for user %s", user_id)
        return

    message = format_sos_message(event)
    for contact in contacts:
        send_sms(contact, message)


def run():
    from kafka import KafkaConsumer

    consumer = KafkaConsumer(
        TOPIC_ALERTS,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=CONSUMER_GROUP,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    logger.info("Alert Service consuming from '%s'", TOPIC_ALERTS)

    for msg in consumer:
        try:
            process_alert(msg.value)
        except Exception as e:
            logger.error("Alert processing failed: %s", e)


if __name__ == "__main__":
    for attempt in range(10):
        try:
            run()
            break
        except Exception as e:
            logger.error("Startup attempt %d failed: %s", attempt + 1, e)
            time.sleep(5 * (attempt + 1))
