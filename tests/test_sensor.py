import pytest
from fastapi.testclient import TestClient
import os
import sys
from unittest.mock import patch, MagicMock

# Ensure services can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["API_KEY"] = "valid-sensor-key"
os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"

from services.sensor_ingest.main import app

client = TestClient(app)

def test_sensor_health():
    response = client.get("/health", headers={"x-api-key": "valid-sensor-key"})
    assert response.status_code == 200

def test_sensor_ingest_auth_failure():
    response = client.post("/ingest/sensor", json={})
    assert response.status_code in [401, 403]
    assert "Not authenticated" in response.json().get("detail", "")

@patch("services.sensor_ingest.main.get_producer")
def test_sensor_ingest_success(mock_get_producer):
    # Mock the Kafka producer send method
    mock_producer = MagicMock()
    mock_producer.send.return_value = MagicMock()
    mock_get_producer.return_value = mock_producer
    
    payload = {
        "user_id": "user123",
        "session_id": "session_123",
        "lat": 35.0,
        "lon": -120.0,
        "gps_accuracy_m": 5.0,
        "speed_kmh": 20.0,
        "heading_deg": 180.0,
        "heading_change_deg": 0.0,
        "road_condition": "Dry",
        "traffic_control": "Signals",
        "num_vehicles_observed": 2,
        "timestamp": "2024-05-18T12:00:00Z"
    }
    response = client.post(
        "/ingest/sensor",
        headers={"x-api-key": "valid-sensor-key"},
        json=payload
    )
    
    assert response.status_code == 202
    assert response.json().get("status") == "queued"
    mock_producer.send.assert_called_once()
