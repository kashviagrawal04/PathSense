import pytest
from fastapi.testclient import TestClient
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["API_KEY"] = "valid-vision-key"
os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"

from services.vision.main import app

client = TestClient(app)

def test_vision_health():
    response = client.get("/health", headers={"x-api-key": "valid-vision-key"})
    assert response.status_code == 200
    assert response.json().get("status") == "ok"

def test_vision_auth_failure():
    response = client.get("/health")
    assert response.status_code in [401, 403]

def test_demo_alert():
    response = client.get("/demo/alert/danger", headers={"x-api-key": "valid-vision-key"})
    assert response.status_code == 200
    assert response.headers.get("content-type") == "audio/wav"
    assert "Danger!" in response.headers.get("x-alert-text", "")

def test_score_endpoint():
    payload = {
        "features": {
            "vehicles_in_danger_zone": 2,
            "vehicles_in_warn_zone": 0,
            "closest_vehicle_m": 4.5,
            "traffic_light_state": "green",
            "alert_severity": "danger",
            "alert_text": "Danger! 2 vehicles within 5 metres.",
            "total_objects": 2,
            "imminent_collision_risk": 0,
            "unsafe_to_cross": 1
        }
    }
    response = client.post("/score", headers={"x-api-key": "valid-vision-key"}, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["alert_severity"] == "danger"
    assert data["vision_risk_score"] == 0.8
