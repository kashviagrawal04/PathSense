import pytest
from fastapi.testclient import TestClient
import os
import sys

# Ensure api module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set dummy env vars for tests
os.environ["API_KEY"] = "test-api-key"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"
os.environ["TWILIO_ACCOUNT_SID"] = "test"
os.environ["TWILIO_AUTH_TOKEN"] = "test"
os.environ["TWILIO_FROM_NUMBER"] = "test"
os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"

from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json().get("status") == "ok"

def test_cors_origin():
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

def test_predict_requires_auth():
    # Missing API Key
    response = client.post("/predict", json={})
    assert response.status_code in [401, 403, 422]  # Unauthenticated or missing body

def test_predict_with_invalid_auth():
    response = client.post("/predict", headers={"x-api-key": "invalid-key"}, json={})
    assert response.status_code in [401, 403, 422]
