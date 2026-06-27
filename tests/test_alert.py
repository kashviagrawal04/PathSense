import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Ensure services can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"
os.environ["TWILIO_ACCOUNT_SID"] = "test"
os.environ["TWILIO_AUTH_TOKEN"] = "test"
os.environ["TWILIO_FROM_NUMBER"] = "test"

from services.alert_service import main as alert_service

@patch("services.alert_service.main.get_emergency_contacts")
@patch("services.alert_service.main.send_sms")
def test_process_alert_very_high_risk(mock_send_sms, mock_get_contacts):
    # Should send SMS when risk is VERY_HIGH
    mock_get_contacts.return_value = ["+123456789"]
    
    event = {
        "user_id": "user123",
        "risk_level": "VERY_HIGH",
        "probability": 0.95,
        "lat": 35.0,
        "lon": -120.0,
        "message": "Imminent collision"
    }
    
    alert_service.process_alert(event)
    
    mock_get_contacts.assert_called_once_with("user123")
    mock_send_sms.assert_called_once()
    assert "Imminent collision" in mock_send_sms.call_args[0][1]

@patch("services.alert_service.main.get_emergency_contacts")
@patch("services.alert_service.main.send_sms")
def test_process_alert_ignores_lower_risk(mock_send_sms, mock_get_contacts):
    # Should ignore MODERATE risk
    event = {
        "user_id": "user123",
        "risk_level": "MODERATE",
        "probability": 0.50
    }
    
    alert_service.process_alert(event)
    
    mock_get_contacts.assert_not_called()
    mock_send_sms.assert_not_called()

@patch("psycopg2.connect")
def test_get_emergency_contacts(mock_connect):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    
    mock_cursor.fetchall.return_value = [("+123456789",), ("+987654321",)]
    
    contacts = alert_service.get_emergency_contacts("user123")
    assert len(contacts) == 2
    assert contacts[0] == "+123456789"
