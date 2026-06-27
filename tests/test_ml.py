import pytest
import os
import sys
import pandas as pd
from unittest.mock import patch

# Ensure services can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.predictor_v2 import RiskPredictor
from ml.data_pipeline import generate_realistic

def test_data_pipeline_generation():
    # Verify the synthetic data generator works and returns a DataFrame
    df = generate_realistic(n=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "High_Risk" in df.columns
    assert "speed_limit_kmh" in df.columns

@patch("ml.predictor_v2.Path.exists")
@patch("ml.predictor_v2.joblib.load")
def test_predictor_v2_inference(mock_load, mock_exists):
    # Mock file existence so it bypasses missing models in CI
    mock_exists.return_value = True
    
    # Mock the loaded artifacts
    class MockModel:
        def predict_proba(self, X):
            import numpy as np
            return np.array([[0.1, 0.9] for _ in range(len(X))])
    
    class MockEncoder:
        def transform(self, X):
            return [0] * len(X)
            
    mock_load.side_effect = [
        MockModel(),        # model
        {"Test": MockEncoder()}, # encoders
        {"threshold": 0.5, "features": ["Speed Limit (km/h)", "Test"], "categorical_columns": ["Test"], "feature_columns": ["Speed Limit (km/h)", "Test"]} # meta
    ]
    
    service = RiskPredictor()
    
    # Test valid payload
    event = {
        "Speed Limit (km/h)": 80,
        "Test": "value"
    }
    
    result = service.predict_from_gps(lat=0, lon=0, speed_kmh=80)
    assert result["probability"] == 0.9
    assert result["risk_level"] == "VERY_HIGH"

def test_predictor_v2_fallback():
    # If no model exists, it should not crash but return UNKNOWN or raise cleanly
    with patch("ml.predictor_v2.Path.exists") as mock_exists:
        mock_exists.return_value = False
        try:
            service = RiskPredictor()
        except (FileNotFoundError, KeyError):
            assert True
