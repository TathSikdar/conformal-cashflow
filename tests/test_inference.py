"""Integration tests for the full Inference API pipeline."""

import pandas as pd
import torch
import pytest
from src.inference import CashFlowInferenceAgent
from src.model import ProbabilisticForecaster
from src.calibration import ConformalCalibrator


@pytest.fixture
def integration_setup():
    """Sets up a mock production environment."""
    # 1. Mock Model
    # Features: amount_log (target), day_sin, day_cos (exog)
    feature_cols = ["amount_log", "day_sin", "day_cos"]
    input_dim = len(feature_cols)
    hidden_dim = 16
    horizon = 7
    num_accounts = 5
    future_dim = 2 # day_sin, day_cos
    
    model = ProbabilisticForecaster(
        input_dim=input_dim, 
        num_accounts=num_accounts,
        hidden_dim=hidden_dim, 
        horizon=horizon,
        future_dim=future_dim
    )
    
    # 2. Mock Calibrator
    calibrator = ConformalCalibrator(alpha=0.1)
    calibrator.q_hat = 0.5 
    
    # 3. Account Map
    account_id_map = {1: 0}
    
    # 4. Agent
    agent = CashFlowInferenceAgent(
        model=model,
        calibrator=calibrator,
        feature_cols=feature_cols,
        account_id_map=account_id_map,
        sequence_length=60,
        horizon=horizon,
        exog_indices=[1, 2] # day_sin, day_cos
    )
    
    return agent, horizon


def test_full_pipeline_inference(integration_setup):
    """Verifies end-to-end flow from raw DataFrame to JSON output."""
    agent, horizon = integration_setup
    
    # Create mock raw data (Relational format)
    data = {
        "account_id": [1] * 80,
        "date": pd.date_range(start="1993-01-01", periods=80, freq="D"),
        "amount": [100.0] * 80
    }
    raw_df = pd.DataFrame(data)
    
    # Run prediction
    result = agent.predict_calibrated(raw_df)
    
    # Assertions
    assert "forecast_horizon" in result
    assert result["forecast_horizon"] == horizon
    assert len(result["predictions"]) == horizon
    
    first_step = result["predictions"][0]
    assert isinstance(first_step["median"], float)
    assert isinstance(first_step["lower_90"], float)
    assert isinstance(first_step["upper_90"], float)


def test_json_serialization(integration_setup):
    """Verifies JSON output format."""
    agent, _ = integration_setup
    mock_result = {
        "forecast_horizon": 2,
        "predictions": [
            {"step": 1, "median": 10.0, "lower_90": 8.0, "upper_90": 12.0}
        ]
    }
    json_str = agent.to_json(mock_result)
    assert '"step": 1' in json_str
    assert '"median": 10.0' in json_str
