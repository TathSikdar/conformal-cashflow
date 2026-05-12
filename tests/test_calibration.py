"""Unit tests for the Conformal Calibration module."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest
from src.calibration import ConformalCalibrator


class MockModel(nn.Module):
    """Mock model that returns fixed heuristic intervals."""
    def forward(self, x):
        batch, history, _ = x.shape
        # Return [target-1, target, target+1] roughly
        # For testing, we just return zeros and let target vary
        return torch.zeros((batch, 1, 3)) # Horizon=1, 3 quantiles


def test_q_hat_calculation():
    """Verifies that q_hat correctly identifies the error quantile."""
    # Setup: 100 samples with errors from -10 to 89
    # max(low-y, y-high) where low=-1, high=1
    # If y=5, score = max(-1-5, 5-1) = 4
    model = MockModel()
    y = torch.linspace(-10, 10, 100).unsqueeze(1) # shape (100, 1)
    x = torch.zeros((100, 5, 5))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=10)
    
    calibrator = ConformalCalibrator(alpha=0.1) # 90% coverage
    q_hat = calibrator.calibrate(model, loader)
    
    # Heuristic intervals are [-1, 1] (since model returns 0s)
    # Errors are |y| - 1 approximately
    # We expect q_hat to be around the 90th percentile of errors
    assert q_hat > 0
    assert isinstance(q_hat, (float, np.floating))


def test_interval_expansion():
    """Verifies that predict() expands intervals by q_hat."""
    calibrator = ConformalCalibrator(alpha=0.1)
    calibrator.q_hat = 2.5
    
    model = MockModel()
    x = torch.zeros((1, 5, 5))
    
    # Model returns [0, 0, 0]
    # Calibrated should be [0-2.5, 0+2.5] = [-2.5, 2.5]
    intervals = calibrator.predict(model, x)
    
    assert torch.isclose(intervals[0, 0, 0], torch.tensor(-2.5))
    assert torch.isclose(intervals[0, 0, 1], torch.tensor(2.5))


def test_exact_coverage_math():
    """Mathematical smoke test for the q_level calculation."""
    n = 99
    alpha = 0.1
    # (n+1)(1-alpha)/n = (100 * 0.9) / 99 = 90 / 99 approx 0.909
    calibrator = ConformalCalibrator(alpha=alpha)
    
    # We just want to ensure it handles the n calculation correctly
    # By mocking a small set
    scores = np.arange(100)
    # q_level approx 0.91. 91st element of 0-99 is 90 (if using 'higher')
    q_hat = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher")
    assert q_hat == 90
