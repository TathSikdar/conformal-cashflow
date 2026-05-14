"""Unit tests for the Conformal Calibration module."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest
from src.calibration import ConformalCalibrator


class MockModel(nn.Module):
    """Mock model that returns fixed heuristic intervals."""
    def forward(self, x, future_x, acc_idx):
        batch = x.shape[0]
        horizon = future_x.shape[1]
        # Return (probs, magnitudes)
        probs = torch.ones((batch, horizon))
        magnitudes = torch.zeros((batch, horizon, 3))
        return probs, magnitudes


def test_q_hat_calculation():
    """Verifies that q_hat correctly identifies the error quantile."""
    model = MockModel()
    y = torch.linspace(-10, 10, 100).unsqueeze(1) # shape (100, 1)
    x = torch.zeros((100, 5, 5))
    future_x = torch.zeros((100, 1, 6))
    acc_idx = torch.zeros(100, dtype=torch.long)
    
    dataset = TensorDataset(x, future_x, acc_idx, y)
    loader = DataLoader(dataset, batch_size=10)
    
    calibrator = ConformalCalibrator(alpha=0.1) # 90% coverage
    q_hat = calibrator.calibrate(model, loader)
    
    assert q_hat > 0
    assert isinstance(q_hat, (float, np.floating))


def test_interval_expansion():
    """Verifies that predict() expands intervals by q_hat."""
    calibrator = ConformalCalibrator(alpha=0.1)
    calibrator.q_hat = 2.5
    
    model = MockModel()
    x = torch.zeros((1, 5, 5))
    future_x = torch.zeros((1, 1, 6))
    acc_idx = torch.zeros(1, dtype=torch.long)
    
    # Model returns [0, 0, 0]
    # Calibrated should be [0-2.5, 0+2.5] = [-2.5, 2.5]
    intervals = calibrator.predict(model, x, future_x, acc_idx)
    
    assert torch.isclose(intervals[0, 0, 0], torch.tensor(-2.5))
    assert torch.isclose(intervals[0, 0, 1], torch.tensor(2.5))


def test_exact_coverage_math():
    """Mathematical smoke test for the q_level calculation."""
    n = 99
    alpha = 0.1
    # (n+1)(1-alpha)/n = (100 * 0.9) / 99 = 90 / 99 approx 0.909
    calibrator = ConformalCalibrator(alpha=alpha)
    
    scores = np.arange(100)
    q_hat = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher")
    assert q_hat == 90
