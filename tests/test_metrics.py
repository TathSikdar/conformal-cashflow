"""Unit tests for the Evaluation Metrics module."""

import torch
import pytest
from src.metrics import ProbabilisticEvaluator


def test_picp_calculation():
    """Verifies PICP correctly identifies coverage proportion."""
    # y_true: [10, 20, 30]
    # y_low:  [ 9, 15, 31]
    # y_high: [11, 25, 35]
    # Coverage: [In, In, Out] -> 2/3 = 0.666...
    y_true = torch.tensor([[10.0, 20.0, 30.0]])
    y_low = torch.tensor([[9.0, 15.0, 31.0]])
    y_high = torch.tensor([[11.0, 25.0, 35.0]])
    
    picp = ProbabilisticEvaluator.calculate_picp(y_true, y_low, y_high)
    assert pytest.approx(picp) == 2.0 / 3.0


def test_mpiw_calculation():
    """Verifies MPIW correctly identifies average interval width."""
    # widths: [2, 10, 4] -> Mean = 16 / 3 = 5.333...
    y_low = torch.tensor([[9.0, 15.0, 31.0]])
    y_high = torch.tensor([[11.0, 25.0, 35.0]])
    
    mpiw = ProbabilisticEvaluator.calculate_mpiw(y_low, y_high)
    assert pytest.approx(mpiw) == 16.0 / 3.0


def test_evaluate_wrapper():
    """Verifies the evaluate method returns correct dictionary keys."""
    evaluator = ProbabilisticEvaluator()
    y_true = torch.randn(10, 5)
    intervals = torch.randn(10, 5, 2)
    # Ensure high > low for width logic
    intervals[:, :, 1] = intervals[:, :, 0] + 1.0
    
    metrics = evaluator.evaluate(y_true, intervals)
    assert "picp" in metrics
    assert "mpiw" in metrics
    assert 0.0 <= metrics["picp"] <= 1.0
    assert metrics["mpiw"] >= 0.0
