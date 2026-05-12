"""Unit tests for the Quantile Loss module."""

import torch
import pytest
from src.loss import QuantileLoss


def test_quantile_loss_symmetry() -> None:
    """Verifies that 0.5 quantile loss is proportional to MAE."""
    criterion = QuantileLoss(quantiles=[0.5])
    
    target = torch.tensor([[10.0, 20.0]])
    preds = torch.tensor([[[9.0], [22.0]]]) # Shape (1, 2, 1)
    
    # Errors: [1.0, -2.0]
    # Pinball(0.5): max(0.5*1, -0.5*1) = 0.5; max(0.5*-2, -0.5*-2) = 1.0
    # Mean: (0.5 + 1.0) / 2 = 0.75
    # MAE would be (1.0 + 2.0) / 2 = 1.5. So 0.5 * MAE = 0.75.
    
    loss = criterion(preds, target)
    assert torch.isclose(loss, torch.tensor(0.75))


def test_asymmetric_penalty() -> None:
    """Verifies that high quantiles penalize underestimation more heavily."""
    # Target is 10. Prediction is 5 (underestimation by 5)
    target = torch.tensor([[10.0]])
    preds = torch.tensor([[[5.0]]])
    
    # Tau = 0.9: max(0.9 * 5, -0.1 * 5) = 4.5
    loss_09 = QuantileLoss(quantiles=[0.9])(preds, target)
    
    # Tau = 0.1: max(0.1 * 5, -0.9 * 5) = 0.5
    loss_01 = QuantileLoss(quantiles=[0.1])(preds, target)
    
    assert loss_09 > loss_01
    assert torch.isclose(loss_09, torch.tensor(4.5))
    assert torch.isclose(loss_01, torch.tensor(0.5))


def test_multi_quantile_shape() -> None:
    """Verifies loss calculation for multiple quantiles."""
    criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    batch, horizon, n_q = 4, 14, 3
    
    preds = torch.randn(batch, horizon, n_q)
    target = torch.randn(batch, horizon)
    
    loss = criterion(preds, target)
    
    assert loss.dim() == 0 # Scalar
    assert not torch.isnan(loss)
