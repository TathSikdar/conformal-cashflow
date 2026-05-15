"""Unit tests for the Quantile Loss module."""

import torch
import pytest
from src.loss import QuantileLoss, SequenceSharpnessLoss


def test_quantile_loss_symmetry() -> None:
    """Verifies that 0.5 quantile loss is proportional to MAE."""
    criterion = QuantileLoss(quantiles=[0.5], use_weighting=False)
    
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
    loss_09 = QuantileLoss(quantiles=[0.9], use_weighting=False)(preds, target)
    
    # Tau = 0.1: max(0.1 * 5, -0.9 * 5) = 0.5
    loss_01 = QuantileLoss(quantiles=[0.1], use_weighting=False)(preds, target)
    
    assert loss_09 > loss_01
    assert torch.isclose(loss_09, torch.tensor(4.5))
    assert torch.isclose(loss_01, torch.tensor(0.5))


def test_multi_quantile_shape() -> None:
    """Verifies loss calculation for multiple quantiles."""
    criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9], use_weighting=False)
    batch, horizon, n_q = 4, 14, 3
    
    preds = torch.randn(batch, horizon, n_q)
    target = torch.randn(batch, horizon)
    
    loss = criterion(preds, target)
    
    assert loss.dim() == 0 # Scalar
    assert not torch.isnan(loss)


def test_magnitude_weighting() -> None:
    """Verifies that large targets increase the loss magnitude."""
    criterion_weighted = QuantileLoss(quantiles=[0.5], use_weighting=True)
    criterion_unweighted = QuantileLoss(quantiles=[0.5], use_weighting=False)
    
    # Same error, different target magnitudes
    target_small = torch.tensor([[1.0]])
    target_large = torch.tensor([[100.0]])
    
    preds_small = torch.tensor([[[0.0]]]) # Error = 1.0
    preds_large = torch.tensor([[[99.0]]]) # Error = 1.0
    
    loss_small = criterion_weighted(preds_small, target_small)
    loss_large = criterion_weighted(preds_large, target_large)
    
    # Weight_small = |1.0| + 0.1 = 1.1
    # Weight_large = |100.0| + 0.1 = 100.1
    # Pinball(0.5) for both is 0.5
    # loss_small = 0.5 * 1.1 = 0.55
    # loss_large = 0.5 * 100.1 = 50.05
    
    assert loss_large > loss_small
    assert torch.isclose(loss_small, torch.tensor(0.55))
    assert torch.isclose(loss_large, torch.tensor(50.05))


def test_sequence_sharpness_loss() -> None:
    """Verifies Volume, Variance, and Shape components of sharpness loss."""
    criterion = SequenceSharpnessLoss()
    
    # Batch=1, Horizon=5
    y_true = torch.tensor([[0.0, 10.0, 0.0, 10.0, 0.0]])
    
    # 1. Perfect match
    loss_perfect = criterion(y_true, y_true)
    assert loss_perfect < 1e-5
    
    # 2. Flat prediction (Zero regression)
    y_flat = torch.tensor([[4.0, 4.0, 4.0, 4.0, 4.0]]) # Same mean, zero variance
    loss_flat = criterion(y_flat, y_true)
    
    # 3. Misaligned peaks (Correct volume and variance, wrong shape)
    y_shifted = torch.tensor([[10.0, 0.0, 10.0, 0.0, 0.0]])
    loss_shifted = criterion(y_shifted, y_true)
    
    # Flat should be worse than perfect
    assert loss_flat > loss_perfect
    # Shifted should be worse than perfect
    assert loss_shifted > loss_perfect
    
    # Correlation for shifted: y_true peaks at 1,3; y_shifted peaks at 0,2.
    # They should have lower correlation than perfect.
