"""Unit tests for the Model Architecture module."""

import torch
import pytest
from src.model import ProbabilisticForecaster, TemporalEncoder


def test_temporal_encoder_shape() -> None:
    """Verifies that the encoder preserves temporal length and maps to hidden dim."""
    batch, time, features = 8, 60, 5
    hidden = 16
    encoder = TemporalEncoder(input_dim=features, hidden_dim=hidden)
    
    x = torch.randn(batch, time, features)
    out = encoder(x)
    
    # Encoder output shape: (Batch, Hidden, Time)
    assert out.shape == (batch, hidden, time)


def test_forecaster_output_shape() -> None:
    """Verifies the final output shape (Batch, Horizon, NumQuantiles)."""
    batch = 4
    history = 60
    features = 7
    horizon = 14
    quantiles = 3
    hidden = 32
    num_accounts = 10
    future_dim = 6
    
    model = ProbabilisticForecaster(
        input_dim=features,
        num_accounts=num_accounts,
        hidden_dim=hidden,
        history_size=history,
        horizon=horizon,
        num_quantiles=quantiles,
        future_dim=future_dim
    )
    
    x = torch.randn(batch, history, features)
    future_x = torch.randn(batch, horizon, future_dim)
    acc_idx = torch.randint(0, num_accounts, (batch,))
    
    probs, magnitudes = model(x, future_x, acc_idx)
    
    assert probs.shape == (batch, horizon)
    assert magnitudes.shape == (batch, horizon, quantiles)


def test_parameter_gradients() -> None:
    """Verifies that gradients flow back through the entire network."""
    num_accounts = 5
    future_dim = 6
    model = ProbabilisticForecaster(input_dim=5, num_accounts=num_accounts, hidden_dim=16, future_dim=future_dim)
    x = torch.randn(2, 60, 5)
    future_x = torch.randn(2, 14, future_dim)
    acc_idx = torch.randint(0, num_accounts, (2,))
    
    probs, magnitudes = model(x, future_x, acc_idx)
    
    # Use both heads to ensure gradients flow everywhere
    loss = probs.sum() + magnitudes.sum()
    loss.backward()
    
    # Check if encoder weights have gradients
    assert model.encoder.layers[0][0].weight.grad is not None
    # Check if head GRN weights have gradients
    assert model.magnitude_grn.lin1.weight.grad is not None
    assert model.gate_grn.lin1.weight.grad is not None
    # Check if account embedding has gradients
    assert model.account_embedding.weight.grad is not None


def test_different_horizons() -> None:
    """Ensures model adapts to different prediction horizons."""
    horizon = 7
    num_accounts = 5
    future_dim = 6
    model = ProbabilisticForecaster(input_dim=5, num_accounts=num_accounts, hidden_dim=16, horizon=horizon, future_dim=future_dim)
    x = torch.randn(2, 60, 5)
    future_x = torch.randn(2, horizon, future_dim)
    acc_idx = torch.randint(0, num_accounts, (2,))
    
    probs, magnitudes = model(x, future_x, acc_idx)
    
    assert probs.shape[1] == horizon
    assert magnitudes.shape[1] == horizon
