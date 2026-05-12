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
    
    model = ProbabilisticForecaster(
        input_dim=features,
        hidden_dim=hidden,
        history_size=history,
        horizon=horizon,
        num_quantiles=quantiles
    )
    
    x = torch.randn(batch, history, features)
    out = model(x)
    
    assert out.shape == (batch, horizon, quantiles)


def test_parameter_gradients() -> None:
    """Verifies that gradients flow back through the entire network."""
    model = ProbabilisticForecaster(input_dim=5, hidden_dim=16)
    x = torch.randn(2, 60, 5)
    out = model(x)
    
    loss = out.sum()
    loss.backward()
    
    # Check if encoder weights have gradients
    assert model.encoder.layers[0][0].weight.grad is not None
    # Check if decoder weights have gradients
    assert model.decoder[-1].weight.grad is not None


def test_different_horizons() -> None:
    """Ensures model adapts to different prediction horizons."""
    horizon = 7
    model = ProbabilisticForecaster(input_dim=5, hidden_dim=16, horizon=horizon)
    x = torch.randn(2, 60, 5)
    out = model(x)
    
    assert out.shape[1] == horizon
