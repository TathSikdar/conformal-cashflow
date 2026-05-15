"""Unit tests for the Trainer module."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest
from src.trainer import Trainer, EarlyStopping


class MockModel(nn.Module):
    """Simple model for testing training loops."""
    def __init__(self):
        super().__init__()
        self.gate_linear = nn.Linear(5, 1) 
        self.magnitude_linear = nn.Linear(5, 3) 
        
    def forward(self, x, future_x, acc_idx):
        # x: (Batch, History, Features)
        context = x.mean(dim=1)
        batch_size = x.size(0)
        horizon = future_x.size(1)
        
        # Simple dependency on future_x and acc_idx to ensure they are used
        # probs: (Batch, Horizon)
        probs = torch.sigmoid(self.gate_linear(context)).repeat(1, horizon)
        # magnitudes: (Batch, Horizon, Quantiles)
        magnitudes = self.magnitude_linear(context).unsqueeze(1).repeat(1, horizon, 1)
        
        return probs, magnitudes


@pytest.fixture
def mock_setup():
    """Sets up a minimal training environment."""
    model = MockModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    def dummy_criterion(magnitudes, target):
        # magnitudes: (N_active, 3), target: (N_active,)
        return torch.mean((magnitudes[:, 1] - target)**2)
    
    # Data: (x, future_x, acc_idx, y)
    # Using horizon > 1 to support SequenceSharpnessLoss (variance/correlation)
    x = torch.randn(10, 5, 5)
    future_x = torch.randn(10, 2, 6) # Horizon 2
    acc_idx = torch.zeros(10, dtype=torch.long)
    y = torch.randn(10, 2) # Horizon 2
    
    dataset = TensorDataset(x, future_x, acc_idx, y)
    loader = DataLoader(dataset, batch_size=2)
    
    return model, optimizer, dummy_criterion, loader


def test_early_stopping():
    """Verifies that early stopping triggers correctly."""
    stopper = EarlyStopping(patience=2)
    
    stopper(10.0) # Best
    assert stopper.early_stop is False
    
    stopper(11.0) # Worse
    assert stopper.early_stop is False
    
    stopper(11.0) # Worse again
    assert stopper.early_stop is True


def test_trainer_epoch(mock_setup):
    """Verifies that a training epoch updates weights."""
    model, optimizer, criterion, loader = mock_setup
    trainer = Trainer(model, optimizer, criterion)
    
    initial_params = [p.clone() for p in model.parameters()]
    loss = trainer.train_epoch(loader)
    
    assert loss > 0
    # Check that weights have changed
    for p1, p2 in zip(initial_params, model.parameters()):
        assert not torch.equal(p1, p2)


def test_trainer_fit(mock_setup):
    """Verifies the full fit cycle with early stopping."""
    model, optimizer, criterion, loader = mock_setup
    trainer = Trainer(model, optimizer, criterion)
    
    history = trainer.fit(loader, loader, epochs=3, early_stopping_patience=5)
    
    assert len(history["train_loss"]) == 3
    assert len(history["val_loss"]) == 3


def test_gradient_clipping(mock_setup):
    """Ensures gradient clipping is applied (smoke test)."""
    model, optimizer, criterion, loader = mock_setup
    trainer = Trainer(model, optimizer, criterion, grad_clip=1e-5)
    
    # Should run without error
    trainer.train_epoch(loader)
