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
        self.linear = nn.Linear(5, 3) # Predict 3 quantiles for 1 horizon step
    def forward(self, x):
        # x: (Batch, History, Features) -> (Batch, 1, 3)
        return self.linear(x.mean(dim=1)).unsqueeze(1)


@pytest.fixture
def mock_setup():
    """Sets up a minimal training environment."""
    model = MockModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = lambda preds, target: torch.mean((preds.squeeze(1)[:, 1] - target.squeeze(1))**2) # Dummy MSE-like loss
    
    # Data: (Batch, History, Features)
    x = torch.randn(10, 5, 5)
    y = torch.randn(10, 1)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)
    
    return model, optimizer, criterion, loader


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
    # Setting grad_clip very low
    trainer = Trainer(model, optimizer, criterion, grad_clip=1e-5)
    
    # Should run without error
    trainer.train_epoch(loader)
