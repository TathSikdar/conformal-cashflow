"""Unit tests for the PyTorch Dataset module."""

import torch
import pytest
from src.dataset import CashFlowDataset, create_dataloader


@pytest.fixture
def mock_tensor() -> torch.Tensor:
    """Creates a mock 3D tensor (2 accounts, 20 days, 12 features)."""
    # Features: [Amount, FeatureB, ..., FeatureL]
    # N=2, T=20, F=12 (to match default exog_indices [2...11])
    t = torch.zeros((2, 20, 12))
    for i in range(20):
        t[:, i, 0] = float(i)  # Amount is the day index
    return t


def test_dataset_length(mock_tensor: torch.Tensor) -> None:
    """Verifies the total number of samples calculated."""
    history = 10
    horizon = 5
    # Samples per account = 20 - 10 - 5 + 1 = 6
    # Total samples = 2 accounts * 6 = 12
    dataset = CashFlowDataset(mock_tensor, history_size=history, horizon=horizon)
    assert len(dataset) == 12


def test_dataset_item_shapes(mock_tensor: torch.Tensor) -> None:
    """Verifies the shapes of input (x) and target (y)."""
    history = 10
    horizon = 5
    dataset = CashFlowDataset(mock_tensor, history_size=history, horizon=horizon)
    x, future_x, acc_idx, y = dataset[0]
    
    assert x.shape == (10, 12)
    assert future_x.shape == (5, 10) # default exog_indices [2...11]
    assert isinstance(acc_idx, int)
    assert y.shape == (5,)


def test_sliding_window_logic(mock_tensor: torch.Tensor) -> None:
    """Verifies that windows are sliced correctly."""
    history = 10
    horizon = 5
    dataset = CashFlowDataset(mock_tensor, history_size=history, horizon=horizon)
    
    # First sample (idx 0)
    x0, future_x0, acc_idx0, y0 = dataset[0]
    # x0 should be days 0 to 9
    assert x0[0, 0] == 0.0
    assert x0[9, 0] == 9.0
    # y0 should be days 10 to 14
    assert y0[0] == 10.0
    assert y0[4] == 14.0

    # Second sample (idx 1)
    x1, future_x1, acc_idx1, y1 = dataset[1]
    # x1 should be days 1 to 10
    assert x1[0, 0] == 1.0
    assert x1[9, 0] == 10.0
    # y1 should be days 11 to 15
    assert y1[0] == 11.0
    assert y1[4] == 15.0


def test_dataloader_batching(mock_tensor: torch.Tensor) -> None:
    """Verifies that the DataLoader produces correct batch shapes."""
    dataloader = create_dataloader(
        mock_tensor, 
        batch_size=4, 
        history_size=10, 
        horizon=5
    )
    
    batch_x, batch_future_x, batch_acc_idx, batch_y = next(iter(dataloader))
    assert batch_x.shape == (4, 10, 12)
    assert batch_future_x.shape == (4, 5, 10)
    assert batch_acc_idx.shape == (4,)
    assert batch_y.shape == (4, 5)


def test_invalid_window_size(mock_tensor: torch.Tensor) -> None:
    """Verifies that error is raised if window > total time."""
    with pytest.raises(ValueError):
        CashFlowDataset(mock_tensor, history_size=15, horizon=10)
