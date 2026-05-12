"""Unit tests for the Tensor Transformation module."""

import pandas as pd
import torch
import pytest
from src.tensor_transformation import TensorTransformer


@pytest.fixture
def featured_df() -> pd.DataFrame:
    """Provides a sample featured DataFrame with multiple accounts."""
    data = {
        "account_id": [1, 1, 1, 2, 2],
        "date": pd.to_datetime(["1993-01-01", "1993-01-02", "1993-01-05", "1993-01-01", "1993-01-02"]),
        "amount_log": [4.6, 5.2, 3.9, 4.0, 4.5],
        "day_sin": [0.0, 0.7, -0.7, 0.0, 0.7],
        "day_cos": [1.0, 0.7, 0.7, 1.0, 0.7]
    }
    return pd.DataFrame(data)


def test_tensor_shape(featured_df: pd.DataFrame) -> None:
    """Verifies the output tensor dimensions (N, T, F)."""
    sequence_length = 10
    features = ["amount_log", "day_sin", "day_cos"]
    transformer = TensorTransformer(
        sequence_length=sequence_length,
        feature_cols=features
    )
    
    tensor = transformer.transform(featured_df)
    
    # N = 2 unique accounts
    # T = 10 days
    # F = 3 features
    assert tensor.shape == (2, 10, 3)
    assert isinstance(tensor, torch.Tensor)


def test_continuous_reindexing(featured_df: pd.DataFrame) -> None:
    """Verifies that missing days are filled with 0 for amounts."""
    sequence_length = 5
    transformer = TensorTransformer(
        sequence_length=sequence_length,
        feature_cols=["amount_log"]
    )
    
    tensor = transformer.transform(featured_df)
    
    # Account 1 has dates Jan 1, 2, 5. 
    # With T=5 and end_date=Jan 5, range is Jan 1, 2, 3, 4, 5.
    # Jan 3 and 4 should be 0.
    account_1_tensor = tensor[0]
    assert account_1_tensor[0, 0] == 4.6  # Jan 1
    assert account_1_tensor[1, 0] == 5.2  # Jan 2
    assert account_1_tensor[2, 0] == 0.0  # Jan 3 (filled)
    assert account_1_tensor[3, 0] == 0.0  # Jan 4 (filled)
    assert account_1_tensor[4, 0] == 3.9  # Jan 5


def test_metadata(featured_df: pd.DataFrame) -> None:
    """Verifies metadata retrieval."""
    transformer = TensorTransformer(sequence_length=90, feature_cols=["a", "b"])
    metadata = transformer.get_metadata()
    
    assert metadata["sequence_length"] == 90
    assert metadata["num_features"] == 2
    assert metadata["feature_names"] == ["a", "b"]
