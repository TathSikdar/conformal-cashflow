"""Unit tests for Temporal Feature Engineering module."""

import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import TemporalFeatureEngineer


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """Provides a cleaned sample DataFrame."""
    data = {
        "account_id": [1] * 10,
        "date": pd.date_range(start="1993-01-01", periods=10, freq="D"),
        "amount": [100.0, 150.0, 0.0, 200.0, 50.0, 300.0, 0.0, 100.0, 150.0, 200.0]
    }
    return pd.DataFrame(data)


def test_cyclical_features(clean_df: pd.DataFrame) -> None:
    """Verifies sine and cosine transforms for temporal cycles."""
    engineer = TemporalFeatureEngineer()
    df = engineer.add_cyclical_features(clean_df)

    assert "day_sin" in df.columns
    assert "day_cos" in df.columns
    assert "month_sin" in df.columns
    assert "month_cos" in df.columns

    # Verify cyclical property: sin^2 + cos^2 = 1
    np.testing.assert_allclose(df["day_sin"]**2 + df["day_cos"]**2, 1.0)
    np.testing.assert_allclose(df["month_sin"]**2 + df["month_cos"]**2, 1.0)


def test_rolling_features(clean_df: pd.DataFrame) -> None:
    """Verifies rolling mean and standard deviation calculations."""
    engineer = TemporalFeatureEngineer()
    # Use a small window for testing
    df = engineer.add_rolling_features(clean_df, windows=[3])

    assert "rolling_mean_3" in df.columns
    assert "rolling_std_3" in df.columns

    # First row mean should be the value itself (min_periods=1)
    assert df.iloc[0]["rolling_mean_3"] == 100.0
    # First row std should be 0 (handled by fillna(0))
    assert df.iloc[0]["rolling_std_3"] == 0.0

    # Third row mean: (100 + 150 + 0) / 3 = 83.333
    expected_mean = (100.0 + 150.0 + 0.0) / 3
    assert np.isclose(df.iloc[2]["rolling_mean_3"], expected_mean)


def test_zero_inflation_handling(clean_df: pd.DataFrame) -> None:
    """Verifies log1p transformation for amount column."""
    engineer = TemporalFeatureEngineer()
    df = engineer.handle_zero_inflation(clean_df)

    assert "amount_log" in df.columns
    # log1p(0) = 0
    assert df.loc[clean_df["amount"] == 0.0, "amount_log"].iloc[0] == 0.0
    # log1p(100) approx 4.615
    assert np.isclose(df.iloc[0]["amount_log"], np.log1p(100.0))


def test_pipeline_execution(clean_df: pd.DataFrame) -> None:
    """Verifies that the full pipeline runs and produces expected columns."""
    engineer = TemporalFeatureEngineer()
    df = engineer.pipeline(clean_df)

    expected_cols = [
        "day_sin", "day_cos", "month_sin", "month_cos",
        "rolling_mean_7", "rolling_std_7", "amount_log"
    ]
    for col in expected_cols:
        assert col in df.columns
