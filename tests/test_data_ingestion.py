"""Unit tests for the Data Ingestion module."""

import os
import pandas as pd
import pytest
from src.data_ingestion import FinancialDataIngestor


@pytest.fixture
def sample_csv(tmp_path: str) -> str:
    """Creates a temporary sample CSV file for testing."""
    data = {
        "account_id": [1, 1, 2, 2, 1],
        "date": ["1993-01-01", "1993-01-02", "1993-01-01", "1993-01-02", "1993-01-01"],  # Duplicate row
        "amount": [100.0, 200.0, 50.0, 150.0, 100.0],
        "other": ["a", "b", "c", "d", "a"]
    }
    df = pd.DataFrame(data)
    file_path = os.path.join(tmp_path, "test_data.csv")
    df.to_csv(file_path, index=False)
    return file_path


def test_load_data(sample_csv: str) -> None:
    """Verifies that data is loaded correctly from CSV."""
    ingestor = FinancialDataIngestor()
    df = ingestor.load_data(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "account_id" in df.columns


def test_clean_data(sample_csv: str) -> None:
    """Verifies data cleaning: deduplication, type casting, and sorting."""
    ingestor = FinancialDataIngestor()
    raw_df = ingestor.load_data(sample_csv)
    cleaned_df = ingestor.clean_data(raw_df)

    # Check deduplication (one row was duplicate)
    assert len(cleaned_df) == 4

    # Check date conversion
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["date"])

    # Check sorting (account_id then date)
    assert cleaned_df.iloc[0]["account_id"] == 1
    assert cleaned_df.iloc[2]["account_id"] == 2
    assert cleaned_df.iloc[0]["date"] < cleaned_df.iloc[1]["date"]


def test_missing_values_handling(tmp_path: str) -> None:
    """Verifies that rows with missing critical values are dropped."""
    data = {
        "account_id": [1, None, 2],
        "date": ["1993-01-01", "1993-01-01", None],
        "amount": [100.0, 200.0, 50.0]
    }
    df = pd.DataFrame(data)
    file_path = os.path.join(tmp_path, "missing_data.csv")
    df.to_csv(file_path, index=False)

    ingestor = FinancialDataIngestor()
    raw_df = ingestor.load_data(file_path)
    cleaned_df = ingestor.clean_data(raw_df)

    # Should only have one valid row
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["account_id"] == 1
