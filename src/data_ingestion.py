"""Module for ingesting and cleaning financial transaction data."""

import abc
from typing import Dict, Optional

import pandas as pd


class DataIngestor(abc.ABC):
    """Abstract base class for data ingestion pipelines."""

    @abc.abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Loads raw data from a source.

        Args:
            file_path: Path to the raw data file.

        Returns:
            A pandas DataFrame containing the raw data.
        """
        pass

    @abc.abstractmethod
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the raw DataFrame.

        Args:
            df: The raw pandas DataFrame.

        Returns:
            A cleaned pandas DataFrame.
        """
        pass


class FinancialDataIngestor(DataIngestor):
    """Ingestor for PKDD'99 / COFINFAD financial datasets.

    Attributes:
        account_id_col: Name of the account identifier column.
        date_col: Name of the transaction date column.
        amount_col: Name of the transaction amount column.
    """

    def __init__(
        self,
        account_id_col: str = "account_id",
        date_col: str = "date",
        amount_col: str = "amount",
    ) -> None:
        """Initializes the ingestor with specific column names.

        Args:
            account_id_col: Name of the account ID column.
            date_col: Name of the transaction date column.
            amount_col: Name of the transaction amount column.
        """
        self.account_id_col = account_id_col
        self.date_col = date_col
        self.amount_col = amount_col

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Loads data from a CSV file with error handling.

        Args:
            file_path: Path to the CSV file.

        Returns:
            Loaded pandas DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            pd.errors.EmptyDataError: If the file is empty.
        """
        try:
            # Low memory set to False for better type inference in large datasets
            return pd.read_csv(file_path, low_memory=False)
        except FileNotFoundError as e:
            print(f"Error: File not found at {file_path}")
            raise e
        except pd.errors.EmptyDataError as e:
            print(f"Error: File at {file_path} is empty")
            raise e

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs standard cleaning: deduplication, handling NaNs, and sorting.

        Args:
            df: Raw DataFrame.

        Returns:
            Cleaned and sorted DataFrame.
        """
        # Drop duplicates
        df = df.drop_duplicates()

        # Handle missing values in critical columns
        critical_cols = [self.account_id_col, self.date_col, self.amount_col]
        df = df.dropna(subset=critical_cols)

        # Convert date to datetime objects
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        # Ensure numeric types for amount
        df[self.amount_col] = pd.to_numeric(df[self.amount_col], errors="coerce")
        df = df.dropna(subset=[self.amount_col])

        # Sort by account and date for sequential processing
        df = df.sort_values(by=[self.account_id_col, self.date_col])

        return df.reset_index(drop=True)
