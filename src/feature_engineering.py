"""Module for temporal feature engineering and cyclical time encodings."""

import numpy as np
import pandas as pd


class TemporalFeatureEngineer:
    """Engineer for temporal features in financial time-series.

    This class handles cyclical encoding of time variables and generation
    of rolling statistics to capture local trends and volatility.
    """

    def __init__(
        self,
        date_col: str = "date",
        amount_col: str = "amount",
        account_id_col: str = "account_id",
    ) -> None:
        """Initializes the engineer.

        Args:
            date_col: Name of the date column.
            amount_col: Name of the transaction amount column.
            account_id_col: Name of the account ID column for grouping.
        """
        self.date_col = date_col
        self.amount_col = amount_col
        self.account_id_col = account_id_col

    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds sine and cosine transforms for day-of-week and month.

        Cyclical encoding ensures that the model understands that Sunday (6)
        is adjacent to Monday (0).

        Args:
            df: The input DataFrame.

        Returns:
            DataFrame with cyclical temporal features added.
        """
        # Day of week (0-6)
        day_of_week = df[self.date_col].dt.dayofweek
        df["day_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        df["day_cos"] = np.cos(2 * np.pi * day_of_week / 7)

        # Month (1-12)
        month = df[self.date_col].dt.month
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)

        return df

    def add_rolling_features(
        self, df: pd.DataFrame, windows: list[int] = [7, 30]
    ) -> pd.DataFrame:
        """Generates rolling statistics for transaction amounts.

        Captures historical trends and local volatility per account.

        Args:
            df: The input DataFrame.
            windows: List of window sizes (in days) for rolling stats.

        Returns:
            DataFrame with rolling features added.
        """
        # Ensure data is sorted for rolling operations
        df = df.sort_values(by=[self.account_id_col, self.date_col])

        for window in windows:
            group = df.groupby(self.account_id_col)[self.amount_col]
            df[f"rolling_mean_{window}"] = (
                group.transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )
            df[f"rolling_std_{window}"] = (
                group.transform(lambda x: x.rolling(window=window, min_periods=1).std())
            ).fillna(0)

        return df

    def handle_zero_inflation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies log-transformation to handle high-variance/zero-inflated data.

        Financial amounts often follow a power-law distribution. Log-scaling
        stabilizes variance and brings the distribution closer to normal.

        Args:
            df: The input DataFrame.

        Returns:
            DataFrame with log-transformed amount column.
        """
        # Use log1p (log(1+x)) to handle zeros and ensure positive values
        # We assume amounts are absolute values or that we handle outflow/inflow separately
        df[f"{self.amount_col}_log"] = np.log1p(df[self.amount_col].abs())
        return df

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes the full feature engineering pipeline.

        Args:
            df: Cleaned input DataFrame.

        Returns:
            DataFrame ready for tensorization.
        """
        df = self.add_cyclical_features(df)
        df = self.add_rolling_features(df)
        df = self.handle_zero_inflation(df)
        return df
