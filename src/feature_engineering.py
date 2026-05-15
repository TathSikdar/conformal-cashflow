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

        # Day of Month (1-31)
        dom = df[self.date_col].dt.day
        df["dom_sin"] = np.sin(2 * np.pi * dom / 31)
        df["dom_cos"] = np.cos(2 * np.pi * dom / 31)

        return df

    def add_spectral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds spectral magnitude features using FFT.

        Captures account-specific periodicity by identifying dominant
        frequencies in transaction history.

        Args:
            df: The input DataFrame.

        Returns:
            DataFrame with 'spectral_top_1' and 'spectral_top_2' features.
        """
        # Ensure data is sorted
        df = df.sort_values(by=[self.account_id_col, self.date_col])
        
        def get_top_frequencies(group: pd.Series) -> pd.Series:
            # We need a dense series for FFT to be meaningful
            # For this feature, we use the raw group as is (assuming daily dense already)
            vals = group.values
            if len(vals) < 8: # Not enough data for meaningful FFT
                return pd.Series([0.0, 0.0], index=["spectral_top_1", "spectral_top_2"])
            
            # Compute FFT magnitudes
            fft_vals = np.abs(np.fft.rfft(vals - np.mean(vals)))
            # Sort and take top 2 (excluding DC component if we didn't subtract mean)
            top_indices = np.argsort(fft_vals)[-2:]
            top_mags = fft_vals[top_indices]
            
            # Normalize by sequence length
            top_mags = top_mags / len(vals)
            
            return pd.Series(top_mags[::-1], index=["spectral_top_1", "spectral_top_2"])

        # Apply per account and unstack to get columns
        spectral = df.groupby(self.account_id_col)[f"{self.amount_col}"].apply(get_top_frequencies).unstack().reset_index()
        
        # Merge back
        df = df.merge(spectral, on=self.account_id_col, how="left")
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
        """Applies a Symmetric Signed Log-Transformation.

        Preserves the sign of the cash flow (inflow/outflow) while stabilizing
        variance for high-magnitude transactions.

        Formula: sign(x) * log1p(abs(x))

        Args:
            df: The input DataFrame.

        Returns:
            DataFrame with signed log-transformed amount column.
        """
        amounts = df[self.amount_col]
        df[f"{self.amount_col}_log"] = np.sign(amounts) * np.log1p(np.abs(amounts))
        return df

    def add_lagged_features(self, df: pd.DataFrame, lags: list[int] = [1, 7, 14, 28, 30]) -> pd.DataFrame:
        """Adds lagged transaction features.

        Args:
            df: Input DataFrame.
            lags: List of days to lag.

        Returns:
            DataFrame with lags.
        """
        for lag in lags:
            df[f"amount_lag_{lag}"] = df.groupby(self.account_id_col)[f"{self.amount_col}_log"].shift(lag).fillna(0)
        return df

    def add_balance_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes the cumulative running balance for each account.

        State-based features are often more predictive than delta-based ones.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with 'balance_log' feature.
        """
        df = df.sort_values(by=[self.account_id_col, self.date_col])
        # Calculate cumulative sum of the raw amounts per account
        df["balance"] = df.groupby(self.account_id_col)[self.amount_col].cumsum()
        # Log-scale the balance (handling negative balances with signed-log)
        df["balance_log"] = np.sign(df["balance"]) * np.log1p(np.abs(df["balance"]))
        return df

    def add_calendar_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds binary flags for weekends, month-ends, and payday anchors.

        Banking transactions often cluster around these periodic events.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with calendar flags and payday anchors added.
        """
        # Weekends (Saturday=5, Sunday=6)
        df["is_weekend"] = (df[self.date_col].dt.dayofweek >= 5).astype(float)
        
        # Month-End (Days 28-31)
        df["is_month_end"] = (df[self.date_col].dt.day >= 28).astype(float)
        
        # PAYDAY ANCHORS: Countdown to key dates
        day = df[self.date_col].dt.day
        
        # Days until 15th (clipped for normalization)
        df["days_to_15th"] = (15 - day).clip(lower=-15, upper=15) / 15.0
        
        # Days until month end
        df["days_to_month_end"] = (df[self.date_col].dt.days_in_month - day) / 31.0
        
        return df

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes the full feature engineering pipeline.

        Args:
            df: Cleaned input DataFrame.

        Returns:
            DataFrame ready for tensorization.
        """
        df = self.add_cyclical_features(df)
        df = self.add_calendar_flags(df)
        df = self.add_spectral_features(df)
        df = self.add_rolling_features(df)
        df = self.handle_zero_inflation(df)
        df = self.add_lagged_features(df)
        df = self.add_balance_feature(df)
        return df
