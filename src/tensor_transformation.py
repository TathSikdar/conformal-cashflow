"""Module for transforming flat DataFrames into 3D tensors for PyTorch."""

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch


class TensorTransformer:
    """Transforms featured financial data into 3D tensors (N, Time, Features).

    Attributes:
        account_id_col: Column name for account identifier.
        date_col: Column name for date.
        feature_cols: List of columns to include as features in the tensor.
        sequence_length: The fixed number of time steps (T) for the tensor.
    """

    def __init__(
        self,
        account_id_col: str = "account_id",
        date_col: str = "date",
        feature_cols: List[str] = None,
        sequence_length: int = 90,
    ) -> None:
        """Initializes the transformer.

        Args:
            account_id_col: Name of account ID column.
            date_col: Name of date column.
            feature_cols: List of feature column names. If None, defaults to basic features.
            sequence_length: Number of daily time steps per account.
        """
        self.account_id_col = account_id_col
        self.date_col = date_col
        self.feature_cols = feature_cols or ["amount_log", "day_sin", "day_cos"]
        self.sequence_length = sequence_length

    def _get_continuous_series(self, group: pd.DataFrame) -> pd.DataFrame:
        """Ensures a group has a continuous daily frequency for the sequence length.

        Args:
            group: DataFrame group for a single account.

        Returns:
            Reindexed DataFrame with daily frequency and fixed length.
        """
        # Set date as index and sort
        group = group.set_index(self.date_col).sort_index()

        # Define the target range (last 'sequence_length' days)
        end_date = group.index.max()
        start_date = end_date - pd.Timedelta(days=self.sequence_length - 1)
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        # Reindex to fill missing days with 0/appropriate values
        # Note: cyclical features like day_sin/cos should ideally be re-calculated for missing days
        # but for this step we assume the input has them or we fill them.
        reindexed = group.reindex(date_range)

        # Fill missing transaction amounts with 0
        if "amount_log" in reindexed.columns:
            reindexed["amount_log"] = reindexed["amount_log"].fillna(0)

        # Forward fill other features (like cyclical ones if they weren't in the range)
        # or fill with 0 if they are rolling stats
        reindexed = reindexed.ffill().fillna(0)

        return reindexed[self.feature_cols]

    def transform(self, df: pd.DataFrame) -> torch.Tensor:
        """Pivots the DataFrame into a 3D Tensor (N, T, F).

        Args:
            df: Featured DataFrame.

        Returns:
            A PyTorch Tensor of shape (num_accounts, sequence_length, num_features).
        """
        # Daily aggregation (in case of multiple transactions per day)
        # We sum the amounts and take the first occurrence for other features (which should be identical per day)
        agg_dict = {col: "first" for col in self.feature_cols}
        if "amount_log" in agg_dict:
            agg_dict["amount_log"] = "sum"

        df_daily = (
            df.groupby([self.account_id_col, self.date_col])
            .agg(agg_dict)
            .reset_index()
        )

        tensor_list = []
        accounts = df_daily[self.account_id_col].unique()

        for account_id in accounts:
            account_group = df_daily[df_daily[self.account_id_col] == account_id]

            # Only process accounts with enough history (optional, here we pad/truncate)
            if len(account_group) > 0:
                continuous_df = self._get_continuous_series(account_group)
                tensor_list.append(continuous_df.values)

        # Stack into (N, T, F)
        np_3d = np.stack(tensor_list)
        return torch.from_numpy(np_3d).float()

    def get_metadata(self) -> dict:
        """Returns metadata about the transformation.

        Returns:
            Dictionary with N, T, F dimensions.
        """
        return {
            "sequence_length": self.sequence_length,
            "num_features": len(self.feature_cols),
            "feature_names": self.feature_cols,
        }
