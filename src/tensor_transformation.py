"""Module for transforming flat DataFrames into 3D tensors for PyTorch."""

from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch


from sklearn.preprocessing import StandardScaler


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
        use_standardization: bool = True,
    ) -> None:
        """Initializes the transformer.

        Args:
            account_id_col: Name of account ID column.
            date_col: Name of date column.
            feature_cols: List of feature column names. If None, defaults to basic features.
            sequence_length: Number of daily time steps per account.
            use_standardization: Whether to apply Z-score normalization.
        """
        self.account_id_col = account_id_col
        self.date_col = date_col
        self.feature_cols = feature_cols or ["amount_log", "day_sin", "day_cos"]
        self.sequence_length = sequence_length
        self.use_standardization = use_standardization
        self.scaler = StandardScaler() if use_standardization else None
        
        # Local stats for per-account normalization: {account_id: (mean, std)}
        # We only track this for the target column (idx 0)
        self.local_stats: Dict[Any, Tuple[float, float]] = {}

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
        reindexed = group.reindex(date_range)

        # Fill missing transaction amounts with 0
        if "amount_log" in reindexed.columns:
            reindexed["amount_log"] = reindexed["amount_log"].fillna(0)
            
        # Binary flags like is_weekend or is_month_end should be re-calculated 
        # but we ffill for simplicity as they are usually already in the df
        reindexed = reindexed.ffill().fillna(0)

        return reindexed[self.feature_cols]

    def transform(self, df: pd.DataFrame) -> Tuple[torch.Tensor, np.ndarray]:
        """Pivots the DataFrame into a 3D Tensor (N, T, F).

        Args:
            df: Featured DataFrame.

        Returns:
            Tuple containing:
                - PyTorch Tensor of shape (N, T, F).
                - Array of Account IDs of shape (N,).
        """
        # Daily aggregation
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
            if len(account_group) > 0:
                continuous_df = self._get_continuous_series(account_group)
                
                vals = continuous_df.values.copy()
                
                # LOCAL SCALING: Normalize target (idx 0) per account
                if self.use_standardization:
                    target_vals = vals[:, 0]
                    
                    # Fit-Once logic: Only compute stats if we haven't seen this account before
                    if account_id not in self.local_stats:
                        mean = np.mean(target_vals)
                        std = np.std(target_vals) + 1e-8
                        self.local_stats[account_id] = (mean, std)
                    
                    mean, std = self.local_stats[account_id]
                    vals[:, 0] = (target_vals - mean) / std
                
                tensor_list.append(vals)

        # Stack into (N, T, F)
        np_3d = np.stack(tensor_list)
        
        # Apply Global Standardization for the REST of the features (idx 1:)
        # This keeps cyclical and binary features in a good range
        if self.use_standardization:
            N, T, F = np_3d.shape
            if F > 1:
                other_features = np_3d[:, :, 1:].reshape(-1, F-1)
                scaled_others = self.scaler.fit_transform(other_features)
                np_3d[:, :, 1:] = scaled_others.reshape(N, T, F-1)

        return torch.from_numpy(np_3d).float(), accounts

    def inverse_transform_target(self, account_id: Any, scaled_vals: np.ndarray) -> np.ndarray:
        """Inverts the local scaling for the target column.

        Args:
            account_id: The identifier for the account.
            scaled_vals: The normalized predicted values.

        Returns:
            Values in the original (log) scale.
        """
        if not self.use_standardization or account_id not in self.local_stats:
            return scaled_vals
            
        mean, std = self.local_stats[account_id]
        return (scaled_vals * std) + mean

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
