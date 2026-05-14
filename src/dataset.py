"""Module for PyTorch Dataset and DataLoader implementations."""

from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class CashFlowDataset(Dataset):
    """Custom Dataset for sliding-window cash flow forecasting.

    This dataset takes a 3D tensor of shape (N, T, F) and yields
    (x, future_x, account_idx, y) pairs based on a sliding window.

    Attributes:
        data: The 3D tensor (Accounts, Time, Features).
        history_size: Number of time steps used as input (Look-back).
        horizon: Number of time steps to predict (Look-ahead).
        target_idx: Index of the feature to be used as the target (e.g., amount_log).
        exog_indices: Indices of exogenous features (e.g., day_sin, is_weekend) 
                      to be provided for the future horizon.
    """

    def __init__(
        self,
        data: torch.Tensor,
        history_size: int = 60,
        horizon: int = 14,
        target_idx: int = 0,
        exog_indices: list[int] = [2, 3, 4, 5, 6, 7],
    ) -> None:
        """Initializes the dataset.

        Args:
            data: Tensor of shape (N, T, F).
            history_size: Look-back window size.
            horizon: Prediction horizon size.
            target_idx: Feature index for the target variable.
            exog_indices: Indices of features available for the future.
        """
        self.data = data
        self.history_size = history_size
        self.horizon = horizon
        self.target_idx = target_idx
        self.exog_indices = exog_indices

        self.num_accounts, self.total_time, self.num_features = data.shape
        
        # Calculate samples per account
        self.samples_per_account = self.total_time - history_size - horizon + 1
        
        if self.samples_per_account <= 0:
            raise ValueError(
                f"Total time {self.total_time} is too short for "
                f"history {history_size} and horizon {horizon}."
            )

    def __len__(self) -> int:
        """Total number of sliding window samples across all accounts."""
        return self.num_accounts * self.samples_per_account

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Returns a single sample with future features and account ID.

        Returns:
            Tuple of (x, future_x, account_idx, y):
                x: History features (history_size, F)
                future_x: Future exogenous features (horizon, len(exog_indices))
                account_idx: The integer index of the account (for embedding)
                y: Target values for the horizon (horizon,)
        """
        account_idx = idx // self.samples_per_account
        start_step = idx % self.samples_per_account
        
        # Slicing the history
        history_end = start_step + self.history_size
        x = self.data[account_idx, start_step:history_end, :]
        
        # Slicing the future exogenous features
        horizon_end = history_end + self.horizon
        future_x = self.data[account_idx, history_end:horizon_end, self.exog_indices]
        
        # Target
        y = self.data[account_idx, history_end:horizon_end, self.target_idx]
        
        return x, future_x, account_idx, y


def create_dataloader(
    data: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
    history_size: int = 60,
    horizon: int = 14,
    exog_indices: list[int] = [2, 3, 4, 5, 6, 7],
) -> DataLoader:
    """Factory function to create a DataLoader.

    Args:
        data: The 3D input tensor.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the samples.
        history_size: Look-back window.
        horizon: Forecast horizon.
        exog_indices: Indices of features available for the future.

    Returns:
        A configured PyTorch DataLoader.
    """
    dataset = CashFlowDataset(
        data=data, 
        history_size=history_size, 
        horizon=horizon,
        exog_indices=exog_indices
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
