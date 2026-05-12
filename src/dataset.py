"""Module for PyTorch Dataset and DataLoader implementations."""

from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class CashFlowDataset(Dataset):
    """Custom Dataset for sliding-window cash flow forecasting.

    This dataset takes a 3D tensor of shape (N, T, F) and yields
    (input, target) pairs based on a sliding window.

    Attributes:
        data: The 3D tensor (Accounts, Time, Features).
        history_size: Number of time steps used as input (Look-back).
        horizon: Number of time steps to predict (Look-ahead).
        target_idx: Index of the feature to be used as the target (e.g., amount_log).
    """

    def __init__(
        self,
        data: torch.Tensor,
        history_size: int = 60,
        horizon: int = 14,
        target_idx: int = 0,
    ) -> None:
        """Initializes the dataset.

        Args:
            data: Tensor of shape (N, T, F).
            history_size: Look-back window size.
            horizon: Prediction horizon size.
            target_idx: Feature index for the target variable.
        """
        self.data = data
        self.history_size = history_size
        self.horizon = horizon
        self.target_idx = target_idx

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a single (input, target) sample.

        Args:
            idx: Global index across all accounts and windows.

        Returns:
            Tuple of (x, y):
                x: Shape (history_size, num_features)
                y: Shape (horizon,) - The target values for the horizon.
        """
        account_idx = idx // self.samples_per_account
        start_step = idx % self.samples_per_account
        
        # Slicing the history
        history_end = start_step + self.history_size
        x = self.data[account_idx, start_step:history_end, :]
        
        # Slicing the horizon (target)
        # We only predict the target feature (usually amount_log)
        horizon_end = history_end + self.horizon
        y = self.data[account_idx, history_end:horizon_end, self.target_idx]
        
        return x, y


def create_dataloader(
    data: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
    history_size: int = 60,
    horizon: int = 14,
) -> DataLoader:
    """Factory function to create a DataLoader.

    Args:
        data: The 3D input tensor.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the samples.
        history_size: Look-back window.
        horizon: Forecast horizon.

    Returns:
        A configured PyTorch DataLoader.
    """
    dataset = CashFlowDataset(
        data=data, 
        history_size=history_size, 
        horizon=horizon
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
