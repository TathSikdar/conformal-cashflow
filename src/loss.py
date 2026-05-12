"""Module for the Multi-Quantile (Pinball) Loss function."""

from typing import List

import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """Implementation of the Pinball (Quantile) Loss function.

    This loss function allows the model to predict specific quantiles
    of the target distribution by asymmetrically penalizing errors.

    For a target quantile tau, the loss is:
    L = max(tau * (y - y_pred), (tau - 1) * (y - y_pred))
    """

    def __init__(self, quantiles: List[float]) -> None:
        """Initializes the loss function with a list of target quantiles.

        Args:
            quantiles: List of floats between 0 and 1 (e.g., [0.1, 0.5, 0.9]).
        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the average pinball loss across all quantiles.

        Args:
            preds: Predicted quantiles of shape (Batch, Horizon, NumQuantiles).
            target: Ground truth values of shape (Batch, Horizon).

        Returns:
            Mean loss value across all samples, steps, and quantiles.
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            # Extract predictions for the i-th quantile
            # preds[:, :, i] has shape (Batch, Horizon)
            errors = target - preds[:, :, i]
            
            # Pinball loss formula
            quantile_loss = torch.max(q * errors, (q - 1) * errors)
            losses.append(quantile_loss.unsqueeze(-1))

        # Combine losses for all quantiles
        # Combined shape: (Batch, Horizon, NumQuantiles)
        combined_loss = torch.cat(losses, dim=-1)
        
        return torch.mean(combined_loss)
