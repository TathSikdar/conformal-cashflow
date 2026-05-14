"""Module for the Multi-Quantile (Pinball) Loss function."""

from typing import List

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in sparse time series.

    Focal Loss down-weights easy examples and focuses on hard negative/positive 
    samples, which is essential for capturing rare transaction spikes.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        """Initializes Focal Loss.

        Args:
            alpha: Balancing factor for rare class.
            gamma: Focusing parameter to reduce loss for well-classified samples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates Focal Loss.

        Args:
            inputs: Probabilities (after Sigmoid).
            targets: Binary labels.

        Returns:
            Mean loss value.
        """
        # Clamp to avoid log(0)
        inputs = torch.clamp(inputs, min=1e-7, max=1.0 - 1e-7)
        
        # Binary Cross Entropy
        bce = -targets * torch.log(inputs) - (1 - targets) * torch.log(1 - inputs)
        
        # pt and alpha_t calculation
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce
        
        return torch.mean(focal_loss)


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
            preds: Predicted quantiles. Last dim must be NumQuantiles.
            target: Ground truth values. Shape must match preds without last dim.

        Returns:
            Mean loss value across all samples and quantiles.
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            # Extract predictions for the i-th quantile
            # Using ellipsis [..., i] handles both (B, H, Q) and (N_active, Q)
            q_preds = preds[..., i]
            errors = target - q_preds
            
            # Pinball loss formula
            quantile_loss = torch.max(q * errors, (q - 1) * errors)
            losses.append(quantile_loss.unsqueeze(-1))

        # Combined shape: same as preds
        combined_loss = torch.cat(losses, dim=-1)
        
        return torch.mean(combined_loss)
