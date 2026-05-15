"""Module for the Multi-Quantile (Pinball) Loss function."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        # pt and alpha_t calculation
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Focal loss formula
        focal_loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt)
        
        return torch.mean(focal_loss)


class QuantileLoss(nn.Module):
    """Implementation of the Pinball (Quantile) Loss function.

    This loss function allows the model to predict specific quantiles
    of the target distribution by asymmetrically penalizing errors.

    For a target quantile tau, the loss is:
    L = max(tau * (y - y_pred), (tau - 1) * (y - y_pred))

    We add Magnitude Weighting: Loss = PinballLoss * (abs(y_true) + epsilon)
    """

    def __init__(self, quantiles: List[float], use_weighting: bool = True) -> None:
        """Initializes the loss function with a list of target quantiles.

        Args:
            quantiles: List of floats between 0 and 1 (e.g., [0.1, 0.5, 0.9]).
            use_weighting: Whether to scale loss by target magnitude.
        """
        super().__init__()
        self.quantiles = quantiles
        self.use_weighting = use_weighting

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the average pinball loss across all quantiles.

        Args:
            preds: Predicted quantiles. Last dim must be NumQuantiles.
            target: Ground truth values. Shape must match preds without last dim.

        Returns:
            Mean loss value across all samples and quantiles.
        """
        losses = []
        
        # Calculate magnitude weight if requested
        # We use a small epsilon to ensure zeros still have some loss
        weight = torch.abs(target) + 0.1 if self.use_weighting else 1.0

        for i, q in enumerate(self.quantiles):
            # Extract predictions for the i-th quantile
            # Using ellipsis [..., i] handles both (B, H, Q) and (N_active, Q)
            q_preds = preds[..., i]
            errors = target - q_preds
            
            # Pinball loss formula
            quantile_loss = torch.max(q * errors, (q - 1) * errors)
            
            # Apply magnitude weighting
            weighted_loss = quantile_loss * weight
            
            losses.append(weighted_loss.unsqueeze(-1))

        # Combined shape: same as preds
        combined_loss = torch.cat(losses, dim=-1)
        
        return torch.mean(combined_loss)


class SequenceSharpnessLoss(nn.Module):
    """Combines Volume, Variance, and Gaussian-Smoothed Correlation losses.
    
    Operates on the expected forecast (probs * magnitudes) across the horizon.
    """

    def __init__(
        self, 
        volume_weight: float = 1.0, 
        variance_weight: float = 1.0, 
        shape_weight: float = 1.0,
        sigma: float = 1.0
    ) -> None:
        """Initializes the Sharpness Loss.

        Args:
            volume_weight: Penalty weight for total cash flow mismatch.
            variance_weight: Penalty weight for insufficient spikiness.
            shape_weight: Penalty weight for temporal misalignment.
            sigma: Standard deviation for Gaussian smoothing kernel.
        """
        super().__init__()
        self.volume_weight = volume_weight
        self.variance_weight = variance_weight
        self.shape_weight = shape_weight
        self.sigma = sigma

    def _apply_gaussian_smoothing(self, x: torch.Tensor) -> torch.Tensor:
        """Applies 1D Gaussian smoothing to the horizon dimension."""
        # x is (Batch, Horizon)
        if self.sigma <= 0:
            return x
            
        kernel_size = int(6 * self.sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Create kernel
        x_axis = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        kernel = torch.exp(-0.5 * (x_axis / self.sigma) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1).to(x.device)
        
        # Pad and convolve
        x_padded = F.pad(x.unsqueeze(1), (kernel_size // 2, kernel_size // 2), mode='replicate')
        smoothed = F.conv1d(x_padded, kernel)
        return smoothed.squeeze(1)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculates combined sharpness loss with Gaussian tolerance.

        Args:
            y_pred: Expected forecast (Batch, Horizon).
            y_true: Ground truth (Batch, Horizon).

        Returns:
            Scalar loss.
        """
        # Apply smoothing to provide temporal tolerance
        y_pred_smooth = self._apply_gaussian_smoothing(y_pred)
        y_true_smooth = self._apply_gaussian_smoothing(y_true)

        # 1. Total Magnitude (Volume) Loss
        vol_pred = y_pred.sum(dim=1)
        vol_true = y_true.sum(dim=1)
        loss_vol = torch.mean(torch.abs(vol_pred - vol_true))

        # 2. Variance Penalty (Encourage Spikiness)
        var_pred = y_pred.var(dim=1)
        var_true = y_true.var(dim=1)
        loss_var = torch.mean(torch.relu(var_true - var_pred))

        # 3. Temporal Correlation (Shape) Loss on SMOOTHED signals
        # Normalize
        yp_norm = y_pred_smooth - y_pred_smooth.mean(dim=1, keepdim=True)
        yt_norm = y_true_smooth - y_true_smooth.mean(dim=1, keepdim=True)
        
        # Covariance
        cov = (yp_norm * yt_norm).sum(dim=1)
        # Standard deviations
        std_p = torch.sqrt((yp_norm ** 2).sum(dim=1) + 1e-8)
        std_t = torch.sqrt((yt_norm ** 2).sum(yt_norm.dim() - 1) + 1e-8)
        
        corr = cov / (std_p * std_t)
        loss_shape = torch.mean(1 - corr)

        return (
            self.volume_weight * loss_vol + 
            self.variance_weight * loss_var + 
            self.shape_weight * loss_shape
        )
