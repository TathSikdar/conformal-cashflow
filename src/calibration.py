"""Module for Conformal Calibration and uncertainty quantification."""

import numpy as np
import torch
import torch.nn as nn


class ConformalCalibrator:
    """Implements Split Conformal Prediction for interval calibration.

    This class computes non-conformity scores on a calibration set and
    uses them to provide distribution-free coverage guarantees for future
    predictions.

    Attributes:
        q_hat: The computed non-conformity score quantile.
        alpha: The target miscoverage rate (e.g., 0.1 for 90% coverage).
    """

    def __init__(self, alpha: float = 0.1) -> None:
        """Initializes the calibrator.

        Args:
            alpha: Significance level (1 - confidence). Default 0.1 (90% CI).
        """
        self.alpha = alpha
        self.q_hat: float = 0.0

    def calibrate(self, model: nn.Module, cal_loader: torch.utils.data.DataLoader, device: str = "cpu") -> float:
        """Computes the q_hat quantile of non-conformity scores.

        Args:
            model: The trained probabilistic model.
            cal_loader: DataLoader for the hold-out calibration set.
            device: Device to run inference on.

        Returns:
            The calculated q_hat value.
        """
        model.eval()
        scores = []

        with torch.no_grad():
            for x, future_x, acc_idx, y in cal_loader:
                x, future_x, acc_idx, y = (
                    x.to(device), 
                    future_x.to(device), 
                    acc_idx.to(device), 
                    y.to(device)
                )
                
                # Forward with all inputs
                probs, magnitudes = model(x, future_x, acc_idx)
                
                # Apply Hurdle weighting
                preds = magnitudes * probs.unsqueeze(-1)
                
                y_low = preds[:, :, 0]
                y_high = preds[:, :, 2]
                
                # Non-conformity score
                batch_scores = torch.max(y_low - y, y - y_high)
                scores.append(batch_scores.cpu().numpy().flatten())

        all_scores = np.concatenate(scores)
        n = len(all_scores)
        
        # Calculate q_hat: (n+1)(1-alpha)/n empirical quantile
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        # Clip q_level to [0, 1] to avoid index errors on small sets
        q_level = min(max(q_level, 0.0), 1.0)
        
        self.q_hat = np.quantile(all_scores, q_level, method="higher")
        return self.q_hat

    def predict(
        self, 
        model: nn.Module, 
        x: torch.Tensor, 
        future_x: torch.Tensor, 
        account_idx: torch.Tensor,
        device: str = "cpu"
    ) -> torch.Tensor:
        """Generates calibrated prediction intervals.

        Args:
            model: The trained model.
            x: Input tensor (Batch, History, F).
            future_x: Future exogenous features (Batch, Horizon, FutureDim).
            account_idx: Account indices (Batch,).
            device: Execution device.

        Returns:
            Calibrated intervals of shape (Batch, Horizon, 2).
        """
        model.eval()
        with torch.no_grad():
            x, future_x, account_idx = (
                x.to(device), 
                future_x.to(device), 
                account_idx.to(device)
            )
            
            probs, magnitudes = model(x, future_x, account_idx)
            preds = magnitudes * probs.unsqueeze(-1) # (Batch, Horizon, 3)
            
            y_low = preds[:, :, 0]
            y_high = preds[:, :, 2]
            
            # Calibrated intervals: [y_low - q_hat, y_high + q_hat]
            cal_low = y_low - self.q_hat
            cal_high = y_high + self.q_hat
            
            return torch.stack([cal_low, cal_high], dim=-1)
