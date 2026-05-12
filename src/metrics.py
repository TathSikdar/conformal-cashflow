"""Module for evaluating probabilistic forecasts and uncertainty intervals."""

from typing import Dict

import torch


class ProbabilisticEvaluator:
    """Evaluates the quality of prediction intervals.

    Focuses on reliability (Coverage) and efficiency (Width).
    """

    @staticmethod
    def calculate_picp(y_true: torch.Tensor, y_low: torch.Tensor, y_high: torch.Tensor) -> float:
        """Calculates Prediction Interval Coverage Probability (PICP).

        PICP measures the proportion of ground truth values that fall within
        the predicted intervals.

        Args:
            y_true: Ground truth values of shape (N, Horizon).
            y_low: Lower interval bounds of shape (N, Horizon).
            y_high: Upper interval bounds of shape (N, Horizon).

        Returns:
            The coverage probability (float between 0 and 1).
        """
        within_bounds = (y_true >= y_low) & (y_true <= y_high)
        return torch.mean(within_bounds.float()).item()

    @staticmethod
    def calculate_mpiw(y_low: torch.Tensor, y_high: torch.Tensor) -> float:
        """Calculates Mean Prediction Interval Width (MPIW).

        MPIW measures the average 'sharpness' or size of the intervals.
        Smaller width is better, provided PICP is maintained.

        Args:
            y_low: Lower interval bounds.
            y_high: Upper interval bounds.

        Returns:
            The mean width of the intervals.
        """
        width = y_high - y_low
        return torch.mean(width).item()

    def evaluate(
        self, y_true: torch.Tensor, intervals: torch.Tensor
    ) -> Dict[str, float]:
        """Runs the full evaluation suite.

        Args:
            y_true: Ground truth values of shape (N, Horizon).
            intervals: Calibrated intervals of shape (N, Horizon, 2).

        Returns:
            Dictionary containing PICP and MPIW metrics.
        """
        y_low = intervals[:, :, 0]
        y_high = intervals[:, :, 1]

        picp = self.calculate_picp(y_true, y_low, y_high)
        mpiw = self.calculate_mpiw(y_low, y_high)

        return {
            "picp": picp,
            "mpiw": mpiw,
        }
