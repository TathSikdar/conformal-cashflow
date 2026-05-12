"""Module for the production Inference API agent."""

import json
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
import torch.nn as nn

from src.calibration import ConformalCalibrator
from src.data_ingestion import FinancialDataIngestor
from src.feature_engineering import TemporalFeatureEngineer
from src.model import ProbabilisticForecaster
from src.tensor_transformation import TensorTransformer


class CashFlowInferenceAgent:
    """Production agent for generating calibrated cash flow forecasts.

    Orchestrates the full pipeline: Ingestion -> Features -> Tensors -> 
    Prediction -> Calibration -> JSON Output.
    """

    def __init__(
        self,
        model: nn.Module,
        calibrator: ConformalCalibrator,
        feature_cols: List[str],
        sequence_length: int = 60,
        horizon: int = 14,
        device: str = "cpu",
    ) -> None:
        """Initializes the inference agent.

        Args:
            model: Trained ProbabilisticForecaster model.
            calibrator: Fitted ConformalCalibrator.
            feature_cols: List of feature names used during training.
            sequence_length: Look-back window size.
            horizon: Prediction horizon.
            device: Execution device.
        """
        self.model = model.to(device)
        self.calibrator = calibrator
        self.device = device
        self.horizon = horizon
        
        # Initialize pipeline components
        self.ingestor = FinancialDataIngestor()
        self.engineer = TemporalFeatureEngineer()
        self.transformer = TensorTransformer(
            feature_cols=feature_cols, 
            sequence_length=sequence_length
        )

    def predict_calibrated(self, raw_data: Union[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generates calibrated forecasts from raw transaction data.

        Args:
            raw_data: Path to CSV or a pandas DataFrame.

        Returns:
            Dictionary containing calibrated forecasts and metadata.
        """
        # 1. Ingestion
        if isinstance(raw_data, str):
            df = self.ingestor.load_data(raw_data)
        else:
            df = raw_data

        # 2. Cleaning & Features
        df = self.ingestor.clean_data(df)
        df = self.engineer.pipeline(df)

        # 3. Tensor Transformation
        # transform returns (N, T, F). We assume a single account for point-inference.
        x_tensor = self.transformer.transform(df)
        
        # 4. Inference & Calibration
        # Conformal prediction expands the intervals
        cal_intervals = self.calibrator.predict(self.model, x_tensor, device=self.device)
        
        # Get point forecast (median) from raw model
        self.model.eval()
        with torch.no_grad():
            raw_preds = self.model(x_tensor.to(self.device))
            median_forecast = raw_preds[:, :, 1] # Index 1 is the 50th percentile

        # 5. Formatting Output
        # Taking the first account in the batch
        output = {
            "forecast_horizon": self.horizon,
            "predictions": []
        }

        low_bounds = cal_intervals[0, :, 0].cpu().numpy()
        high_bounds = cal_intervals[0, :, 1].cpu().numpy()
        median_vals = median_forecast[0].cpu().numpy()

        for i in range(self.horizon):
            output["predictions"].append({
                "step": i + 1,
                "median": float(median_vals[i]),
                "lower_90": float(low_bounds[i]),
                "upper_90": float(high_bounds[i])
            })

        return output

    def to_json(self, forecast: Dict[str, Any]) -> str:
        """Serializes the forecast dictionary to a JSON string.

        Args:
            forecast: The forecast dictionary.

        Returns:
            JSON string.
        """
        return json.dumps(forecast, indent=2)
