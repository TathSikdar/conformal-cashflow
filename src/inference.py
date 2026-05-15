"""Module for the production Inference API agent."""

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
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
        account_id_map: Dict[Any, int],
        sequence_length: int = 60,
        horizon: int = 14,
        device: str = "cpu",
        exog_indices: list[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ) -> None:
        """Initializes the inference agent.

        Args:
            model: Trained ProbabilisticForecaster model.
            calibrator: Fitted ConformalCalibrator.
            feature_cols: List of feature names used during training.
            account_id_map: Map of account_id to integer index.
            sequence_length: Look-back window size.
            horizon: Prediction horizon.
            device: Execution device.
            exog_indices: Feature indices for future guidance.
        """
        self.model = model.to(device)
        self.calibrator = calibrator
        self.device = device
        self.horizon = horizon
        self.account_id_map = account_id_map
        self.exog_indices = exog_indices
        
        # Initialize pipeline components
        self.ingestor = FinancialDataIngestor()
        self.engineer = TemporalFeatureEngineer()
        self.transformer = TensorTransformer(
            feature_cols=feature_cols, 
            sequence_length=sequence_length
        )

    def _get_future_features(self, last_date: pd.Timestamp) -> torch.Tensor:
        """Generates future calendar features for the horizon."""
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=self.horizon, 
            freq="D"
        )
        # Create a dummy DataFrame to run through the feature engineer
        dummy_df = pd.DataFrame({"date": future_dates})
        dummy_df["account_id"] = 0 # Dummy
        dummy_df["amount"] = 0.0 # Dummy
        
        # We only need cyclical features and calendar flags
        dummy_df = self.engineer.add_cyclical_features(dummy_df)
        dummy_df = self.engineer.add_calendar_flags(dummy_df)
        
        # Extract the same exogenous features used in training
        exog_cols = [
            "day_sin", "day_cos", "month_sin", "month_cos", 
            "dom_sin", "dom_cos", "is_weekend", "is_month_end",
            "days_to_15th", "days_to_month_end"
        ]
        
        # Filter to only include columns that are actually in feature_cols
        full_cols = self.transformer.feature_cols
        exog_cols = [c for c in exog_cols if c in full_cols]
        
        np_exog = dummy_df[exog_cols].values
        
        if self.transformer.scaler:
            # The global scaler is only fitted on non-target features (idx 1:)
            # We create a dummy block for all features EXCEPT the target
            num_other_features = len(full_cols) - 1
            full_row_others = np.zeros((self.horizon, num_other_features))
            
            for i, col in enumerate(exog_cols):
                idx = full_cols.index(col)
                if idx > 0:
                    # Offset by 1 because scaler only knows features 1:
                    full_row_others[:, idx - 1] = np_exog[:, i]
            
            # Transform just the non-target block
            scaled_others = self.transformer.scaler.transform(full_row_others)
            
            # Extract just the exogenous parts back
            standardized_exog = []
            for col in exog_cols:
                idx = full_cols.index(col)
                # Exogenous features are never the target (idx 0), but we check for safety
                if idx > 0:
                    standardized_exog.append(scaled_others[:, idx - 1])
                else:
                    standardized_exog.append(np_exog[:, exog_cols.index(col)])
            
            np_exog = np.stack(standardized_exog, axis=1)

        return torch.from_numpy(np_exog).float().unsqueeze(0) # (1, H, FutureDim)

    def predict_calibrated(self, raw_data: pd.DataFrame) -> Dict[str, Any]:
        """Generates calibrated forecasts from raw transaction data.

        Args:
            raw_data: Pandas DataFrame for a SINGLE account.

        Returns:
            Dictionary containing calibrated forecasts.
        """
        # 1. Ingestion & Features
        # Apply Berka-specific signing before standard cleaning
        df = self.ingestor.preprocess_berka(raw_data)
        df = self.ingestor.clean_data(df)
        last_date = df["date"].max()
        account_id = df["account_id"].iloc[0]
        
        df = self.engineer.pipeline(df)

        # 2. Tensor Transformation
        # transform now returns (tensor, ids)
        x_tensor, _ = self.transformer.transform(df)
        
        # 3. Future Features & Account ID mapping
        future_x = self._get_future_features(last_date)
        acc_idx_int = self.account_id_map.get(account_id, 0) # Default to 0 if new
        acc_idx = torch.tensor([acc_idx_int], dtype=torch.long)
        
        # 4. Inference & Calibration
        cal_intervals = self.calibrator.predict(
            self.model, x_tensor, future_x, acc_idx, device=self.device
        )
        
        # Point forecast
        self.model.eval()
        with torch.no_grad():
            probs, magnitudes = self.model(
                x_tensor.to(self.device), 
                future_x.to(self.device), 
                acc_idx.to(self.device)
            )
            # SHARP MEDIAN: If prob < 0.4, median is 0. If prob >= 0.4, use 50th quantile.
            median_forecast = torch.where(probs > 0.4, magnitudes[:, :, 1], 0.0)

        # 5. UNSCALING (Local Scaling Inversion)
        # Point estimate
        median_vals = median_forecast[0].cpu().numpy()
        median_unscaled = self.transformer.inverse_transform_target(account_id, median_vals)
        
        # Uncertainty intervals (Calibration expands the heuristic, so we unscale the result)
        low_bounds = cal_intervals[0, :, 0].cpu().numpy()
        high_bounds = cal_intervals[0, :, 1].cpu().numpy()
        
        low_unscaled = self.transformer.inverse_transform_target(account_id, low_bounds)
        high_unscaled = self.transformer.inverse_transform_target(account_id, high_bounds)

        # 6. Formatting Output
        output = {
            "account_id": str(account_id),
            "forecast_horizon": self.horizon,
            "predictions": []
        }

        for i in range(self.horizon):
            output["predictions"].append({
                "step": i + 1,
                "median": float(median_unscaled[i]),
                "lower_90": float(low_unscaled[i]),
                "upper_90": float(high_unscaled[i])
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
