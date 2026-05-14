"""Module for visualizing probabilistic forecasts with historical context."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

def plot_forecast(
    forecast: Dict[str, Any], 
    history_df: Optional[pd.DataFrame] = None,
    save_path: str = "forecast.png"
) -> None:
    """Plots the median forecast, confidence bands, and optional history.

    Args:
        forecast: The dictionary returned by CashFlowInferenceAgent.
        history_df: Optional DataFrame containing recent transaction history.
        save_path: Path to save the resulting image.
    """
    # 1. Extract Forecast Data
    f_steps = [p["step"] for p in forecast["predictions"]]
    f_medians = [p["median"] for p in forecast["predictions"]]
    f_lowers = [p["lower_90"] for p in forecast["predictions"]]
    f_uppers = [p["upper_90"] for p in forecast["predictions"]]

    plt.figure(figsize=(14, 7))
    
    # 2. Plot History (if provided)
    if history_df is not None:
        # We align history to end at step 0
        h_vals = history_df["amount_log"].values
        h_steps = np.arange(-len(h_vals) + 1, 1)
        plt.plot(h_steps, h_vals, label="Historical Activity", color="black", alpha=0.5, linestyle='--')
        
    # 3. Plot Forecast
    plt.plot(f_steps, f_medians, label="Median Forecast", color="blue", marker="o", linewidth=2)
    plt.fill_between(f_steps, f_lowers, f_uppers, color="blue", alpha=0.2, label="90% Conformal Interval")
    
    # Formatting
    plt.axvline(0, color='red', linestyle='-', alpha=0.3, label="Forecast Start")
    plt.axhline(0, color='black', linestyle='-', alpha=0.2)
    
    plt.title("Executive Cash Flow Dashboard: History & Calibrated Forecast")
    plt.xlabel("Days (Relative to Today)")
    plt.ylabel("Transaction Magnitude (Signed Log-Scale)")
    plt.legend(loc="upper left")
    plt.grid(True, which='both', alpha=0.3)
    
    # Add currency guidance in text box
    info_text = "Log-Scale Guide:\n3.0 ≈ 19 units\n6.0 ≈ 400 units\n9.0 ≈ 8,000 units"
    plt.text(0.95, 0.05, info_text, transform=plt.gca().transAxes, 
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Executive visualization saved to {save_path}")
