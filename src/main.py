"""Main entry point for the Probabilistic Cash Flow Agent."""

import logging
import pandas as pd
import torch
from src.data_ingestion import FinancialDataIngestor
from src.feature_engineering import TemporalFeatureEngineer
from src.tensor_transformation import TensorTransformer
from src.dataset import create_dataloader
from src.model import ProbabilisticForecaster
from src.loss import QuantileLoss
from src.trainer import Trainer
from src.calibration import ConformalCalibrator
from src.inference import CashFlowInferenceAgent
from src.visualization import plot_forecast

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    # 1. Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    history_size = 60
    horizon = 14
    batch_size = 32
    hidden_dim = 128
    num_layers = 5
    quantiles = [0.1, 0.5, 0.9]
    feature_cols = [
        "amount_log", "balance_log", 
        "day_sin", "day_cos", 
        "month_sin", "month_cos",
        "dom_sin", "dom_cos",
        "is_weekend", "is_month_end",
        "days_to_15th", "days_to_month_end",
        "spectral_top_1", "spectral_top_2",
        "amount_lag_1", "amount_lag_7",
        "amount_lag_14", "amount_lag_28", "amount_lag_30",
        "rolling_mean_7", "rolling_std_7"
    ]
    # Indices of calendar-based features that are known in the future
    # Features 2-11 are calendar-based
    exog_indices = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    logging.info(f"Starting pipeline on device: {device}")

    # 2. Data Ingestion & Engineering
    logging.info("Loading PKDD'99 Berka Dataset...")
    ingestor = FinancialDataIngestor()
    engineer = TemporalFeatureEngineer()
    
    # Load raw data
    raw_df = pd.read_csv("data/trans.csv.gz", compression="gzip", low_memory=False)
    
    # Pre-processing for Berka Dataset: Sign the amounts
    logging.info("Signing amounts and filtering active accounts...")
    df = ingestor.preprocess_berka(raw_df)
    
    # FILTER: Only accounts with > 200 transactions total (High Signal)
    activity_counts = df.groupby("account_id").size()
    active_account_ids = activity_counts[activity_counts > 200].index
    
    # SCALE: Use all active accounts
    df = df[df["account_id"].isin(active_account_ids)].copy()
    
    df = ingestor.clean_data(df)
    df = engineer.pipeline(df)

    # 3. Tensor Transformation & Data Loading
    transformer = TensorTransformer(feature_cols=feature_cols, sequence_length=history_size + horizon)
    full_tensor, account_ids = transformer.transform(df) # (N, T, F), (N,)
    
    # Map account_id to integer index for embedding
    account_id_map = {id_val: i for i, id_val in enumerate(account_ids)}
    num_accounts = len(account_ids)

    # Split into Train, Validation, and Calibration sets (70/15/15 split)
    n = len(full_tensor)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    train_data = full_tensor[:n_train]
    val_data = full_tensor[n_train:n_train+n_val]
    cal_data = full_tensor[n_train+n_val:]

    # Use multi-process loading and pinned memory for acceleration
    num_workers = 4 # Using 4 CPU cores for data loading
    pin_memory = True if device == "cuda" else False

    train_loader = create_dataloader(
        train_data, 
        batch_size=batch_size, 
        history_size=history_size, 
        horizon=horizon, 
        exog_indices=exog_indices,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = create_dataloader(
        val_data, 
        batch_size=batch_size, 
        history_size=history_size, 
        horizon=horizon, 
        exog_indices=exog_indices,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    cal_loader = create_dataloader(
        cal_data, 
        batch_size=batch_size, 
        history_size=history_size, 
        horizon=horizon, 
        exog_indices=exog_indices,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # 4. Model Training (Auto-Iterative Sharpness Search)
    max_retries = 3
    current_sharpness_weight = 2.0
    model_final = None
    
    for attempt in range(max_retries):
        logging.info(f"--- Training Attempt {attempt+1}/{max_retries} (Sharpness Weight: {current_sharpness_weight}) ---")
        
        model = ProbabilisticForecaster(
            input_dim=len(feature_cols),
            num_accounts=num_accounts,
            hidden_dim=hidden_dim,
            history_size=history_size,
            horizon=horizon,
            num_quantiles=len(quantiles),
            num_layers=num_layers,
            future_dim=len(exog_indices)
        )
        
        # On retries, use a higher LR to 'jolt' the model out of the flat-line local minimum
        lr = 5e-4 if attempt == 0 else 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = QuantileLoss(quantiles=quantiles)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        trainer = Trainer(
            model, 
            optimizer, 
            criterion, 
            scheduler=scheduler, 
            device=device,
            sharpness_weight=current_sharpness_weight
        )
        
        trainer.fit(train_loader, val_loader, epochs=100, early_stopping_patience=20)
        
        # VALIDATE SHARPNESS: Ensure the model isn't predicting a flat line
        model.eval()
        with torch.no_grad():
            # Check a batch from validation
            vx, vf, va, vy = next(iter(val_loader))
            v_probs, _ = model(vx.to(device), vf.to(device), va.to(device))
            
            # Variance across the horizon (Batch-averaged)
            # If variance is near 0, the model is 'flat'
            avg_var = v_probs.var(dim=1).mean().item()
            logging.info(f"Validation Sharpness (Variance): {avg_var:.6f}")
            
            # Threshold for 'straight-line' rejection
            if avg_var > 0.005 or attempt == max_retries - 1:
                logging.info("Model passed sharpness validation or reached max retries.")
                model_final = model
                break
            else:
                logging.warning("REJECTED: Model produced a straight-line forecast. Increasing penalty and retraining...")
                current_sharpness_weight *= 2.5 # Aggressive increase

    # 5. Conformal Calibration
    logging.info("Calibrating uncertainty intervals...")
    calibrator = ConformalCalibrator(alpha=0.1) # 90% Confidence
    calibrator.calibrate(model_final, cal_loader, device=device)

    # 6. Production Inference Agent
    agent = CashFlowInferenceAgent(
        model=model_final,
        calibrator=calibrator,
        feature_cols=feature_cols,
        account_id_map=account_id_map,
        sequence_length=history_size,
        horizon=horizon,
        device=device
    )

    # Predict for the first account in our dataset
    target_account = account_ids[0]
    # IMPORTANT: Pass RAW data to the agent as it runs its own pipeline
    full_account_raw = raw_df[raw_df["account_id"] == target_account]
    
    # Use only the tail for the inference agent input
    sample_raw_data = full_account_raw.tail(history_size)
    forecast = agent.predict_calibrated(sample_raw_data)
    
    logging.info("Final Calibrated Forecast (First 3 steps):")
    print(agent.to_json(forecast["predictions"][:3]))

    # 7. Visualization with History Context
    # We pass the last 30 days of history for the plot context
    # Note: We use the engineered df for plotting as it contains amount_log
    full_account_engineered = df[df["account_id"] == target_account]
    history_context = full_account_engineered.tail(history_size + 30).head(30)
    plot_forecast(forecast, history_df=history_context, save_path="data/forecast_viz.png")

if __name__ == "__main__":
    main()
