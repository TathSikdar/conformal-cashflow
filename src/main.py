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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    # 1. Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    history_size = 60
    horizon = 14
    batch_size = 32
    hidden_dim = 64
    quantiles = [0.1, 0.5, 0.9]
    feature_cols = ["amount_log", "day_sin", "day_cos", "month_sin", "month_cos"]

    logging.info(f"Starting pipeline on device: {device}")

    # 2. Data Ingestion & Engineering
    logging.info("Loading PKDD'99 Berka Dataset...")
    ingestor = FinancialDataIngestor()
    engineer = TemporalFeatureEngineer()
    
    # Load raw data
    raw_df = pd.read_csv("data/trans.csv.gz", compression="gzip", low_memory=False)
    
    # Pre-processing for Berka Dataset: Sign the amounts
    # PRIJEM = credit, VYDAJ = debit, VYBER = withdrawal
    logging.info("Signing amounts based on transaction type...")
    raw_df["amount"] = raw_df.apply(
        lambda x: x["amount"] if x["type"] == "PRIJEM" else -x["amount"], axis=1
    )
    
    # Filter for a subset of accounts to speed up demo (e.g., first 100 accounts)
    subset_accounts = raw_df["account_id"].unique()[:100]
    df = raw_df[raw_df["account_id"].isin(subset_accounts)].copy()
    
    df = ingestor.clean_data(df)
    df = engineer.pipeline(df)

    # 3. Tensor Transformation & Data Loading
    transformer = TensorTransformer(feature_cols=feature_cols, sequence_length=history_size + horizon)
    full_tensor = transformer.transform(df) # (N, T, F)

    # Split into Train and Calibration sets (80/20 split)
    n_train = int(len(full_tensor) * 0.8)
    train_data = full_tensor[:n_train] 
    cal_data = full_tensor[n_train:]

    train_loader = create_dataloader(train_data, batch_size=batch_size, history_size=history_size, horizon=horizon)
    cal_loader = create_dataloader(cal_data, batch_size=batch_size, history_size=history_size, horizon=horizon)

    # 4. Model Training
    model = ProbabilisticForecaster(
        input_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        history_size=history_size,
        horizon=horizon,
        num_quantiles=len(quantiles)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = QuantileLoss(quantiles=quantiles)
    
    trainer = Trainer(model, optimizer, criterion, device=device)
    logging.info("Training model...")
    trainer.fit(train_loader, train_loader, epochs=5) # Using train as val for demo brevity

    # 5. Conformal Calibration
    logging.info("Calibrating uncertainty intervals...")
    calibrator = ConformalCalibrator(alpha=0.1) # 90% Confidence
    calibrator.calibrate(model, cal_loader, device=device)

    # 6. Production Inference Agent
    agent = CashFlowInferenceAgent(
        model=model,
        calibrator=calibrator,
        feature_cols=feature_cols,
        sequence_length=history_size,
        horizon=horizon,
        device=device
    )

    # Predict for the first account in our subset
    target_account = subset_accounts[0]
    sample_raw_data = df[df["account_id"] == target_account].tail(history_size)
    forecast = agent.predict_calibrated(sample_raw_data)
    
    logging.info("Final Calibrated Forecast (First 3 steps):")
    print(agent.to_json(forecast["predictions"][:3]))

if __name__ == "__main__":
    main()
