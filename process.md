# Engineering Blueprint & Interactive Roadmap

## Data Source & Tensor Mapping Strategy

We are utilizing the **PKDD'99 Financial Dataset** (Czech bank transactions) or the simulated **COFINFAD (2026)** dataset.

- **Objective:** Map raw relational transaction logs into a 3D Tensor format: `(N, Time, Features)`.
- **N (Batch/Accounts):** Unique Account IDs.
- **Time (Sequence Length):** Daily aggregated time steps (e.g., T = 90 days of history to predict H = 14 days ahead).
- **Features:**
  - Historical Balances.
  - Inflow/Outflow sums.
  - Categorical embeddings (Day of week, Month).
  - Rolling statistics (7-day mean, 30-day variance).

## Technical Checklist

_Agent Instructions: Mark these with an `[x]` as they are completed._

- [x] **Step 1: Project Scaffolding & Docker Setup.** Create directory structure (`src/`, `data/`, `tests/`), `requirements.txt`, and a multi-stage `Dockerfile`.
- [x] **Step 2: Data Ingestion Pipeline.** Write a modular class to download, clean, and parse PKDD'99/COFINFAD CSV files.
- [x] **Step 3: Feature Engineering (Temporal).** Implement Cyclical Time Encodings (Sine/Cosine transforms for day/month) and zero-inflation handling mechanisms.
- [x] **Step 4: 3D Tensor Transformation.** Build the pipeline to pivot the flat data into the `(N, Time, Features)` PyTorch tensor format.
- [x] **Step 5: PyTorch Dataset & DataLoader.** Implement custom `Dataset` class supporting sliding window generation and variable-length sequence padding if necessary.
- [x] **Step 6: Core Model Architecture.** Implement the base Sequence-to-Sequence model (combining concepts from TFT/N-BEATS) using `torch.nn`. Include Multi-Head Attention or Dilated Convolutions.
- [x] **Step 7: Loss Function Implementation.** Code the Quantile (Pinball) Loss function as a custom `nn.Module` to predict 10th, 50th, and 90th percentiles.
- [x] **Step 8: Training Loop.** Create the training and validation loops, incorporating gradient clipping, learning rate scheduling (Cosine Annealing), and Early Stopping.
- [x] **Step 9: Conformal Calibration Module.** Implement Split Conformal Prediction. Compute non-conformity scores on a hold-out set to guarantee distribution-free coverage.
- [x] **Step 10: Evaluation & Metrics.** Implement PICP (Prediction Interval Coverage Probability) and MPIW (Mean Prediction Interval Width) calculators.
- [x] **Step 11: Inference API.** Wrap the trained, calibrated model into a clean inference class capable of taking a raw transaction list and outputting calibrated JSON predictions.

## Implementation Log

### Step 1: Project Scaffolding & Docker Setup
- **Directory Structure:** Established a standard enterprise layout. `src/` for source code, `data/` for raw and processed datasets (to be ignored by git), and `tests/` for unit and integration testing to ensure high coverage and reliability.
- **Docker Architecture:** Implemented a **multi-stage Dockerfile**.
    - *Stage 1 (Builder):* Uses `python:3.10-slim` to install build-essential and compile Python dependencies. This keeps the build environment isolated and ensures all necessary build-time tools are available without bloating the final image.
    - *Stage 2 (Runner):* Copies only the installed site-packages (from `/root/.local`) and the application code into a fresh `python:3.10-slim` image. 
    - *Rationale:* This significantly reduces the attack surface and the final image size (by excluding build tools like GCC). For a banking environment (TD Layer 6), this minimizes vulnerabilities and optimizes deployment speed across Kubernetes clusters.
- **Dependencies:** Specified version-pinned requirements (`torch>=2.0.0`, etc.) to ensure environment reproducibility, a critical requirement for research-to-production pipelines where training results must be deterministic.

### Step 2: Data Ingestion Pipeline
- **Modular Design:** Implemented an abstract base class `DataIngestor` to define a standard interface for all ingestion tasks. This follows the **Dependency Inversion Principle**, allowing the system to swap data sources (e.g., from CSV to a SQL database or a cloud stream) without modifying the downstream logic.
- **Data Cleaning Strategy:** The `FinancialDataIngestor` class centralizes deduplication and type safety.
    - *Zero-Inflation & Missingness:* Adopted a strict dropping policy for rows missing critical identifiers (`account_id`, `date`, `amount`) during the ingestion phase. This ensures that the deep learning model later receives high-signal, clean sequences.
    - *Type Safety:** Explicitly cast dates to `pd.to_datetime` and amounts to numeric, using `coerce` to handle malformed strings gracefully.
- **Performance:** Used `low_memory=False` in `pd.read_csv` to ensure consistent type inference across large files, preventing the "mixed type" warnings common in large financial logs.
- **Verification:** Implemented unit tests in `tests/test_data_ingestion.py` covering successful loads, deduplication logic, and missing value handling, achieving initial validation of the data contract.

### Step 3: Feature Engineering (Temporal)
- **Cyclical Time Encodings:** Implemented Sine/Cosine transforms for `day_of_week` and `month`. 
    - *Technical Why:* Traditional integer encoding (0-6 for days) introduces a false discontinuity between Sunday (6) and Monday (0). Cyclical encoding maps these to a unit circle, preserving the temporal proximity required for neural networks to learn weekly/monthly seasonalities effectively.
- **Rolling Statistics:** Integrated rolling mean and standard deviation windows (7-day and 30-day). This provides the model with local "context" and volatility measures, which are predictive of future balance shifts in liquidity forecasting.
- **Zero-Inflation Handling:** Applied `log1p` (log(1+x)) transformation to transaction amounts. 
    - *Technical Why:* Financial data is often heavily skewed with a high frequency of zero-transaction days (zero-inflation). Log-scaling stabilizes variance across orders of magnitude and reduces the impact of extreme outliers, improving the convergence of the subsequent Deep Learning model.
- **Verification:** Verified mathematical properties (e.g., $sin^2 + cos^2 = 1$) and rolling window logic through unit tests in `tests/test_feature_engineering.py`.

### Step 4: 3D Tensor Transformation
- **Temporal Alignment:** Developed the `TensorTransformer` to solve the heterogeneity of financial logs. Relational databases store transactions as events; however, Deep Learning models require dense, equidistant time steps.
- **Reindexing & Continuity:** Implemented a robust reindexing mechanism using `pd.date_range`. For every account, the system generates a continuous daily grid.
    - *Missing Data Strategy:* Missing transaction days are explicitly filled with `0.0` for amounts. This is a critical architectural choice for bank cash-flow modeling, as a "missing" record in a ledger signifies zero activity, not "missing information" in the traditional sense.
- **3D Pivot Logic:** Aggregated transactions by day and pivoted the data into the `(N, Time, Features)` format. This 3D structure is the standard input for multi-account sequence models (like TFT), enabling batch processing across thousands of independent account histories simultaneously.
- **Performance:** Utilized NumPy stacking for efficient memory allocation when converting from list-of-arrays to the final 3D Torch tensor.

### Step 5: PyTorch Dataset & DataLoader
- **Sliding Window Implementation:** Developed `CashFlowDataset` to convert static 3D tensors into dynamic training samples.
    - *Multi-Segment Learning:* Instead of treating each account as a single sample, the sliding window logic ($T_{history} + T_{horizon}$) allows the model to learn from multiple temporal offsets per account. This significantly augments the training data and helps the model generalize across different phases of a month/quarter.
- **Input-Target Decoupling:** The dataset yields pairs $(X, y)$, where $X$ contains all features (amount, cyclical time, rolling stats) for the look-back window, and $y$ contains only the target variable (log-amount) for the forecast horizon.
- **Batching Strategy:** Implemented a factory function `create_dataloader` to standardize batching and shuffling. This abstraction ensures that downstream training code remains decoupled from the specific dataset indexing logic.
- **Verification:** Unit tests in `tests/test_dataset.py` confirmed the mathematical correctness of the window indices and the integrity of the $(X, y)$ alignment.

### Step 6: Core Model Architecture
- **Multi-Scale Temporal Encoding:** Implemented a `TemporalEncoder` using **Dilated 1D Convolutions**.
    - *Technical Why:* Dilated convolutions provide an exponential increase in the receptive field without a linear increase in parameter count or depth. This is superior to standard LSTMs for financial time series as it captures both short-term noise and long-term seasonality (monthly trends) while avoiding the vanishing gradient problems associated with long-sequence RNNs.
- **Global Attention Mechanism:** Integrated `nn.MultiheadAttention` after the encoder.
    - *Technical Why:* While convolutions extract local features, the attention mechanism allows the model to "look back" and weigh specific historical days (e.g., previous month-end or paydays) more heavily when predicting future cash flows.
- **Quantile Projection Head:** Designed the final linear layer to project the hidden context into a $(Horizon \times NumQuantiles)$ space.
    - *Architecture Choice:* Predicting multiple quantiles ($10^{th}, 50^{th}, 90^{th}$) simultaneously allows the model to learn the shape of the uncertainty distribution directly from the data, rather than assuming Gaussianity. This is critical for high-volatility banking data.
- **Verification:** Unit tests in `tests/test_model.py` verified the tensor flow, output shapes, and backward pass (gradient flow), ensuring the model is ready for training.

### Step 7: Loss Function Implementation
- **Pinball (Quantile) Loss:** Implemented `QuantileLoss` as a custom PyTorch module.
    - *Mathematical Why:* Standard MSE (Mean Squared Error) encourages the model to predict the mean, which is insufficient for risk management. The Pinball loss penalizes under-forecasts and over-forecasts asymmetrically (weighted by $\tau$). For example, at the $0.9$ quantile, an under-forecast is penalized 9x more than an over-forecast of the same magnitude, forcing the model to identify the upper bound of the cash flow.
- **Vectorized Multi-Quantile Support:** Optimized the loss calculation to handle an arbitrary list of quantiles simultaneously. This enables the model to share representations in the hidden layers while specializing the output for different confidence levels.
- **Verification:** Verified asymmetric penalty behavior and symmetry with MAE (at $\tau=0.5$) via unit tests in `tests/test_loss.py`.

### Step 8: Training Loop
- **Robust Orchestration:** Implemented the `Trainer` class in `src/trainer.py` to handle the model's lifecycle.
- **Regularization & Stability:**
    - *Gradient Clipping:** Integrated `nn.utils.clip_grad_norm_`. 
        - *Technical Why:* Sequence models are prone to exploding gradients. Clipping ensures the norm of the gradients stays within a set threshold, preventing the catastrophic "divergence" during training.
    - *Early Stopping:** Coded an `EarlyStopping` class to monitor validation loss. This prevents overfitting by terminating training once the model ceases to improve on unseen data.
- **Optimization Strategy:** Designed the `fit` method to support Cosine Annealing scheduling.
    - *Technical Why:* Learning rate decay is essential for fine-tuning near the global minimum. Cosine Annealing provides a smooth, periodic decay that helps escape local minima better than standard step-decay.
- **Verification:** Unit tests in `tests/test_trainer.py` confirmed weight updates, early stopping triggers, and the stability of the loop under gradient clipping.

### Step 9: Conformal Calibration Module
- **Split Conformal Prediction:** Implemented the `ConformalCalibrator` in `src/calibration.py`.
    - *The Problem:* Neural networks, even those trained with quantile loss, are often overconfident or poorly calibrated due to the heuristic nature of their output layers.
    - *The Solution:* We use a hold-out calibration set to calculate **non-conformity scores** ($E_i = \max(\hat{y}_{low} - y_i, y_i - \hat{y}_{high})$). We then find the empirical quantile $\hat{q}$ of these scores.
    - *Coverage Guarantee:* By expanding the predicted intervals by $\hat{q}$, we provide a **mathematical guarantee** that future cash flow values will fall within our bounds with $1-\alpha$ probability (e.g., 90%), regardless of the underlying data distribution.
- **Trustworthy AI Architecture:** This post-hoc calibration step ensures that the banking system can rely on the confidence intervals for capital allocation and risk modeling, fulfilling a key TD Layer 6 requirement.
- **Verification:** Unit tests in `tests/test_calibration.py` confirmed that the calibrator correctly calculates $q\_hat$ and appropriately expands heuristic intervals.

### Step 10: Evaluation & Metrics
- **Probabilistic Verification:** Implemented the `ProbabilisticEvaluator` in `src/metrics.py`.
- **Reliability vs. Sharpness:**
    - *PICP (Prediction Interval Coverage Probability):* Measures the percentage of actual outcomes captured within the calibrated bounds. In a banking context, if PICP < $1-\alpha$, the model is underestimating risk, which could lead to liquidity shortfalls.
    - *MPIW (Mean Prediction Interval Width):* Measures the "cost" of certainty. An infinitely wide interval has perfect coverage but zero utility. MPIW allows us to compare models and select the one that provides the tightest (sharpest) bounds while still maintaining the target coverage.
- **Verification:** Unit tests in `tests/test_metrics.py` confirmed the mathematical accuracy of coverage and width calculations, ensuring the system can be objectively benchmarked against Basel III liquidity standards.

### Step 11: Inference API
- **End-to-End Orchestration:** Developed the `CashFlowInferenceAgent` in `src/inference.py`.
    - *Production Design:* This class encapsulates the entire complexity of the pipeline. A user or a downstream microservice only needs to provide raw transaction logs; the agent handles ingestion, feature engineering, tensorization, and calibrated probabilistic forecasting.
- **Output Standardization:** Standardized the output format to JSON, providing point estimates (median) alongside statistically guaranteed 90% confidence intervals. This format is ready for integration into treasury dashboards or automated capital rebalancing systems.
- **Verification:** Executed a full integration test in `tests/test_inference.py`, confirming that the entire software stack—from data cleaning to conformal prediction—operates as a cohesive, production-grade unit.
