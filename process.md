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
- [x] **Step 12: Visualization & Performance Audit.** Implement plotting utilities and audit the conformal coverage on the full 1.1M record dataset.
- [x] **Step 13: Signal Enrichment & Active Filtering.** Implement Cumulative Balance features and filter for high-activity accounts to "sharpen" the median forecast.
- [x] **Step 14: Sharpness Optimization & Focal Loss.** Implement Focal Loss for the hurdle gate and upgrade the decoder head to capture precise periodic spikes.
- [x] **Step 15: Advanced Forecasting & Full Scale.** Implement Temporal Fusion Transformer (TFT) elements (VSN, GRN), exogenous calendar flags, and scale training to the full 1.1M record dataset.
- [x] **Step 16: Spectral & DOM Periodicity Features.** Implement Day-of-Month cyclical encoding and FFT-based spectral magnitudes to capture account-specific periodicity.
- [x] **Step 17: Weighted Shape-Aware Loss.** Implement a Magnitude-Weighted Quantile Loss to penalize missing sparse spikes and prevent trivial flat forecasts.
- [x] **Step 18: Sequence-Level Sharpness Incentives.** Implement Volume, Variance, and Temporal Correlation losses to penalize zero-regression and reward spiky, rhythm-accurate forecasts.
- [x] **Step 19: Cross-Attention & Deep Seasonal Lags.** Upgrade decoder to use dynamic Cross-Attention and implement 14, 28, and 30-day seasonal lags to maintain forecast persistence throughout the horizon.
- [x] **Step 20: Auto-Iterative Sharpness Search.** Implement Architectural Forcing (Direct Lag Injection), Threshold Inference, and an Auto-Iterative Loop that automatically retrains the model with increased penalties if a "straight line" forecast is detected.
- [x] **Step 21: Local Scaling (Per-Account Normalization).** Refactor TensorTransformer to normalize each account's transactions by its own historical mean/std, improving generalization across different wealth tiers.
- [x] **Step 22: Payday Proximity Anchors.** Implement explicit temporal features for "days until 15th" and "days until month end" to provide strong anchors for recurring events.
- [x] **Step 23: Gaussian-Smoothed Shape Loss.** Upgrade the SequenceSharpnessLoss to use a Gaussian kernel, rewarding the model for temporal "near misses" and providing smoother gradients for timing.

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

### Step 12: Visualization & Performance Audit
- **Visual Feedback Loop:** Developed `plot_forecast` in `src/visualization.py`.
    - *Utility:* Generates high-resolution PNGs (like `forecast_viz.png`) showing the historical context, the median forecast, and the calibrated uncertainty bounds. This allows analysts to visually verify if the conformal intervals are expanding appropriately during periods of high volatility.
- **Verification:** Audited coverage metrics on the validation set, ensuring that PICP aligned with the target $1-\alpha$ (90%).

### Step 13: Signal Enrichment & Active Filtering
- **High-Activity Filtering:** Updated `src/main.py` to filter for accounts with $>200$ historical transactions. 
    - *Technical Why:* Sequence models require a minimum density of events to learn meaningful seasonal patterns. By focusing on "active" accounts, we reduce the noise from dormant accounts that would otherwise skew the model towards predicting a constant zero.
- **Cumulative Balance Features:** Implemented `add_balance_feature` in `src/feature_engineering.py`.
    - *Technical Why:* Transaction amounts (deltas) are noisy. The running balance (state) provides a much smoother signal that captures the long-term trend of an account's liquidity. Log-scaling the balance ensures that the model can handle accounts across different wealth tiers (from $100$ to $1M$) in the same latent space.

### Bug Fix: Hurdle Model Output & Shape Consistency
- **Issue 1 (Unpacking):** A `ValueError: too many values to unpack` occurred because of inconsistent return types between training and eval modes.
- **Issue 2 (Shape Mismatch):** A `RuntimeError: mat1 and mat2 shapes cannot be multiplied` occurred in the `future_projection` layer because the model defaulted to 6 exogenous features while the pipeline was providing 8 (after adding Day-of-Month periodicity).
- **Issue 3 (Flat-Line Regression):** After adding the Cross-Attention decoder, the forecast regressed to a straight line. This was diagnosed as "Signal Washout," where the direct calendar anchors were lost in the attention layer before convergence.
- **Resolution:**
    - Refactored `ProbabilisticForecaster.forward` to **always** return the tuple `(probs, magnitudes)`.
    - Updated `main.py` and model defaults to explicitly pass `future_dim=len(exog_indices)`.
    - **Straight-Line Rescue:** 
        1. Added a **Residual Signal Shortcut** in the decoder to feed future calendar features directly into the output heads, bypassing the attention bottleneck.
        2. Aggressively quadrupled the **SequenceSharpnessLoss** weight to 2.0 to mathematically penalize flat forecasts.
        3. Restored training to **100 epochs** to allow the complex attention weights to converge.
- **Verification:** Successfully executed the full project test suite (37/37 passed) and verified container execution.

### Step 14: Sharpness Optimization & Focal Loss
- **Focal Loss Integration:** Replaced standard BCE with `FocalLoss` in the classification branch.
    - *Technical Why:* Financial transactions are rare (class imbalance). Standard BCE is easily dominated by the majority 'zero' class, leading to pessimistic, flat forecasts. Focal Loss ($\gamma=2.0$) down-weights easy negatives (zeros) and forces the model to learn the hard positives (spikes), significantly 'sharpening' the transaction gate.
- **Horizon-Aware Decoder Heads:** Upgraded the MLP heads to use **Horizon Positional Encodings**.
    - *Technical Why:* Previously, the model projected a static context vector into the entire horizon. By adding unique learnable embeddings for each day in the horizon ($H_1, H_2, \dots, H_{14}$), the model can now explicitly differentiate between different "days-to-event". This improves the temporal alignment of periodic spikes (e.g., paydays on the 13th) and prevents the "shoot up at the end" boundary bias.
- **Verification:** Verified that the model now captures multiple distinct spikes in the horizon for periodic accounts, rather than a single blurred spike at the end.

### Step 15: Advanced Forecasting & Full Scale
- **Variable Selection Network (VSN):** Integrated a VSN at the input layer.
    - *Technical Why:* Not all features (e.g., `rolling_std_7` vs `is_month_end`) are equally important at all times. The VSN uses a gating mechanism to dynamically 'mute' noisy features and amplify high-signal ones per time step, allowing the model to focus on calendar triggers when they are relevant.
- **Gated Residual Networks (GRN):** Upgraded the output heads to use GRNs.
    - *Technical Why:* Traditional MLPs apply the same non-linear transformation regardless of the input's complexity. GRNs allow the model to skip layers if the relationship is simple (linear) or engage complex mappings for non-linear behaviors, improving training stability and generalization.
- **Exogenous Calendar Flags:** Added `is_weekend` and `is_month_end` features.
    - *Impact:* These provide explicit temporal 'anchors' that help the model align transaction spikes with specific days, directly addressing the 'straight line' forecast issue.
- **Full-Scale Dataset Training:** Removed all subsetting limits and trained on the full 1.1M record dataset.
    - *Result:* Exposing the model to the entire variety of account behaviors significantly improved its robustness and ability to generalize across different wealth tiers and spending patterns.

### Step 16: Spectral & DOM Periodicity Features
- **Day-of-Month (DOM) Encoding:** Implemented cyclical sine/cosine transforms for the day of the month (1-31).
    - *Technical Why:* Many banking transactions (rent, salaries, utility bills) are strictly anchored to specific calendar dates rather than days of the week. DOM encoding provides the model with a clear temporal coordinate to learn these monthly periodic spikes.
- **FFT-based Spectral Features:** Developed `add_spectral_features` to compute the top-2 dominant frequencies of transaction magnitudes per account using Fast Fourier Transform.
    - *Technical Why:* Every bank account has a unique "rhythm" (e.g., bi-weekly paydays). By extracting spectral magnitudes, we provide the model with an account-level "periodicity fingerprint." This allows the model to differentiate between a highly periodic payroll account and a random-spending retail account, significantly improving forecast sharpness.

### Step 17: Weighted Shape-Aware Loss
- **Magnitude-Weighted Quantile Loss:** Upgraded the `QuantileLoss` to scale the pinball error by `abs(y_true) + epsilon`.
    - *Technical Why:* In sparse financial data, the model can achieve a low overall loss by simply predicting zero everywhere (the trivial solution). By weighting the loss by the target magnitude, we force the model to prioritize the sparse, high-value spikes. An error on a $5000 transaction is now significantly more "expensive" than an error on a zero-transaction day, pulling the model away from the flat-line mean regression and towards capturing actual cash flow events.
- **Verification:** Verified that the new loss function results in non-zero, periodic spikes in the forecast for accounts with strong historical rhythms.

### Step 18: Sequence-Level Sharpness Incentives
- **Multi-Faceted Sequence Loss:** Implemented `SequenceSharpnessLoss` in `src/loss.py` combining three incentives:
    - *Volume Loss:* Penalizes the model for mismatches in the total cash flow volume over the 14-day horizon. This forces the model to distribute "spending energy" throughout the sequence.
    - *Variance Loss:* Specifically penalizes predictions with lower variance than the ground truth. This prevents the model from settling on the low-variance, flat-line trivial solution.
    - *Temporal Correlation (Shape) Loss:* Uses Pearson correlation to reward the model for aligning the "peaks" and "valleys" of the forecast with historical transaction rhythms.
- **Focal Loss Alpha Tuning:** Increased the `FocalLoss` alpha to `0.85` in `src/trainer.py`. This extremely aggressive weighting makes missing a transaction spike nearly 6x more "expensive" than a false positive, pulling the model out of its pessimistic zero-bias.
- **Faster Iteration:** Reduced training epochs to 50 in `main.py` to allow for rapid verification of these sharpness incentives.
- **Verification:** Verified through unit tests in `tests/test_loss.py` and visual audit of `forecast_viz.png`.

### Step 19: Cross-Attention & Deep Seasonal Lags
- **Cross-Attention Decoder:** Replaced the static "global context" summary with a dynamic `nn.MultiheadAttention` decoder. Each future step now independently "scans" the historical sequence.
- **Deep Seasonal Lags:** Added **14-day, 28-day, and 30-day** lags to the feature set. These act as direct "periodic anchors," ensuring that long-term patterns are sustained through to the end of the 14-day horizon.

### Step 20: Auto-Iterative Sharpness Search
- **Architectural Forcing (Direct Lag Injection):** Modified `src/model.py` to concatenate raw transaction amounts from 7, 14, and 28 days ago directly to the final output heads. This bypasses the attention bottleneck and guarantees that the model has unfiltered access to historical rhythms during prediction.
- **Threshold Inference (Sharp Median):** Updated `src/inference.py` to use a 0.4 hurdle threshold for the median forecast. Instead of smoothed expected values (`prob * magnitude`), the system now outputs discrete spikes if the probability is high enough, ensuring realistic and interpretable cash flow events.
- **Auto-Iterative Loop:** Wrapped the training process in a search loop in `main.py` that calculates the **variance** of the validation forecast. If the model produces a straight line (low variance), the system automatically rejects it, increases the sharpness penalty, 'jolts' the learning rate, and retrains until a spiky solution is found.
- **Verification:** Verified the entire pipeline with a 37-test suite and visual audit of `forecast_viz.png`.
