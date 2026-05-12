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

- [ ] **Step 1: Project Scaffolding & Docker Setup.** Create directory structure (`src/`, `data/`, `tests/`), `requirements.txt`, and a multi-stage `Dockerfile`.
- [ ] **Step 2: Data Ingestion Pipeline.** Write a modular class to download, clean, and parse PKDD'99/COFINFAD CSV files.
- [ ] **Step 3: Feature Engineering (Temporal).** Implement Cyclical Time Encodings (Sine/Cosine transforms for day/month) and zero-inflation handling mechanisms.
- [ ] **Step 4: 3D Tensor Transformation.** Build the pipeline to pivot the flat data into the `(N, Time, Features)` PyTorch tensor format.
- [ ] **Step 5: PyTorch Dataset & DataLoader.** Implement custom `Dataset` class supporting sliding window generation and variable-length sequence padding if necessary.
- [ ] **Step 6: Core Model Architecture.** Implement the base Sequence-to-Sequence model (combining concepts from TFT/N-BEATS) using `torch.nn`. Include Multi-Head Attention or Dilated Convolutions.
- [ ] **Step 7: Loss Function Implementation.** Code the Quantile (Pinball) Loss function as a custom `nn.Module` to predict 10th, 50th, and 90th percentiles.
- [ ] **Step 8: Training Loop.** Create the training and validation loops, incorporating gradient clipping, learning rate scheduling (Cosine Annealing), and Early Stopping.
- [ ] **Step 9: Conformal Calibration Module.** Implement Split Conformal Prediction. Compute non-conformity scores on a hold-out set to guarantee distribution-free coverage.
- [ ] **Step 10: Evaluation & Metrics.** Implement PICP (Prediction Interval Coverage Probability) and MPIW (Mean Prediction Interval Width) calculators.
- [ ] **Step 11: Inference API.** Wrap the trained, calibrated model into a clean inference class capable of taking a raw transaction list and outputting calibrated JSON predictions.

## Implementation Log

Agent Instructions: After completing a step, append a rigorous technical log entry below explaining your architectural decisions.
