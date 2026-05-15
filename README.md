# Probabilistic Cash Flow Agent

**Layer 6 AI Engineering Blueprint | Trustworthy Time-Series Forecasting**

## System Overview

In enterprise banking, point-estimate predictions for cash flow are insufficient. Treasury and liquidity management require rigorous uncertainty quantification to balance capital efficiency with risk. The **Probabilistic Cash Flow Agent** is an advanced forecasting pipeline designed to predict future account balances and transaction volumes while providing statistically rigorous confidence intervals.

Built on state-of-the-art (SOTA 2026) non-LLM Deep Learning architectures—specifically drawing inspiration from Temporal Fusion Transformers (TFT) and DeepAR—this system is engineered to handle the complexities of financial time series: zero-inflation, high volatility, and multi-horizon dependencies. It embodies TD Layer 6's core values of **Trustworthy AI, Scalable Reliability**, and **Clean Software Design**.

## Mathematical Foundation & Architectural Strategy

**1. The Hurdle Model: Two-Stage Sparse Forecasting**

Financial transaction data is highly sparse (zero-inflated). To address this, we implement a **Two-Stage Hurdle Model**. The model consists of two specialized heads:
- **Classification Gate:** Predicts the probability $P(y_t > 0)$ using **Focal Loss** to handle class imbalance.
- **Quantile Head:** Predicts conditional magnitudes using **Magnitude-Weighted Quantile Loss** ($L = \text{Pinball} \times (|y| + \epsilon)$).

The final point forecast is a "Sharp Median" derived via **Threshold Inference**: 

$$
\hat{y}_{median} = \mathbb{I}(P(trans) > \gamma) \cdot \hat{y}_{\tau=0.5}
$$

where $\gamma$ is a trained hurdle threshold (e.g., 0.4), preventing the mathematical "blurring" of discrete cash flow events.

**2. Sharpness Incentives: Sequence-Level Losses**

Standard point-wise losses often result in "flat" mean-regressed forecasts. We implement a three-pronged **Sequence Sharpness Loss** to force the model to capture the rhythm and volume of the cash flow:
- **Volume Loss:** 
  
$$
\mathcal{L}_{vol} = |\sum \hat{y} - \sum y|
$$

enforcing conservation of mass over the 14-day horizon.

- **Variance Penalty:** 

$$
\mathcal{L}_{var} = \text{ReLU}(\text{Var}(y) - \text{Var}(\hat{y}))
$$

rewarding the model for producing realistic "spiky" variance.

- **Gaussian-Smoothed Shape Loss:** We apply a **1D Gaussian Kernel** smoothing to both sequences before calculating the Pearson Correlation. This creates a "temporal tolerance radius," rewarding the model for being *close* in timing, which provides smoother gradients for the attention mechanism.

**3. Architectural Forcing: Direct Lag Injection**

To prevent "Signal Washout" in deep attention layers, we implement **Direct Lag Injection**. Raw transaction amounts from significant periodic offsets (7, 14, and 28 days ago) are extracted from history and concatenated directly to the final output heads. This guarantees that the model has unfiltered access to historical rhythms right before prediction, bypassing the attention bottleneck.

**4. Uncertainty Quantification: Conformal Calibration**

Neural networks are notoriously overconfident. We apply **Split Conformal Prediction** on top of the model's heuristic intervals to provide distribution-free, mathematically proven coverage guarantees:

1. **Non-conformity Scores:** Calculate the heuristic error on the calibration set: $E_i = \max(\hat{y}_{\text{low}} - y_i, y_i - \hat{y}_{\text{high}})$ .
2. **Empirical Quantile:** Find $\hat{q}$, the $(1-\alpha)$ quantile of these scores.
3. **Guaranteed Bounds:** Expand heuristic predictions to achieve targeted coverage: $[\hat{y}_{\text{low}} - \hat{q}, \hat{y}_{\text{high}} + \hat{q}]$ .

This ensures that the true cash flow falls within our predicted bounds exactly $1 - \alpha$% of the time (e.g., 90% coverage).

**5. Scaling: Per-Account Local Normalization**

To generalize across wealth tiers, we implement **Local Scaling**. Each account is normalized by its own historical mean and standard deviation. This allows the model to learn universal temporal patterns (e.g., "a 3x salary spike") independent of whether the absolute dollar amount is $1,000 or $100,000.

## Tech Stack

- **Deep Learning:** `PyTorch` (Multi-Head Attention, Dilated CNN, GRN/VSN).
- **Optimization:** Cosine Annealing LR, Early Stopping, Magnitude-Weighted Loss.
- **Data Engineering:** `NumPy`, `Pandas` (Vectorized FFT spectral analysis, Cyclical encoding).
- **Deployment:** `Docker` (GPU passthrough support with NVIDIA Container Toolkit).
