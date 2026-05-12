"""Module for the core Deep Learning model architecture."""

from typing import List

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """Encoder using Dilated Convolutions to capture multi-scale temporal patterns.

    Dilated convolutions allow for a large receptive field without the
    computational cost or vanishing gradient issues of very deep RNNs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        num_layers: int = 3,
    ) -> None:
        """Initializes the encoder.

        Args:
            input_dim: Number of input features.
            hidden_dim: Number of hidden channels.
            kernel_size: Size of the convolutional kernel.
            num_layers: Number of dilated convolutional layers.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation // 2
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=input_dim if i == 0 else hidden_dim,
                        out_channels=hidden_dim,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (Batch, Time, Features).

        Returns:
            Encoded sequence of shape (Batch, Hidden, Time).
        """
        # Conv1d expects (Batch, Channels, Length)
        x = x.transpose(1, 2)
        for layer in self.layers:
            # Residual connection could be added here for deeper networks
            x = layer(x)
        return x


class ProbabilisticForecaster(nn.Module):
    """Sequence-to-sequence model for multi-quantile cash flow forecasting.

    Combines a Dilated CNN encoder with Multi-Head Attention and a
    quantile-specific output head.

    Attributes:
        history_size: Look-back window size.
        horizon: Prediction horizon size.
        num_quantiles: Number of quantiles to predict (e.g., 3 for 10/50/90).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        history_size: int = 60,
        horizon: int = 14,
        num_quantiles: int = 3,
        num_heads: int = 4,
    ) -> None:
        """Initializes the forecaster.

        Args:
            input_dim: Number of input features per time step.
            hidden_dim: Dimension of hidden representations.
            history_size: Number of look-back steps.
            horizon: Number of forecast steps.
            num_quantiles: Number of target quantiles.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.history_size = history_size
        self.horizon = horizon
        self.num_quantiles = num_quantiles

        # 1. Encoder: Extract temporal features
        self.encoder = TemporalEncoder(input_dim, hidden_dim)

        # 2. Attention: Weight historical contexts
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # 3. Decoder Head: Map context to horizon quantiles
        # Input to decoder is (Batch, Hidden_Context)
        # Output is (Batch, Horizon * Num_Quantiles)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, horizon * num_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass generating multi-quantile forecasts.

        Args:
            x: Input tensor of shape (Batch, History, Features).

        Returns:
            Forecast tensor of shape (Batch, Horizon, Num_Quantiles).
        """
        batch_size = x.size(0)

        # Encode: (Batch, Hidden, History)
        encoded = self.encoder(x)
        
        # Attention expects (Batch, Seq, Dim)
        encoded = encoded.transpose(1, 2)
        
        # Self-attention to find global temporal dependencies
        attn_out, _ = self.attention(encoded, encoded, encoded)
        
        # Pooling: Use the last context or average
        # Here we take the mean across the history dimension
        context = torch.mean(attn_out, dim=1)

        # Decode: (Batch, Horizon * Num_Quantiles)
        out = self.decoder(context)
        
        # Reshape to (Batch, Horizon, Num_Quantiles)
        return out.view(batch_size, self.horizon, self.num_quantiles)
