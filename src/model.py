"""Module for the core Deep Learning model architecture."""

from typing import List

import torch
import torch.nn as nn


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) as described in the TFT paper.

    Provides a flexible non-linear mapping with gating and residual connections.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: int = None,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.context_dim = context_dim

        self.lin1 = nn.Linear(input_dim, hidden_dim)
        if context_dim:
            self.context_lin = nn.Linear(context_dim, hidden_dim, bias=False)

        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.project = nn.Linear(hidden_dim, output_dim)
        
        # Residual connection
        if input_dim != output_dim:
            self.res_project = nn.Linear(input_dim, output_dim)
        else:
            self.res_project = nn.Identity()
            
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for GRN.

        Args:
            x: Input tensor.
            context: Optional static context.

        Returns:
            Gated output.
        """
        h = self.lin1(x)
        if context is not None and self.context_dim:
            h = h + self.context_lin(context)
        
        h = torch.relu(h)
        h = self.lin2(h)
        
        # Gating Unit
        g = self.gate(h)
        h = self.project(h)
        
        # Residual + Gated output
        out = self.layernorm(self.res_project(x) + self.dropout(g * h))
        return out


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN) to dynamically weigh feature importance."""

    def __init__(
        self,
        input_dim: int,
        num_vars: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: int = None,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim

        # GRN for each variable
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout, context_dim)
            for _ in range(num_vars)
        ])

        # GRN for variable selection weights
        self.weight_grn = GatedResidualNetwork(
            input_dim * num_vars, hidden_dim, num_vars, dropout, context_dim
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for VSN.

        Args:
            x: Input tensor (Batch, Time, num_vars * input_dim).
            context: Optional context.

        Returns:
            Weighted combination of transformed variables.
        """
        # x is (B, T, D_total) where D_total = num_vars * d_var
        # We split it back to (B, T, num_vars, d_var)
        batch, time, d_total = x.shape
        d_var = d_total // self.num_vars
        x_split = x.view(batch, time, self.num_vars, d_var)

        # 1. Transform each variable
        var_outputs = []
        for i in range(self.num_vars):
            var_outputs.append(self.var_grns[i](x_split[:, :, i, :], context))
        var_outputs = torch.stack(var_outputs, dim=-2) # (B, T, num_vars, hidden_dim)

        # 2. Calculate selection weights
        weights = self.weight_grn(x, context) # (B, T, num_vars)
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1) # (B, T, num_vars, 1)

        # 3. Weighted sum
        out = torch.sum(weights * var_outputs, dim=-2)
        return out


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
    """Two-Stage Hurdle Model for sparse cash flow forecasting.

    Stage 1: Classification (Will a transaction occur?)
    Stage 2: Quantile Regression (What is the magnitude if it occurs?)
    """

    def __init__(
        self,
        input_dim: int,
        num_accounts: int,
        hidden_dim: int = 128,
        history_size: int = 60,
        horizon: int = 14,
        num_quantiles: int = 3,
        num_heads: int = 4,
        num_layers: int = 5,
        future_dim: int = 6,
    ) -> None:
        super().__init__()
        self.history_size = history_size
        self.horizon = horizon
        self.num_quantiles = num_quantiles
        self.hidden_dim = hidden_dim

        # 1. Variable Selection Network (Input Scaling)
        self.input_vsn = VariableSelectionNetwork(
            input_dim=1,
            num_vars=input_dim,
            hidden_dim=hidden_dim,
            dropout=0.1
        )

        # 2. Entity Embeddings for Account Personalization
        self.account_embedding = nn.Embedding(num_accounts, hidden_dim)

        # 3. Shared Backbone
        self.pos_embedding = nn.Parameter(torch.randn(1, history_size, hidden_dim))
        self.horizon_pos_embedding = nn.Parameter(torch.randn(1, horizon, hidden_dim))
        
        self.encoder = TemporalEncoder(hidden_dim, hidden_dim, num_layers=num_layers)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=0.1
        )

        # 4. Future Exogenous Projection
        self.future_projection = nn.Linear(future_dim, hidden_dim)

        # 5. Gated Heads
        self.gate_grn = GatedResidualNetwork(hidden_dim, hidden_dim, 1)
        self.magnitude_grn = GatedResidualNetwork(hidden_dim, hidden_dim, num_quantiles)

    def forward(
        self, 
        x: torch.Tensor, 
        future_x: torch.Tensor, 
        account_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with personalization and future guidance.

        Args:
            x: History features (Batch, Time, F).
            future_x: Future exogenous features (Batch, Horizon, FutureDim).
            account_idx: Account indices (Batch,).

        Returns:
            Tuple of (prob_gate, magnitudes).
        """
        batch_size = x.size(0)

        # 1. Variable Selection & Account Personalization
        x_vsn = self.input_vsn(x)
        acc_embed = self.account_embedding(account_idx).unsqueeze(1) # (B, 1, Hidden)
        
        # 2. Backbone
        # Inject account context into history
        x_enc = x_vsn + self.pos_embedding + acc_embed
        encoded = self.encoder(x_enc).transpose(1, 2)
        
        attn_out, _ = self.attention(encoded, encoded, encoded)
        
        # Global Context (Max + Mean)
        ctx_mean = attn_out.mean(dim=1)
        ctx_max, _ = attn_out.max(dim=1)
        context = ctx_mean + ctx_max # (B, Hidden)

        # 3. Decoder Expansion with Future Guidance
        # Static Context (B, 1, Hidden)
        h_static = (context + acc_embed.squeeze(1)).unsqueeze(1).repeat(1, self.horizon, 1)
        
        # Dynamic Future Guidance (B, H, Hidden)
        h_future = self.future_projection(future_x)
        
        # Combine Step-specific Positional Embedding
        h_context = h_static + h_future + self.horizon_pos_embedding

        # 4. Gated Heads
        prob_gate = torch.sigmoid(self.gate_grn(h_context)).squeeze(-1)
        magnitudes = self.magnitude_grn(h_context)

        return prob_gate, magnitudes
