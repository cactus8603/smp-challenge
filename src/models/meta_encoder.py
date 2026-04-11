from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


def build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    dropout: float = 0.1,
    activation: str = "relu",
    use_layernorm: bool = True,
) -> nn.Sequential:
    """
    Build a simple configurable MLP block.

    Example:
        input_dim=16, hidden_dims=[128, 256]
        -> Linear(16,128) -> Act -> LN -> Dropout
        -> Linear(128,256) -> Act -> LN -> Dropout
    """
    if activation.lower() == "relu":
        act_layer = nn.ReLU
    elif activation.lower() == "gelu":
        act_layer = nn.GELU
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(act_layer())
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim

    return nn.Sequential(*layers)


class MetaEncoder(nn.Module):
    """
    Metadata encoder for numeric / engineered metadata features.

    Design goals:
    - Accept a dense metadata vector from dataset.py
    - Produce a fixed-size representation for fusion
    - Easy to deepen / widen later without changing caller code
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, output_dim]

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one dimension.")

        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]

        self.encoder = build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            use_layernorm=use_layernorm,
        )

        # Optional projection to guarantee exact output_dim
        self.proj = None
        if self.output_dim != output_dim:
            self.proj = nn.Linear(self.output_dim, output_dim)
            self.output_dim = output_dim

    def forward(self, meta_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            meta_features: [B, D]

        Returns:
            encoded metadata feature: [B, output_dim]
        """
        x = self.encoder(meta_features)

        if self.proj is not None:
            x = self.proj(x)

        return x