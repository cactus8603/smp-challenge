from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseHead(nn.Module, ABC):
    """
    Base prediction head.

    Input:
        fused feature tensor [B, D]

    Output:
        task-specific prediction
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RegressionHead(BaseHead):
    """
    Regression head for popularity prediction.

    Design goals:
    - Simple and stable baseline
    - Easy to deepen later
    - Output shape: [B, 1]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        if activation.lower() == "relu":
            act_layer = nn.ReLU
        elif activation.lower() == "gelu":
            act_layer = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if hidden_dim is None:
            # single linear layer head
            self.head = nn.Linear(input_dim, 1)
        else:
            layers = [
                nn.Linear(input_dim, hidden_dim),
                act_layer(),
            ]

            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_dim, 1))
            self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D]

        Returns:
            prediction: [B, 1]
        """
        return self.head(x)