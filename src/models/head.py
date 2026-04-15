from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseHead(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RegressionHead(BaseHead):
    """
    Anti-collapse regression head.

    Main changes:
    - remove LayerNorm by default
    - add a skip projection so the output can preserve variance from fused features
    - keep the MLP shallow
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layernorm: bool = False,
        use_skip: bool = True,
    ) -> None:
        super().__init__()

        if activation.lower() == "relu":
            act_layer = nn.ReLU
        elif activation.lower() == "gelu":
            act_layer = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.use_skip = use_skip

        if hidden_dim is None:
            self.head = nn.Linear(input_dim, 1)
            self.skip = None
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
            self.skip = nn.Linear(input_dim, 1) if use_skip else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(x)
        if self.skip is not None:
            out = out + self.skip(x)
        return out
