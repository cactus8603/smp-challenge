from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn


class BaseHead(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class RegressionHead(BaseHead):
    """
    Stronger regression head for SMP.

    Main updates:
    - supports multiple hidden layers
    - keeps residual skip projection to preserve variance from fused features
    - stays lightweight enough for stable training
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layernorm: bool = False,
        use_skip: bool = True,
    ) -> None:
        super().__init__()

        self.use_skip = use_skip

        if hidden_dims is None:
            if hidden_dim is None:
                hidden_dims = []
            else:
                # default: stronger than before, but still compact
                hidden_dims = [hidden_dim, hidden_dim]

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(_get_activation(activation))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.head = nn.Sequential(*layers)

        self.skip = nn.Linear(input_dim, 1) if use_skip else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(x)
        if self.skip is not None:
            out = out + self.skip(x)
        return out