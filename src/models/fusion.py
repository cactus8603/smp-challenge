from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class BaseFusion(nn.Module, ABC):
    """
    Base class for multimodal fusion modules.

    Expected input:
        features = {
            "text": Tensor[B, D],
            "meta": Tensor[B, D],
            "image": Tensor[B, D],
            ...
        }

    Only non-None features should be fused.
    """

    @abstractmethod
    def forward(self, features: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        raise NotImplementedError


class ConcatFusion(BaseFusion):
    """
    Concatenate available modality features, then apply an MLP.

    Design goals:
    - Very stable baseline
    - Easy to expand with new modalities
    - Keeps caller code simple

    Example:
        fusion = ConcatFusion(
            input_dims={"text": 256, "meta": 256, "image": 256},
            hidden_dim=256,
            output_dim=256,
            dropout=0.1,
        )
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 256,
        output_dim: int = 256,
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

        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        total_input_dim = sum(input_dims.values())
        if total_input_dim <= 0:
            raise ValueError("ConcatFusion requires at least one positive input dimension.")

        layers: List[nn.Module] = [
            nn.Linear(total_input_dim, hidden_dim),
            act_layer(),
        ]

        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.extend(
            [
                nn.Linear(hidden_dim, output_dim),
                act_layer(),
            ]
        )

        if use_layernorm:
            layers.append(nn.LayerNorm(output_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.fusion = nn.Sequential(*layers)

    def forward(self, features: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            features:
                dict of modality_name -> tensor [B, D] or None

        Returns:
            fused tensor: [B, output_dim]
        """
        collected = []

        for name, dim in self.input_dims.items():
            feat = features.get(name, None)
            if feat is None:
                continue

            if feat.ndim != 2:
                raise ValueError(
                    f"Feature '{name}' must have shape [B, D], got {tuple(feat.shape)}"
                )

            if feat.size(1) != dim:
                raise ValueError(
                    f"Feature '{name}' dim mismatch: expected {dim}, got {feat.size(1)}"
                )

            collected.append(feat)

        if len(collected) == 0:
            raise ValueError("No valid features provided to fusion.")

        x = torch.cat(collected, dim=1)
        x = self.fusion(x)
        return x