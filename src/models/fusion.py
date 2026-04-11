from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class BaseFusion(nn.Module, ABC):
    """
    Base class for multimodal fusion modules.
    """

    @abstractmethod
    def forward(
        self,
        features: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        raise NotImplementedError


class ConcatFusion(BaseFusion):
    """
    Concatenate available modality features, then apply an MLP.
    Kept as a simple baseline.
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

        layers.extend([
            nn.Linear(hidden_dim, output_dim),
            act_layer(),
        ])
        if use_layernorm:
            layers.append(nn.LayerNorm(output_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.input_dims = input_dims
        self.fusion = nn.Sequential(*layers)

    def forward(
        self,
        features: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        collected = []

        for name, dim in self.input_dims.items():
            feat = features.get(name, None)
            if feat is None:
                continue
            if feat.ndim != 2:
                raise ValueError(f"Feature '{name}' must have shape [B, D], got {tuple(feat.shape)}")
            if feat.size(1) != dim:
                raise ValueError(f"Feature '{name}' dim mismatch: expected {dim}, got {feat.size(1)}")
            collected.append(feat)

        if len(collected) == 0:
            raise ValueError("No valid features provided to fusion.")

        x = torch.cat(collected, dim=-1)
        return self.fusion(x)


class CrossFeatureFusion(BaseFusion):
    """
    Explicit text-metadata interaction fusion.

    Design:
        text_repr: [B, H]
        meta_repr: [B, H]
        cross_raw = text_repr * meta_repr                  # [B, H]
        cross_repr = cross_proj(cross_raw)                # [B, H]
        fused_input = concat(text_repr, meta_repr, cross_repr)  # [B, 3H]
        fused = fusion_mlp(fused_input)                   # [B, H]

    Notes:
    - This version is intended for the current text + metadata stage.
    - Image features are ignored for now even if present in `features`.
    """

    def __init__(
        self,
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

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        cross_layers: List[nn.Module] = [
            nn.Linear(hidden_dim, hidden_dim),
            act_layer(),
        ]
        if use_layernorm:
            cross_layers.append(nn.LayerNorm(hidden_dim))
        if dropout > 0:
            cross_layers.append(nn.Dropout(dropout))
        self.cross_proj = nn.Sequential(*cross_layers)

        fusion_layers: List[nn.Module] = [
            nn.Linear(hidden_dim * 3, hidden_dim),
            act_layer(),
        ]
        if use_layernorm:
            fusion_layers.append(nn.LayerNorm(hidden_dim))
        if dropout > 0:
            fusion_layers.append(nn.Dropout(dropout))

        fusion_layers.extend([
            nn.Linear(hidden_dim, output_dim),
            act_layer(),
        ])
        if use_layernorm:
            fusion_layers.append(nn.LayerNorm(output_dim))
        if dropout > 0:
            fusion_layers.append(nn.Dropout(dropout))

        self.fusion = nn.Sequential(*fusion_layers)

    def forward(
        self,
        features: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        text_feat = features.get("text", None)
        meta_feat = features.get("meta", None)

        if text_feat is None or meta_feat is None:
            raise ValueError("CrossFeatureFusion requires both 'text' and 'meta' features.")

        if text_feat.ndim != 2 or meta_feat.ndim != 2:
            raise ValueError(
                f"text/meta features must be [B, D], got text={tuple(text_feat.shape)}, meta={tuple(meta_feat.shape)}"
            )

        if text_feat.size(1) != self.hidden_dim:
            raise ValueError(
                f"text feature dim mismatch: expected {self.hidden_dim}, got {text_feat.size(1)}"
            )
        if meta_feat.size(1) != self.hidden_dim:
            raise ValueError(
                f"meta feature dim mismatch: expected {self.hidden_dim}, got {meta_feat.size(1)}"
            )

        cross_raw = text_feat * meta_feat
        cross_repr = self.cross_proj(cross_raw)

        fused_input = torch.cat([text_feat, meta_feat, cross_repr], dim=-1)
        fused = self.fusion(fused_input)
        return fused
