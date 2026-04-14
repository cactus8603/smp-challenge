from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Explicit multimodal interaction fusion for text + metadata + image.

    Design:
        text_repr:  [B, H]
        meta_repr:  [B, H]
        image_repr: [B, H]

        tm_raw = text_repr * meta_repr
        it_raw = image_repr * text_repr
        im_raw = image_repr * meta_repr

        tm_repr = tm_proj(tm_raw)
        it_repr = it_proj(it_raw)
        im_repr = im_proj(im_raw)

        sim_it = cosine_similarity(image_repr, text_repr)   # [B, 1]

        fused_input = concat(
            text_repr,
            meta_repr,
            image_repr,
            tm_repr,
            it_repr,
            im_repr,
            sim_it
        )  # [B, 6H + 1]

        fused = fusion_mlp(fused_input)  # [B, output_dim]

    Notes:
    - This version expects all three modalities to be present.
    - If you want optional image support later, we can extend this to a more flexible variant.
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

        def build_interaction_proj() -> nn.Sequential:
            layers: List[nn.Module] = [
                nn.Linear(hidden_dim, hidden_dim),
                act_layer(),
            ]
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        self.tm_proj = build_interaction_proj()
        self.it_proj = build_interaction_proj()
        self.im_proj = build_interaction_proj()

        fusion_input_dim = hidden_dim * 6 + 1  # text, meta, image, tm, it, im, sim_it

        fusion_layers: List[nn.Module] = [
            nn.Linear(fusion_input_dim, hidden_dim),
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

    def _validate_feature(self, feat: Optional[torch.Tensor], name: str) -> torch.Tensor:
        if feat is None:
            raise ValueError(f"CrossFeatureFusion requires '{name}' feature, but got None.")
        if feat.ndim != 2:
            raise ValueError(f"Feature '{name}' must have shape [B, D], got {tuple(feat.shape)}")
        if feat.size(1) != self.hidden_dim:
            raise ValueError(
                f"Feature '{name}' dim mismatch: expected {self.hidden_dim}, got {feat.size(1)}"
            )
        return feat

    def forward(
        self,
        features: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        text_feat = self._validate_feature(features.get("text", None), "text")
        meta_feat = self._validate_feature(features.get("meta", None), "meta")
        image_feat = self._validate_feature(features.get("image", None), "image")

        tm_raw = text_feat * meta_feat
        it_raw = image_feat * text_feat
        im_raw = image_feat * meta_feat

        tm_repr = self.tm_proj(tm_raw)
        it_repr = self.it_proj(it_raw)
        im_repr = self.im_proj(im_raw)

        sim_it = F.cosine_similarity(image_feat, text_feat, dim=-1).unsqueeze(-1)

        fused_input = torch.cat(
            [
                text_feat,
                meta_feat,
                image_feat,
                tm_repr,
                it_repr,
                im_repr,
                sim_it,
            ],
            dim=-1,
        )

        fused = self.fusion(fused_input)
        return fused