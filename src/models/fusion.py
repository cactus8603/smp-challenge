from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseFusion(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        features: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        raise NotImplementedError


def _get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "gelu":
        return nn.GELU
    raise ValueError(f"Unsupported activation: {name}")


class ConcatFusion(BaseFusion):
    """
    Concatenate available modality features, then apply a light MLP plus residual skip.
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()

        act_layer = _get_activation(activation)

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
        self.skip = nn.Linear(total_input_dim, output_dim)

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
        return self.fusion(x) + self.skip(x)


class CrossFeatureFusion(BaseFusion):
    """
    Original multiplicative interaction fusion.
    Kept for comparison / ablation.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()

        act_layer = _get_activation(activation)

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

        fusion_input_dim = hidden_dim * 6 + 1

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
        self.base_proj = nn.Linear(hidden_dim, output_dim)

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

        base = (text_feat + meta_feat + image_feat) / 3.0
        fused = self.fusion(fused_input) + self.base_proj(base)
        return fused


class PairwiseGatedFusion(BaseFusion):
    """
    Score-first multimodal fusion:
    - keep a strong base concat path
    - add three pairwise interaction branches
    - use sample-wise gates to weight interaction branches

    Pairwise branches:
    - text-meta
    - text-image
    - meta-image

    Each branch uses:
    [a, b, a*b, |a-b|]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 256,
        pair_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()

        act_layer = _get_activation(activation)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pair_hidden_dim = pair_hidden_dim or hidden_dim

        # base concat path
        self.base_fusion = ConcatFusion(
            input_dims={"text": hidden_dim, "meta": hidden_dim, "image": hidden_dim},
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            use_layernorm=use_layernorm,
        )

        def build_pair_proj() -> nn.Sequential:
            layers: List[nn.Module] = [
                nn.Linear(hidden_dim * 4, self.pair_hidden_dim),
                act_layer(),
            ]
            if use_layernorm:
                layers.append(nn.LayerNorm(self.pair_hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.extend([
                nn.Linear(self.pair_hidden_dim, output_dim),
                act_layer(),
            ])
            if use_layernorm:
                layers.append(nn.LayerNorm(output_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        self.tm_proj = build_pair_proj()
        self.ti_proj = build_pair_proj()
        self.mi_proj = build_pair_proj()

        # sample-wise gate over three interaction branches
        gate_input_dim = output_dim * 4  # base + tm + ti + mi
        gate_hidden_dim = max(output_dim // 2, 64)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            act_layer(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(gate_hidden_dim, 3),
        )

        # final fusion after gated residual interaction aggregation
        final_layers: List[nn.Module] = [
            nn.Linear(output_dim * 4, hidden_dim),
            act_layer(),
        ]
        if use_layernorm:
            final_layers.append(nn.LayerNorm(hidden_dim))
        if dropout > 0:
            final_layers.append(nn.Dropout(dropout))
        final_layers.extend([
            nn.Linear(hidden_dim, output_dim),
            act_layer(),
        ])
        if use_layernorm:
            final_layers.append(nn.LayerNorm(output_dim))
        if dropout > 0:
            final_layers.append(nn.Dropout(dropout))

        self.final_fusion = nn.Sequential(*final_layers)
        self.final_skip = nn.Linear(output_dim * 4, output_dim)

    def _validate_feature(self, feat: Optional[torch.Tensor], name: str) -> torch.Tensor:
        if feat is None:
            raise ValueError(f"PairwiseGatedFusion requires '{name}' feature, but got None.")
        if feat.ndim != 2:
            raise ValueError(f"Feature '{name}' must have shape [B, D], got {tuple(feat.shape)}")
        if feat.size(1) != self.hidden_dim:
            raise ValueError(
                f"Feature '{name}' dim mismatch: expected {self.hidden_dim}, got {feat.size(1)}"
            )
        return feat

    def _build_pair_input(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b, a * b, torch.abs(a - b)], dim=-1)

    def forward(
        self,
        features: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        text_feat = self._validate_feature(features.get("text", None), "text")
        meta_feat = self._validate_feature(features.get("meta", None), "meta")
        image_feat = self._validate_feature(features.get("image", None), "image")

        base = self.base_fusion(
            {
                "text": text_feat,
                "meta": meta_feat,
                "image": image_feat,
            }
        )

        tm_repr = self.tm_proj(self._build_pair_input(text_feat, meta_feat))
        ti_repr = self.ti_proj(self._build_pair_input(text_feat, image_feat))
        mi_repr = self.mi_proj(self._build_pair_input(meta_feat, image_feat))

        gate_input = torch.cat([base, tm_repr, ti_repr, mi_repr], dim=-1)
        gate_logits = self.gate_mlp(gate_input)          # [B, 3]
        gate_weights = F.softmax(gate_logits, dim=-1)    # [B, 3]

        gated_tm = gate_weights[:, 0:1] * tm_repr
        gated_ti = gate_weights[:, 1:2] * ti_repr
        gated_mi = gate_weights[:, 2:3] * mi_repr

        final_input = torch.cat([base, gated_tm, gated_ti, gated_mi], dim=-1)
        fused = self.final_fusion(final_input) + self.final_skip(final_input)
        return fused