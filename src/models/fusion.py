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
        extra_dim: int = 0,
    ) -> None:
        super().__init__()

        act_layer = _get_activation(activation)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.extra_dim = int(extra_dim)

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

        fusion_input_dim = hidden_dim * 6 + 1 + self.extra_dim

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

        fused_parts = [
            text_feat,
            meta_feat,
            image_feat,
            tm_repr,
            it_repr,
            im_repr,
            sim_it,
        ]
        if self.extra_dim > 0:
            extra_feat = features.get("clip_sim", None)
            if extra_feat is None:
                raise ValueError("CrossFeatureFusion expected 'clip_sim' but got None.")
            if extra_feat.ndim != 2 or extra_feat.size(1) != self.extra_dim:
                raise ValueError(
                    f"clip_sim dim mismatch: expected [B, {self.extra_dim}], got {tuple(extra_feat.shape)}"
                )
            fused_parts.append(extra_feat)

        fused_input = torch.cat(fused_parts, dim=-1)

        base = (text_feat + meta_feat + image_feat) / 3.0
        fused = self.fusion(fused_input) + self.base_proj(base)
        return fused


class PairwiseGatedFusion(BaseFusion):
    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 256,
        pair_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layernorm: bool = False,
        extra_dim: int = 0,
    ) -> None:
        super().__init__()

        act_layer = _get_activation(activation)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pair_hidden_dim = pair_hidden_dim or hidden_dim
        self.extra_dim = extra_dim

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

        gate_input_dim = output_dim * 4
        gate_hidden_dim = max(output_dim // 2, 64)

        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            act_layer(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(gate_hidden_dim, 3),
        )

        final_input_dim = output_dim * 4 + extra_dim

        final_layers: List[nn.Module] = [
            nn.Linear(final_input_dim, hidden_dim),
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
        self.final_skip = nn.Linear(final_input_dim, output_dim)

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
        gate_logits = self.gate_mlp(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)

        gated_tm = gate_weights[:, 0:1] * tm_repr
        gated_ti = gate_weights[:, 1:2] * ti_repr
        gated_mi = gate_weights[:, 2:3] * mi_repr

        final_parts = [base, gated_tm, gated_ti, gated_mi]

        if self.extra_dim > 0:
            extra_feat = features.get("clip_sim", None)
            if extra_feat is None:
                raise ValueError("PairwiseGatedFusion expected 'clip_sim' but got None.")
            if extra_feat.ndim != 2 or extra_feat.size(1) != self.extra_dim:
                raise ValueError(
                    f"clip_sim dim mismatch: expected [B, {self.extra_dim}], got {tuple(extra_feat.shape)}"
                )
            final_parts.append(extra_feat)

        final_input = torch.cat(final_parts, dim=-1)
        fused = self.final_fusion(final_input) + self.final_skip(final_input)
        return fused

class ResidualCrossAttentionFusion(BaseFusion):
    """
    Non-MoE residual cross-attention fusion for text/meta/image.

    Design goals:
    - Fuse at the representation level, not at the prediction/expert level.
    - Default: text + metadata interact first; image is added after that.
    - Optional: image + metadata can interact first for ablation.
    - No softmax competition between modalities.
    - Use concat pooling instead of mean pooling so metadata is not averaged away.
    - Preserve each modality through explicit learnable residual scales.

    Inputs expected in features:
        text: [B, H]
        meta: [B, H]
        image: [B, H]
        clip_sim: optional [B, extra_dim]
        user_desc: optional [B, H], when use_user_desc=True
    Output:
        fused: [B, output_dim]
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = True,
        extra_dim: int = 0,
        meta_res_scale: float = 1.20,
        text_res_scale: float = 0.50,
        image_res_scale: float = 0.50,
        tm_res_scale: float = 0.75,
        use_user_desc: bool = False,
        user_res_scale: float = 0.35,
        user_gate_init: float = -2.0,
        max_res_scale: float = 3.0,
        order: str = "text_meta_first",
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}."
            )

        act_layer = _get_activation(activation)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.extra_dim = int(extra_dim)
        self.use_user_desc = bool(use_user_desc)
        self.max_res_scale = float(max_res_scale)
        self.order = str(order or "text_meta_first").lower()
        if self.order not in {"text_meta_first", "image_meta_first"}:
            raise ValueError(f"Unsupported ResidualCrossAttentionFusion order: {order}")

        self.tm_type_embed = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        self.tmi_type_embed = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        nn.init.normal_(self.tm_type_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.tmi_type_embed, mean=0.0, std=0.02)

        self.tm_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.tmi_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.tm_norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.tmi_norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

        self.user_type_embed: Optional[nn.Parameter] = None
        self.user_attn: Optional[nn.MultiheadAttention] = None
        self.user_norm: Optional[nn.Module] = None
        self.user_res: Optional[nn.Module] = None
        self.user_gate: Optional[nn.Module] = None

        self.tm_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, output_dim),  # text_ctx, meta_ctx, raw text, raw meta
            nn.LayerNorm(output_dim) if use_layernorm else nn.Identity(),
            act_layer(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim) if use_layernorm else nn.Identity(),
        )

        self.text_res = nn.Linear(hidden_dim, output_dim)
        self.meta_res = nn.Linear(hidden_dim, output_dim)
        self.image_res = nn.Linear(hidden_dim, output_dim)
        self.tm_res = nn.Linear(output_dim, output_dim)

        # Independent learnable scales, intentionally NOT softmax-normalized.
        # Multiple modalities can be strong at the same time.
        self.meta_res_scale = nn.Parameter(torch.tensor(float(meta_res_scale)))
        self.text_res_scale = nn.Parameter(torch.tensor(float(text_res_scale)))
        self.image_res_scale = nn.Parameter(torch.tensor(float(image_res_scale)))
        self.tm_res_scale = nn.Parameter(torch.tensor(float(tm_res_scale)))
        self.user_res_scale: Optional[nn.Parameter] = None

        if self.use_user_desc:
            self.user_type_embed = nn.Parameter(torch.zeros(1, 2, hidden_dim))
            nn.init.normal_(self.user_type_embed, mean=0.0, std=0.02)
            self.user_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.user_norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
            self.user_res = nn.Linear(hidden_dim, output_dim)
            self.user_res_scale = nn.Parameter(torch.tensor(float(user_res_scale)))

            gate_hidden_dim = max(hidden_dim // 4, 32)
            self.user_gate = nn.Sequential(
                nn.Linear(output_dim + hidden_dim + 1, gate_hidden_dim),
                act_layer(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(gate_hidden_dim, 1),
            )
            nn.init.zeros_(self.user_gate[-1].weight)
            nn.init.constant_(self.user_gate[-1].bias, float(user_gate_init))

        final_input_dim = output_dim * 6 + self.extra_dim
        # concat = tm_ctx, image_ctx, tm_repr, text_res, meta_res, image_res, optional sim
        self.final_proj = nn.Sequential(
            nn.Linear(final_input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2) if use_layernorm else nn.Identity(),
            act_layer(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim) if use_layernorm else nn.Identity(),
            act_layer(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim) if use_layernorm else nn.Identity(),
        )
        self.final_skip = nn.Linear(final_input_dim, output_dim)

    def _validate_feature(self, feat: Optional[torch.Tensor], name: str) -> torch.Tensor:
        if feat is None:
            raise ValueError(f"ResidualCrossAttentionFusion requires '{name}' feature, but got None.")
        if feat.ndim != 2:
            raise ValueError(f"Feature '{name}' must have shape [B, D], got {tuple(feat.shape)}")
        if feat.size(1) != self.hidden_dim:
            raise ValueError(
                f"Feature '{name}' dim mismatch: expected {self.hidden_dim}, got {feat.size(1)}"
            )
        return feat

    def _scale(self, p: torch.Tensor) -> torch.Tensor:
        return p.clamp(0.0, self.max_res_scale)

    def _attn_block(
        self,
        attn: nn.MultiheadAttention,
        norm: nn.Module,
        tokens: torch.Tensor,
        type_embed: torch.Tensor,
    ) -> torch.Tensor:
        x = tokens + type_embed[:, : tokens.size(1), :]
        attn_out, _ = attn(x, x, x, need_weights=False)
        return norm(x + attn_out)

    def forward(self, features: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        text_feat = self._validate_feature(features.get("text", None), "text")
        meta_feat = self._validate_feature(features.get("meta", None), "meta")
        image_feat = self._validate_feature(features.get("image", None), "image")

        # Stage 1: one strong modality interacts with metadata first.
        # Default is text_meta_first because text/meta have been the strongest
        # signals in our ablations. image_meta_first is available for ablation
        # if image drop becomes larger than meta drop.
        if self.order == "text_meta_first":
            early_a = text_feat
            early_b = meta_feat
            late_feat = image_feat
        else:  # image_meta_first
            early_a = image_feat
            early_b = meta_feat
            late_feat = text_feat

        early_tokens = torch.stack([early_a, early_b], dim=1)  # [B, 2, H]
        early_tokens = self._attn_block(self.tm_attn, self.tm_norm, early_tokens, self.tm_type_embed)
        early_a_ctx = early_tokens[:, 0, :]
        meta_ctx = early_tokens[:, 1, :]

        # Concat pooling avoids averaging metadata away.
        early_repr = self.tm_proj(torch.cat([early_a_ctx, meta_ctx, early_a, meta_feat], dim=-1))

        # Stage 2: early representation interacts with the remaining modality.
        late_tokens = torch.stack([early_repr, late_feat], dim=1)  # [B, 2, H]
        late_tokens = self._attn_block(self.tmi_attn, self.tmi_norm, late_tokens, self.tmi_type_embed)
        tm_ctx = late_tokens[:, 0, :]
        image_ctx = late_tokens[:, 1, :]

        text_r = self._scale(self.text_res_scale) * self.text_res(text_feat)
        meta_r = self._scale(self.meta_res_scale) * self.meta_res(meta_feat)
        image_r = self._scale(self.image_res_scale) * self.image_res(image_feat)
        tm_r = self._scale(self.tm_res_scale) * self.tm_res(early_repr)

        final_parts = [tm_ctx, image_ctx, tm_r, text_r, meta_r, image_r]

        if self.extra_dim > 0:
            extra_feat = features.get("clip_sim", None)
            if extra_feat is None:
                raise ValueError("ResidualCrossAttentionFusion expected 'clip_sim' but got None.")
            if extra_feat.ndim != 2 or extra_feat.size(1) != self.extra_dim:
                raise ValueError(
                    f"clip_sim dim mismatch: expected [B, {self.extra_dim}], got {tuple(extra_feat.shape)}"
                )
            final_parts.append(extra_feat)

        final_input = torch.cat(final_parts, dim=-1)
        fused = self.final_proj(final_input) + self.final_skip(final_input)

        if not self.use_user_desc:
            return fused

        user_feat = features.get("user_desc", None)
        if user_feat is None:
            return fused
        user_feat = self._validate_feature(user_feat, "user_desc")
        user_mask = features.get("user_desc_mask", None)
        if user_mask is None:
            user_mask = torch.ones(user_feat.size(0), 1, device=user_feat.device, dtype=user_feat.dtype)
        else:
            user_mask = user_mask.to(device=user_feat.device, dtype=user_feat.dtype)

        if (
            self.user_attn is None
            or self.user_norm is None
            or self.user_type_embed is None
            or self.user_res is None
            or self.user_gate is None
            or self.user_res_scale is None
        ):
            raise RuntimeError("ResidualCrossAttentionFusion user_desc modules are not initialized.")

        user_tokens = torch.stack([fused, user_feat], dim=1)
        user_tokens = self._attn_block(
            self.user_attn,
            self.user_norm,
            user_tokens,
            self.user_type_embed,
        )
        user_ctx = user_tokens[:, 1, :]
        user_gate = torch.sigmoid(
            self.user_gate(torch.cat([fused, user_feat, user_mask], dim=-1))
        ) * user_mask
        features["user_desc_fusion_gate"] = user_gate.detach()

        return fused + user_gate * self._scale(self.user_res_scale) * self.user_res(user_ctx)
