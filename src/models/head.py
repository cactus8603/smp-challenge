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


class MetaWeightedMoEHead(BaseHead):
    """
    LEGACY / experimental head.

    This class is intentionally kept for ablation only, but fusion_model.py
    does NOT use it in the current image_v2_sim_v1-style version.

    Previous behavior:
      - 5 experts: full / meta_A / meta_B / text / image
      - gate also received runtime availability flags

    It tended to make the model routing harder to interpret after adding
    user_desc auxiliary features, so the active head is ModalityAwareMoEHead.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_skip: bool = True,
        n_modality_flags: int = 3,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.use_skip = use_skip
        self.n_experts = 5

        def act():
            return _get_activation(activation)

        def proj(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim), act(),
                nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
            )

        self.full_proj   = proj(input_dim)
        self.meta_A_proj = proj(input_dim * 2)
        self.meta_B_proj = proj(input_dim)
        self.text_proj   = proj(input_dim * 2)
        self.image_proj  = proj(input_dim * 2)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), act(),
                nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
            )
            for _ in range(self.n_experts)
        ])

        gate_input_dim = hidden_dim * self.n_experts + n_modality_flags
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, max(hidden_dim // 2, 64)), act(),
            nn.Dropout(dropout), nn.Linear(max(hidden_dim // 2, 64), self.n_experts),
        )
        self.skip = nn.Linear(input_dim, 1) if use_skip else None

    def forward(
        self,
        fused: torch.Tensor,
        text_feat: torch.Tensor,
        meta_feat: torch.Tensor,
        image_feat: torch.Tensor,
        modality_flags: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_full   = self.full_proj(fused)
        h_meta_A = self.meta_A_proj(torch.cat([fused, meta_feat], dim=-1))
        h_meta_B = self.meta_B_proj(meta_feat)
        h_text   = self.text_proj(torch.cat([fused, text_feat], dim=-1))
        h_image  = self.image_proj(torch.cat([fused, image_feat], dim=-1))

        expert_hiddens = [h_full, h_meta_A, h_meta_B, h_text, h_image]
        expert_outputs = torch.stack(
            [expert(h) for expert, h in zip(self.experts, expert_hiddens)], dim=1
        )  # [B, 5, 1]

        gate_parts = [torch.cat(expert_hiddens, dim=-1)]
        if modality_flags is not None:
            gate_parts.append(modality_flags.float())
        gate_weights = torch.softmax(self.gate(torch.cat(gate_parts, dim=-1)), dim=-1)

        out = (expert_outputs.squeeze(-1) * gate_weights).sum(dim=-1, keepdim=True)
        if self.skip is not None:
            out = out + self.skip(fused)
        return out


class ModalityAwareMoEHead(BaseHead):
    """
    Stable 4-expert MoE head, close to the image_v2_sim_v1-style setup.

    Experts:
      - full:  fused only
      - meta:  fused + structured metadata feature
      - text:  fused + post text feature
      - image: fused + image feature (+ clip_sim if enabled)

    This head does not receive user_desc/loc_desc availability flags.
    If profile/location auxiliary vectors are enabled, fusion_model.py adds
    them as small residuals to fused before this head.
    """
    def __init__(
        self,
        input_dim: int,
        num_experts: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_skip: bool = True,
        use_clip_similarity: bool = False,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        self.use_skip = use_skip
        self.use_clip_similarity = use_clip_similarity

        def act():
            return _get_activation(activation)

        extra_dim = 1 if use_clip_similarity else 0

        self.full_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.meta_proj = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            act(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            act(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.image_proj = nn.Sequential(
            nn.Linear(input_dim * 2 + extra_dim, hidden_dim),
            act(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                act(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_experts)
        ])

        gate_input_dim = hidden_dim * num_experts + extra_dim

        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, max(hidden_dim // 2, 64)),
            act(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, 64), num_experts),
        )

        self.skip = nn.Linear(input_dim, 1) if use_skip else None

    def forward(
        self,
        fused: torch.Tensor,
        text_feat: torch.Tensor,
        meta_feat: torch.Tensor,
        image_feat: torch.Tensor,
        clip_sim: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_clip_similarity:
            if clip_sim is None:
                raise ValueError("use_clip_similarity=True but clip_sim is None.")
            image_input = torch.cat([fused, image_feat, clip_sim], dim=-1)
        else:
            image_input = torch.cat([fused, image_feat], dim=-1)

        full_h = self.full_proj(fused)
        meta_h = self.meta_proj(torch.cat([fused, meta_feat], dim=-1))
        text_h = self.text_proj(torch.cat([fused, text_feat], dim=-1))
        image_h = self.image_proj(image_input)

        expert_inputs = [full_h, meta_h, text_h, image_h]

        expert_outputs = torch.stack(
            [expert(h) for expert, h in zip(self.experts, expert_inputs)],
            dim=1,
        )  # [B, 4, 1]

        gate_parts = expert_inputs
        if self.use_clip_similarity:
            gate_parts = gate_parts + [clip_sim]

        gate_input = torch.cat(gate_parts, dim=-1)
        gate_weights = torch.softmax(self.gate(gate_input), dim=-1)

        out = (expert_outputs.squeeze(-1) * gate_weights).sum(
            dim=-1,
            keepdim=True,
        )

        if self.skip is not None:
            out = out + self.skip(fused)

        return out