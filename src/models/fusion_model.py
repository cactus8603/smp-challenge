from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.text_encoder import TextEncoder
from src.models.meta_encoder import MetaEncoder, VectorCompressor
from src.models.image_encoder import build_image_encoder
from src.models.fusion import ConcatFusion, CrossFeatureFusion, PairwiseGatedFusion, ResidualCrossAttentionFusion
from src.models.head import RegressionHead, ModalityAwareMoEHead


class SMPFusionModel(nn.Module):
    """
    1c89-style backbone adapted to the current data pipeline.

    Backbone from 1c89:
    - PairwiseGatedFusion
    - optional raw CLIP text-image cosine similarity
    - ModalityAwareMoEHead

    Current-data adaptations:
    - semantic grouped metadata support via meta_num_cols/meta_cat_cols/meta_bin_cols
    - user_desc optional semantic auxiliary vector, compressed to a tiny gated residual
    - loc_desc is metadata-only, not an auxiliary modality
    - supports modality_mask including user_desc for ablation
    """

    def __init__(
        self,
        text_model_name: str = "openai/clip-vit-base-patch32",
        meta_num_dim: int = 0,
        meta_cat_cardinalities: Optional[Sequence[int]] = None,
        meta_bin_dim: int = 0,
        meta_num_cols: Optional[Sequence[str]] = None,
        meta_cat_cols: Optional[Sequence[str]] = None,
        meta_bin_cols: Optional[Sequence[str]] = None,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_text: bool = True,
        use_tag_embedding: bool = False,
        tag_encoder: Optional[nn.Module] = None,
        use_meta: bool = True,
        use_image: bool = True,
        image_model_name: str = "openai/clip-vit-base-patch32",
        text_pooling: str = "clip",
        text_trainable: bool = False,
        image_pretrained: bool = True,
        image_trainable: bool = False,
        fusion_type: str = "pairwise_gated",
        meta_branch_dim: int = 128,
        use_semantic_meta_groups: bool = True,
        meta_encoder_type: str = "legacy",
        ft_token_dim: int = 64,
        ft_num_layers: int = 2,
        ft_num_heads: int = 8,
        ft_ffn_mult: int = 2,
        ft_dropout: Optional[float] = None,
        use_clip_similarity: bool = True,
        clip_similarity_mode: str = "raw",  # raw | projected | both
        head_type: str = "moe",             # moe | regression
        # non-MoE metadata rescue path
        use_meta_aux_head: bool = False,
        meta_aux_scale: float = 0.0,
        use_meta_residual: bool = True,
        meta_residual_init: float = 0.0,
        meta_residual_dropout: float = 0.0,
        head_hidden_mult: float = 1.0,
        head_num_layers: int = 2,
        fusion_num_heads: int = 4,
        fusion_meta_res_scale: float = 0.75,
        fusion_text_res_scale: float = 0.50,
        fusion_image_res_scale: float = 0.50,
        fusion_tm_res_scale: float = 0.75,
        fusion_order: str = "text_meta_first",
        normalize_image_feature: bool = False,
        # current user/location description handling
        use_user_desc: bool = False,
        use_loc_desc: bool = False,
        user_desc_dim: int = 768,
        loc_desc_dim: int = 400,
        desc_bottleneck_dim: int = 32,
        user_desc_scale: float = 0.01,
        loc_desc_scale: float = 0.0,
    ) -> None:
        super().__init__()

        if not any([use_text, use_tag_embedding, use_meta, use_image]):
            raise ValueError("At least one modality must be enabled.")

        self.use_text = use_text
        self.use_tag_embedding = use_tag_embedding
        self.use_meta = use_meta
        self.use_image = use_image
        self.use_clip_similarity = bool(use_clip_similarity)
        self.clip_similarity_mode = str(clip_similarity_mode).lower()
        if self.clip_similarity_mode not in {"raw", "projected", "both"}:
            raise ValueError(f"Unsupported clip_similarity_mode: {clip_similarity_mode}")
        self.head_type = str(head_type).lower()
        if self.head_type not in {"moe", "regression"}:
            raise ValueError(f"Unsupported head_type: {head_type}")
        self.normalize_image_feature = bool(normalize_image_feature)
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type.lower()
        self.use_meta_aux_head = bool(use_meta_aux_head)
        self.meta_aux_scale = float(meta_aux_scale)
        self.use_meta_residual = bool(use_meta_residual)
        self.meta_residual_init = float(meta_residual_init)
        self.meta_residual_dropout = float(meta_residual_dropout)
        self.fusion_num_heads = int(fusion_num_heads)
        self.fusion_order = str(fusion_order or "text_meta_first").lower()
        self.head_hidden_mult = float(head_hidden_mult)
        self.head_num_layers = int(head_num_layers)

        if self.use_clip_similarity and not (self.use_text and self.use_image):
            raise ValueError("use_clip_similarity=True requires both use_text=True and use_image=True.")

        if self.use_clip_similarity and text_model_name != image_model_name:
            raise ValueError(
                "use_clip_similarity=True requires text_model_name and image_model_name "
                "to be the same CLIP model."
            )

        self.text_encoder: Optional[nn.Module] = None
        self.tag_encoder: Optional[nn.Module] = None
        self.meta_encoder: Optional[nn.Module] = None
        self.image_encoder: Optional[nn.Module] = None

        fusion_input_dims: Dict[str, int] = {}

        clip_text_dim = 0
        tag_text_dim = 0

        if self.use_text:
            self.text_encoder = TextEncoder(
                model_name=text_model_name,
                output_dim=hidden_dim,
                pooling=text_pooling,
                dropout=dropout,
                trainable=text_trainable,
            )
            clip_text_dim = hidden_dim

        if self.use_tag_embedding:
            if tag_encoder is None:
                raise ValueError("use_tag_embedding=True but tag_encoder is None.")
            self.tag_encoder = tag_encoder
            tag_text_dim = int(
                getattr(tag_encoder, "output_dim", getattr(tag_encoder, "embed_dim", hidden_dim))
            )

        self.text_branch_enabled = self.use_text or self.use_tag_embedding

        self.tag_proj: Optional[nn.Module] = None
        self.tag_only_proj: Optional[nn.Module] = None
        self.text_dropout = nn.Dropout(dropout)
        self.tag_alpha = nn.Parameter(torch.tensor(0.10))

        if self.use_tag_embedding and tag_text_dim != hidden_dim:
            self.tag_proj = nn.Sequential(
                nn.Linear(tag_text_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        if (not self.use_text) and self.use_tag_embedding and tag_text_dim != hidden_dim:
            self.tag_only_proj = nn.Sequential(
                nn.Linear(tag_text_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        if self.text_branch_enabled:
            fusion_input_dims["text"] = hidden_dim

        if self.use_meta:
            if meta_num_dim <= 0 and meta_bin_dim <= 0 and not meta_cat_cardinalities:
                raise ValueError(
                    "When use_meta=True, at least one of "
                    "meta_num_dim/meta_bin_dim/meta_cat_cardinalities must be non-empty."
                )

            self.meta_encoder = MetaEncoder(
                num_input_dim=meta_num_dim,
                cat_cardinalities=list(meta_cat_cardinalities or []),
                bin_input_dim=meta_bin_dim,
                output_dim=hidden_dim,
                branch_dim=meta_branch_dim,
                dropout=dropout,
                activation="gelu",
                use_layernorm=True,
                use_user_desc=False,
                use_loc_desc=False,
                user_desc_dim=user_desc_dim,
                loc_desc_dim=loc_desc_dim,
                desc_bottleneck_dim=desc_bottleneck_dim,
                meta_num_cols=list(meta_num_cols or []),
                meta_cat_cols=list(meta_cat_cols or []),
                meta_bin_cols=list(meta_bin_cols or []),
                use_semantic_groups=bool(use_semantic_meta_groups),
                encoder_type=str(meta_encoder_type),
                ft_token_dim=int(ft_token_dim),
                ft_num_layers=int(ft_num_layers),
                ft_num_heads=int(ft_num_heads),
                ft_ffn_mult=int(ft_ffn_mult),
                ft_dropout=ft_dropout,
            )
            fusion_input_dims["meta"] = hidden_dim

        self.image_encoder = build_image_encoder(
            use_image=use_image,
            image_model_name=image_model_name,
            output_dim=hidden_dim,
            pretrained=image_pretrained,
            trainable=image_trainable,
            dropout=dropout,
            placeholder_when_disabled=True,
        )

        if self.use_image:
            fusion_input_dims["image"] = hidden_dim

        self.clip_sim_dim = 0
        if self.use_clip_similarity:
            self.clip_sim_dim = 2 if self.clip_similarity_mode == "both" else 1
            fusion_input_dims["clip_sim"] = self.clip_sim_dim

        # user_desc is a weak auxiliary residual. This is intentionally tiny
        # because previous ablation showed small user_desc contribution.
        self.use_user_desc_aux = bool(use_user_desc and user_desc_dim > 0 and user_desc_scale != 0)
        self.user_desc_aux_scale = float(user_desc_scale)
        self.use_loc_desc_aux = False
        self.loc_desc_aux_scale = 0.0

        self.user_desc_aux_encoder: Optional[nn.Module] = (
            VectorCompressor(
                input_dim=user_desc_dim,
                bottleneck_dim=desc_bottleneck_dim,
                output_dim=hidden_dim,
                dropout=dropout,
            )
            if self.use_user_desc_aux
            else None
        )
        self.loc_desc_aux_encoder: Optional[nn.Module] = None

        gate_hidden_dim = max(hidden_dim // 4, 32)
        self.user_desc_gate: Optional[nn.Module] = (
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + 1, gate_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden_dim, 1),
            )
            if self.use_user_desc_aux
            else None
        )
        if self.user_desc_gate is not None:
            # Start almost closed; let the model use user_desc only if helpful.
            nn.init.constant_(self.user_desc_gate[-1].bias, -4.0)
            nn.init.zeros_(self.user_desc_gate[-1].weight)

        self.loc_desc_gate: Optional[nn.Module] = None

        if self.fusion_type == "concat":
            self.fusion = ConcatFusion(
                input_dims=fusion_input_dims,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
                activation="relu",
                use_layernorm=False,
            )
        elif self.fusion_type == "cross_feature":
            if not (self.text_branch_enabled and self.use_meta and self.use_image):
                raise ValueError(
                    "cross_feature fusion requires text_branch_enabled=True, "
                    "use_meta=True, and use_image=True."
                )
            self.fusion = CrossFeatureFusion(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
                activation="relu",
                use_layernorm=False,
                extra_dim=self.clip_sim_dim,
            )
        elif self.fusion_type == "pairwise_gated":
            if not (self.text_branch_enabled and self.use_meta and self.use_image):
                raise ValueError(
                    "pairwise_gated fusion requires text_branch_enabled=True, "
                    "use_meta=True, and use_image=True."
                )
            self.fusion = PairwiseGatedFusion(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                pair_hidden_dim=hidden_dim,
                dropout=dropout,
                activation="relu",
                use_layernorm=False,
                extra_dim=self.clip_sim_dim,
            )
        elif self.fusion_type in {"residual_cross_attention", "res_cross_attn", "cross_attention_residual"}:
            if not (self.text_branch_enabled and self.use_meta and self.use_image):
                raise ValueError(
                    "residual_cross_attention fusion requires text_branch_enabled=True, "
                    "use_meta=True, and use_image=True."
                )
            self.fusion = ResidualCrossAttentionFusion(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_heads=self.fusion_num_heads,
                dropout=dropout,
                activation="gelu",
                use_layernorm=True,
                extra_dim=self.clip_sim_dim,
                meta_res_scale=float(fusion_meta_res_scale),
                text_res_scale=float(fusion_text_res_scale),
                image_res_scale=float(fusion_image_res_scale),
                tm_res_scale=float(fusion_tm_res_scale),
                order=self.fusion_order,
            )
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        print(
            f"[DEBUG] fusion_type = {self.fusion_type}, "
            f"fusion_class = {self.fusion.__class__.__name__}, "
            f"use_clip_similarity = {self.use_clip_similarity}, "
            f"clip_similarity_mode = {self.clip_similarity_mode}, "
            f"head_type = {self.head_type}, "
            f"normalize_image_feature = {self.normalize_image_feature}, "
            f"use_user_desc_aux = {self.use_user_desc_aux}, "
            f"user_desc_scale = {self.user_desc_aux_scale}, "
            f"use_meta_aux_head = {self.use_meta_aux_head}, "
            f"meta_aux_scale = {self.meta_aux_scale}, "
            f"use_meta_residual = {self.use_meta_residual}, "
            f"meta_residual_gate_init = {self.meta_residual_init}, "
            f"fusion_order = {self.fusion_order}, "
            f"head_hidden_mult = {self.head_hidden_mult}, "
            f"head_num_layers = {self.head_num_layers}"
        )

        head_hidden_dim = max(hidden_dim, int(round(hidden_dim * self.head_hidden_mult)))
        head_hidden_dims = [head_hidden_dim for _ in range(max(self.head_num_layers, 1))]

        if self.head_type == "moe":
            self.head = ModalityAwareMoEHead(
                input_dim=hidden_dim,
                num_experts=4,
                hidden_dim=head_hidden_dim,
                dropout=dropout,
                activation="relu",
                use_skip=True,
                use_clip_similarity=self.use_clip_similarity,
            )
        else:
            self.head = RegressionHead(
                input_dim=hidden_dim,
                hidden_dims=head_hidden_dims,
                dropout=dropout,
                activation="relu",
                use_layernorm=True,
                use_skip=True,
            )

        # Always build meta_aux_head when metadata is enabled. It is only added to
        # the final prediction when use_meta_aux_head=True, but the same head is
        # also used by optional meta-only warmup training.
        self.meta_aux_head: Optional[nn.Module] = None
        if self.use_meta:
            self.meta_aux_head = RegressionHead(
                input_dim=hidden_dim,
                hidden_dims=head_hidden_dims,
                dropout=dropout,
                activation="relu",
                use_layernorm=True,
                use_skip=True,
            )

        # Gated metadata residual injection after modality fusion.
        # This keeps metadata influential at representation level without widening
        # the head input and without using MoE/expert competition.
        self.meta_residual_proj: Optional[nn.Module] = None
        self.meta_residual_norm: Optional[nn.Module] = None
        self.meta_residual_gate: Optional[nn.Parameter] = None
        self.meta_residual_dropout_layer: nn.Module = nn.Dropout(self.meta_residual_dropout) if self.meta_residual_dropout > 0 else nn.Identity()
        if self.use_meta and self.use_meta_residual:
            self.meta_residual_proj = nn.Linear(hidden_dim, hidden_dim)
            self.meta_residual_norm = nn.LayerNorm(hidden_dim)
            # sigmoid(0.0)=0.5, so metadata residual starts half-open.
            self.meta_residual_gate = nn.Parameter(torch.tensor(self.meta_residual_init, dtype=torch.float32))

    def _infer_batch_size(
        self,
        input_ids: Optional[torch.Tensor],
        meta_num: Optional[torch.Tensor],
        meta_cat: Optional[torch.Tensor],
        meta_bin: Optional[torch.Tensor],
        image_tensor: Optional[torch.Tensor],
        tag_token_count: Optional[torch.Tensor] = None,
        tag_tokens: Optional[Sequence[Sequence[str]]] = None,
        user_desc: Optional[torch.Tensor] = None,
        loc_desc: Optional[torch.Tensor] = None,
    ) -> int:
        if input_ids is not None:
            return input_ids.size(0)
        if meta_num is not None:
            return meta_num.size(0)
        if meta_cat is not None:
            return meta_cat.size(0)
        if meta_bin is not None:
            return meta_bin.size(0)
        if image_tensor is not None:
            return image_tensor.size(0)
        if user_desc is not None:
            return user_desc.size(0)
        if loc_desc is not None:
            return loc_desc.size(0)
        if tag_token_count is not None:
            return tag_token_count.size(0)
        if tag_tokens is not None:
            return len(tag_tokens)
        raise ValueError("Cannot infer batch size from all-None inputs.")

    def _infer_device(
        self,
        input_ids: Optional[torch.Tensor],
        meta_num: Optional[torch.Tensor],
        meta_cat: Optional[torch.Tensor],
        meta_bin: Optional[torch.Tensor],
        image_tensor: Optional[torch.Tensor],
        tag_token_count: Optional[torch.Tensor] = None,
        user_desc: Optional[torch.Tensor] = None,
        loc_desc: Optional[torch.Tensor] = None,
    ) -> torch.device:
        if input_ids is not None:
            return input_ids.device
        if meta_num is not None:
            return meta_num.device
        if meta_cat is not None:
            return meta_cat.device
        if meta_bin is not None:
            return meta_bin.device
        if image_tensor is not None:
            return image_tensor.device
        if user_desc is not None:
            return user_desc.device
        if loc_desc is not None:
            return loc_desc.device
        if tag_token_count is not None:
            return tag_token_count.device
        return torch.device("cpu")

    def extract_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        meta_num: Optional[torch.Tensor] = None,
        meta_cat: Optional[torch.Tensor] = None,
        meta_bin: Optional[torch.Tensor] = None,
        image_tensor: Optional[torch.Tensor] = None,
        tag_tokens: Optional[Sequence[Sequence[str]]] = None,
        tag_text: Optional[Sequence[str]] = None,
        tag_token_count: Optional[torch.Tensor] = None,
        user_desc: Optional[torch.Tensor] = None,
        loc_desc: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        batch_size = self._infer_batch_size(
            input_ids=input_ids,
            meta_num=meta_num,
            meta_cat=meta_cat,
            meta_bin=meta_bin,
            image_tensor=image_tensor,
            tag_token_count=tag_token_count,
            tag_tokens=tag_tokens,
            user_desc=user_desc,
            loc_desc=loc_desc,
        )
        device = self._infer_device(
            input_ids=input_ids,
            meta_num=meta_num,
            meta_cat=meta_cat,
            meta_bin=meta_bin,
            image_tensor=image_tensor,
            tag_token_count=tag_token_count,
            user_desc=user_desc,
            loc_desc=loc_desc,
        )

        features: Dict[str, Optional[torch.Tensor]] = {
            "text": None,
            "meta": None,
            "image": None,
            "clip_text": None,
            "tag_text": None,
            "clip_sim": None,
            "user_desc": None,
            "user_desc_mask": None,
            "user_desc_gate": None,
            "loc_desc": None,
            "loc_desc_mask": None,
        }

        clip_text_feat: Optional[torch.Tensor] = None
        raw_clip_text_feat: Optional[torch.Tensor] = None
        raw_clip_image_feat: Optional[torch.Tensor] = None

        # -------------------------
        # CLIP text
        # -------------------------
        if self.use_text:
            if input_ids is None or attention_mask is None:
                raise ValueError("CLIP text branch enabled but input_ids/attention_mask is missing.")

            if self.use_clip_similarity:
                clip_text_feat, raw_clip_text_feat = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_raw_clip=True,
                )
            else:
                clip_text_feat = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            features["clip_text"] = clip_text_feat

        # -------------------------
        # fastText tag / keyword embedding branch
        # Reuses batch["tag_tokens"] as tokenized tags/keywords.
        # -------------------------
        tag_text_feat: Optional[torch.Tensor] = None
        if self.use_tag_embedding:
            if tag_tokens is None:
                raise ValueError("Tag embedding branch enabled but tag_tokens/tag_tokens is missing.")
            tag_text_feat = self.tag_encoder(tag_tokens=tag_tokens)
            features["tag_text"] = tag_text_feat

        # -------------------------
        # Text branch fusion: CLIP + tag
        # -------------------------
        if self.text_branch_enabled:
            if clip_text_feat is not None and tag_text_feat is not None:
                tag_delta = tag_text_feat
                if self.tag_proj is not None:
                    tag_delta = self.tag_proj(tag_delta)
                features["text"] = self.text_dropout(
                    clip_text_feat + self.tag_alpha * tag_delta
                )

            elif clip_text_feat is not None:
                features["text"] = clip_text_feat

            elif tag_text_feat is not None:
                tag_only = tag_text_feat
                if tag_only.size(-1) != self.hidden_dim:
                    if self.tag_only_proj is None:
                        raise ValueError(
                            f"tag-only text feature dim mismatch: "
                            f"expected {self.hidden_dim}, got {tag_only.size(-1)}."
                        )
                    tag_only = self.tag_only_proj(tag_only)
                features["text"] = tag_only

        # -------------------------
        # Metadata
        # -------------------------
        if self.use_meta:
            if meta_num is None and meta_cat is None and meta_bin is None:
                raise ValueError("Meta branch enabled but meta_num/meta_cat/meta_bin are all missing.")

            features["meta"] = self.meta_encoder(
                meta_num=meta_num,
                meta_cat=meta_cat,
                meta_bin=meta_bin,
            )

        # -------------------------
        # CLIP image
        # -------------------------
        if self.use_image:
            if image_tensor is None:
                raise ValueError("Image branch enabled but image_tensor is missing.")

            if self.use_clip_similarity:
                image_feat, raw_clip_image_feat = self.image_encoder(
                    image_tensor,
                    return_raw_clip=True,
                )
            else:
                image_feat = self.image_encoder(image_tensor)

            if self.normalize_image_feature:
                image_feat = F.normalize(image_feat, p=2, dim=-1)

            # default / 1c89 behavior: keep projected image feature as-is.
            features["image"] = image_feat

        else:
            features["image"] = self.image_encoder(
                image_tensor=None,
                batch_size=batch_size,
                device=device,
            )

        # -------------------------
        # CLIP text-image similarity
        # -------------------------
        if self.use_clip_similarity:
            sim_parts = []
            if self.clip_similarity_mode in {"raw", "both"}:
                if raw_clip_text_feat is None or raw_clip_image_feat is None:
                    raise ValueError(
                        "clip_similarity_mode='raw/both' requires raw CLIP text/image features."
                    )
                raw_sim = F.cosine_similarity(
                    raw_clip_text_feat,
                    raw_clip_image_feat,
                    dim=-1,
                    eps=1e-8,
                ).unsqueeze(-1)
                sim_parts.append(raw_sim)

            if self.clip_similarity_mode in {"projected", "both"}:
                if clip_text_feat is None or features.get("image") is None:
                    raise ValueError(
                        "clip_similarity_mode='projected/both' requires projected text/image features."
                    )
                projected_sim = F.cosine_similarity(
                    clip_text_feat,
                    features["image"],
                    dim=-1,
                    eps=1e-8,
                ).unsqueeze(-1)
                sim_parts.append(projected_sim)

            features["clip_sim"] = torch.cat(sim_parts, dim=-1)

        # -------------------------
        # Tiny user_desc auxiliary residual feature
        # -------------------------
        if self.use_user_desc_aux:
            if user_desc is None:
                features["user_desc"] = torch.zeros(batch_size, self.hidden_dim, device=device)
                features["user_desc_mask"] = torch.zeros(batch_size, 1, device=device)
            else:
                user_desc = user_desc.to(device).float()
                user_desc_mask = (user_desc.abs().sum(dim=-1, keepdim=True) > 0).float()
                user_desc_feat = self.user_desc_aux_encoder(user_desc) * user_desc_mask
                features["user_desc"] = user_desc_feat
                features["user_desc_mask"] = user_desc_mask

        # loc_desc intentionally ignored as a standalone modality.
        # location information should enter through metadata features.
        features["loc_desc"] = None
        features["loc_desc_mask"] = None

        return features

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        meta_num: Optional[torch.Tensor] = None,
        meta_cat: Optional[torch.Tensor] = None,
        meta_bin: Optional[torch.Tensor] = None,
        image_tensor: Optional[torch.Tensor] = None,
        tag_tokens: Optional[Sequence[Sequence[str]]] = None,
        tag_text: Optional[Sequence[str]] = None,
        tag_token_count: Optional[torch.Tensor] = None,
        user_desc: Optional[torch.Tensor] = None,
        loc_desc: Optional[torch.Tensor] = None,
        return_features: bool = False,
        modality_mask: Optional[Dict[str, bool]] = None,
        meta_only: bool = False,
    ):
        features = self.extract_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            meta_num=meta_num,
            meta_cat=meta_cat,
            meta_bin=meta_bin,
            image_tensor=image_tensor,
            tag_tokens=tag_tokens,
            tag_text=tag_text,
            tag_token_count=tag_token_count,
            user_desc=user_desc,
            loc_desc=loc_desc,
        )

        if modality_mask is not None:
            for name in ["text", "meta", "image"]:
                if modality_mask.get(name, False) and features.get(name) is not None:
                    features[name] = torch.zeros_like(features[name])

            if modality_mask.get("user_desc", False) and features.get("user_desc") is not None:
                features["user_desc"] = torch.zeros_like(features["user_desc"])
                if features.get("user_desc_mask") is not None:
                    features["user_desc_mask"] = torch.zeros_like(features["user_desc_mask"])

            if modality_mask.get("text", False) or modality_mask.get("image", False):
                if features.get("clip_sim") is not None:
                    features["clip_sim"] = torch.zeros_like(features["clip_sim"])

        if meta_only:
            if features.get("meta") is None:
                raise ValueError("meta_only=True requires metadata features.")
            if self.meta_aux_head is None:
                raise ValueError("meta_only=True requires meta_aux_head.")
            output = self.meta_aux_head(features["meta"])
            if return_features:
                return {
                    "output": output,
                    "fused": features["meta"],
                    "features": features,
                }
            return output

        fused = self.fusion(features)

        # Gated metadata residual after fusion.
        # Skip this during mask_meta ablation so the ablation remains honest.
        meta_masked = bool(modality_mask is not None and modality_mask.get("meta", False))
        if (
            self.use_meta_residual
            and not meta_masked
            and features.get("meta") is not None
            and self.meta_residual_proj is not None
            and self.meta_residual_norm is not None
            and self.meta_residual_gate is not None
        ):
            meta_residual = self.meta_residual_norm(self.meta_residual_proj(features["meta"]))
            meta_residual = self.meta_residual_dropout_layer(meta_residual)
            meta_gate = torch.sigmoid(self.meta_residual_gate)
            fused = fused + meta_gate * meta_residual
            features["meta_residual_gate"] = meta_gate.detach().reshape(1)

        # Add tiny gated user_desc residual after fusion.
        if features.get("user_desc") is not None:
            user_desc_mask = features.get(
                "user_desc_mask",
                torch.ones(fused.size(0), 1, device=fused.device),
            )
            if self.user_desc_gate is not None:
                user_gate = torch.sigmoid(
                    self.user_desc_gate(
                        torch.cat([fused, features["user_desc"], user_desc_mask], dim=-1)
                    )
                ) * user_desc_mask
            else:
                user_gate = user_desc_mask
            fused = fused + self.user_desc_aux_scale * user_gate * features["user_desc"]
            features["user_desc_gate"] = user_gate

        if self.head_type == "moe":
            output = self.head(
                fused=fused,
                text_feat=features["text"],
                meta_feat=features["meta"],
                image_feat=features["image"],
                clip_sim=features["clip_sim"],
            )
        else:
            output = self.head(fused)

        # Optional independent metadata prediction path.
        # This avoids forcing metadata to compete through MoE/fusion only.
        # During mask_meta ablation, skip this branch so the ablation is honest.
        if self.meta_aux_head is not None and not meta_masked and features.get("meta") is not None:
            output = output + self.meta_aux_scale * self.meta_aux_head(features["meta"])

        if return_features:
            return {
                "output": output,
                "fused": fused,
                "features": features,
            }

        return output
