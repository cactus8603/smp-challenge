from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.text_encoder import TextEncoder
from src.models.meta_encoder import MetaEncoder, VectorCompressor
from src.models.image_encoder import build_image_encoder
from src.models.fusion import ConcatFusion, CrossFeatureFusion, PairwiseGatedFusion
from src.models.head import RegressionHead, ModalityAwareMoEHead
# Legacy head is intentionally not imported/used in this version:
# from src.models.head import MetaWeightedMoEHead


class SMPFusionModel(nn.Module):
    """
    Main multimodal model for SMP popularity prediction.

    Score-first version:
    - keep CLIP text / image encoders frozen by default
    - keep each modality projected to the same hidden_dim
    - support stronger fusion via PairwiseGatedFusion
    - optionally add CLIP text-image cosine similarity as an extra fusion feature
    """

    def __init__(
        self,
        text_model_name: str = "openai/clip-vit-base-patch32",
        meta_num_dim: int = 0,
        meta_cat_cardinalities: Optional[Sequence[int]] = None,
        meta_bin_dim: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_text: bool = True,
        use_glove: bool = False,
        glove_encoder: Optional[nn.Module] = None,
        use_meta: bool = True,
        use_image: bool = True,
        image_model_name: str = "openai/clip-vit-base-patch32",
        text_pooling: str = "clip",
        text_trainable: bool = False,
        image_pretrained: bool = True,
        image_trainable: bool = False,
        fusion_type: str = "pairwise_gated",
        meta_branch_dim: int = 128,
        use_clip_similarity: bool = False,
        # ── user/location description vector branches ──
        use_user_desc: bool = True,
        use_loc_desc: bool = False,
        user_desc_dim: int = 768,
        loc_desc_dim: int = 400,
        desc_bottleneck_dim: int = 64,
        user_desc_scale: float = 0.10,
        loc_desc_scale: float = 0.05,
    ) -> None:
        super().__init__()

        if not any([use_text, use_glove, use_meta, use_image]):
            raise ValueError("At least one modality must be enabled.")

        self.use_text = use_text
        self.use_glove = use_glove
        self.use_meta = use_meta
        self.use_image = use_image
        self.use_clip_similarity = use_clip_similarity
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type.lower()

        if self.use_clip_similarity and not (self.use_text and self.use_image):
            raise ValueError("use_clip_similarity=True requires both use_text=True and use_image=True.")

        if self.use_clip_similarity and text_model_name != image_model_name:
            raise ValueError(
                "use_clip_similarity=True requires text_model_name and image_model_name "
                "to be the same CLIP model."
            )

        self.text_encoder: Optional[nn.Module] = None
        self.glove_encoder: Optional[nn.Module] = None
        self.meta_encoder: Optional[nn.Module] = None
        self.image_encoder: Optional[nn.Module] = None

        fusion_input_dims: Dict[str, int] = {}

        clip_text_dim = 0
        glove_text_dim = 0

        if self.use_text:
            self.text_encoder = TextEncoder(
                model_name=text_model_name,
                output_dim=hidden_dim,
                pooling=text_pooling,
                dropout=dropout,
                trainable=text_trainable,
            )
            clip_text_dim = hidden_dim

        if self.use_glove:
            if glove_encoder is None:
                raise ValueError("use_glove=True but glove_encoder is None.")
            self.glove_encoder = glove_encoder
            glove_text_dim = int(
                getattr(glove_encoder, "output_dim", getattr(glove_encoder, "embed_dim", hidden_dim))
            )

        self.text_branch_enabled = self.use_text or self.use_glove

        self.glove_proj: Optional[nn.Module] = None
        self.glove_only_proj: Optional[nn.Module] = None
        self.text_dropout = nn.Dropout(dropout)
        self.glove_alpha = nn.Parameter(torch.tensor(0.10))

        if self.use_glove and glove_text_dim != hidden_dim:
            self.glove_proj = nn.Sequential(
                nn.Linear(glove_text_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        if (not self.use_text) and self.use_glove and glove_text_dim != hidden_dim:
            self.glove_only_proj = nn.Sequential(
                nn.Linear(glove_text_dim, hidden_dim),
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
                activation="relu",
                use_layernorm=True,
                # IMPORTANT: keep structured metadata clean.
                # user_desc / loc_desc are handled as auxiliary residual branches
                # in SMPFusionModel, not mixed into MetaEncoder.
                use_user_desc=False,
                use_loc_desc=False,
                user_desc_dim=user_desc_dim,
                loc_desc_dim=loc_desc_dim,
                desc_bottleneck_dim=desc_bottleneck_dim,
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

        # Auxiliary profile/location text branches.
        # These are intentionally NOT part of MetaEncoder, so structured metadata
        # keeps its own capacity and is not diluted by text-like profile vectors.
        self.use_user_desc_aux = bool(use_user_desc and user_desc_dim > 0)
        self.use_loc_desc_aux = bool(use_loc_desc and loc_desc_dim > 0)
        self.user_desc_aux_scale = float(user_desc_scale)
        self.loc_desc_aux_scale = float(loc_desc_scale)

        self.user_desc_aux_encoder: Optional[nn.Module] = (
            VectorCompressor(
                input_dim=user_desc_dim,
                bottleneck_dim=desc_bottleneck_dim,
                output_dim=hidden_dim,
                dropout=dropout,
            )
            if self.use_user_desc_aux else None
        )
        self.loc_desc_aux_encoder: Optional[nn.Module] = (
            VectorCompressor(
                input_dim=loc_desc_dim,
                bottleneck_dim=desc_bottleneck_dim,
                output_dim=hidden_dim,
                dropout=dropout,
            )
            if self.use_loc_desc_aux else None
        )

        if self.use_image:
            fusion_input_dims["image"] = hidden_dim

        if self.use_clip_similarity:
            fusion_input_dims["clip_sim"] = 1

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
                extra_dim=1 if self.use_clip_similarity else 0,
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
                extra_dim=1 if self.use_clip_similarity else 0,
            )
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        print(
            f"[DEBUG] fusion_type = {self.fusion_type}, "
            f"fusion_class = {self.fusion.__class__.__name__}, "
            f"use_clip_similarity = {self.use_clip_similarity}"
        )

        self.RegressionHead = RegressionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation="relu",
            use_layernorm=False,
            use_skip=True,
        )

        self.ModalityAwareMoEHead = ModalityAwareMoEHead(
            input_dim=hidden_dim,
            num_experts=4,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation="relu",
            use_skip=True,
            use_clip_similarity=self.use_clip_similarity,
        )

        # Legacy / current experimental head (disabled):
        # It over-weighted meta/profile routing through runtime flags and made
        # ablation harder to interpret.  Keep this block here as a reminder,
        # but use the stable 4-expert ModalityAwareMoEHead below in forward().
        #
        # self.MetaWeightedMoEHead = MetaWeightedMoEHead(
        #     input_dim=hidden_dim,
        #     hidden_dim=hidden_dim,
        #     dropout=dropout,
        #     activation="relu",
        #     use_skip=True,
        #     n_modality_flags=3,
        # )

    def _infer_batch_size(
        self,
        input_ids: Optional[torch.Tensor],
        meta_num: Optional[torch.Tensor],
        meta_cat: Optional[torch.Tensor],
        meta_bin: Optional[torch.Tensor],
        image_tensor: Optional[torch.Tensor],
        glove_token_count: Optional[torch.Tensor] = None,
        glove_tokens: Optional[Sequence[Sequence[str]]] = None,
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
        if glove_token_count is not None:
            return glove_token_count.size(0)
        if glove_tokens is not None:
            return len(glove_tokens)
        raise ValueError("Cannot infer batch size from all-None inputs.")

    def _infer_device(
        self,
        input_ids: Optional[torch.Tensor],
        meta_num: Optional[torch.Tensor],
        meta_cat: Optional[torch.Tensor],
        meta_bin: Optional[torch.Tensor],
        image_tensor: Optional[torch.Tensor],
        glove_token_count: Optional[torch.Tensor] = None,
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
        if glove_token_count is not None:
            return glove_token_count.device
        return torch.device("cpu")

    def extract_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        meta_num: Optional[torch.Tensor] = None,
        meta_cat: Optional[torch.Tensor] = None,
        meta_bin: Optional[torch.Tensor] = None,
        image_tensor: Optional[torch.Tensor] = None,
        glove_tokens: Optional[Sequence[Sequence[str]]] = None,
        glove_text: Optional[Sequence[str]] = None,
        glove_token_count: Optional[torch.Tensor] = None,
        user_desc: Optional[torch.Tensor] = None,
        loc_desc: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        batch_size = self._infer_batch_size(
            input_ids=input_ids,
            meta_num=meta_num,
            meta_cat=meta_cat,
            meta_bin=meta_bin,
            image_tensor=image_tensor,
            glove_token_count=glove_token_count,
            glove_tokens=glove_tokens,
        )
        device = self._infer_device(
            input_ids=input_ids,
            meta_num=meta_num,
            meta_cat=meta_cat,
            meta_bin=meta_bin,
            image_tensor=image_tensor,
            glove_token_count=glove_token_count,
        )

        features: Dict[str, Optional[torch.Tensor]] = {
            "text": None,
            "meta": None,
            "image": None,
            "clip_text": None,
            "glove_text": None,
            "clip_sim": None,
            "user_desc": None,
            "loc_desc": None,
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
        # GloVe text
        # -------------------------
        glove_text_feat: Optional[torch.Tensor] = None
        if self.use_glove:
            if glove_tokens is None:
                raise ValueError("GloVe branch enabled but glove_tokens is missing.")
            glove_text_feat = self.glove_encoder(glove_tokens=glove_tokens)
            features["glove_text"] = glove_text_feat

        # -------------------------
        # Text branch fusion: CLIP + GloVe
        # -------------------------
        if self.text_branch_enabled:
            if clip_text_feat is not None and glove_text_feat is not None:
                glove_delta = glove_text_feat
                if self.glove_proj is not None:
                    glove_delta = self.glove_proj(glove_delta)
                features["text"] = self.text_dropout(
                    clip_text_feat + self.glove_alpha * glove_delta
                )

            elif clip_text_feat is not None:
                features["text"] = clip_text_feat

            elif glove_text_feat is not None:
                glove_only = glove_text_feat
                if glove_only.size(-1) != self.hidden_dim:
                    if self.glove_only_proj is None:
                        raise ValueError(
                            f"GloVe-only text feature dim mismatch: "
                            f"expected {self.hidden_dim}, got {glove_only.size(-1)}."
                        )
                    glove_only = self.glove_only_proj(glove_only)
                features["text"] = glove_only

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

            features["image"] = image_feat

        else:
            features["image"] = self.image_encoder(
                image_tensor=None,
                batch_size=batch_size,
                device=device,
            )

        # -------------------------
        # Raw CLIP text-image similarity
        # -------------------------
        if self.use_clip_similarity:
            if raw_clip_text_feat is None or raw_clip_image_feat is None:
                raise ValueError(
                    "use_clip_similarity=True requires raw CLIP text/image features."
                )

            # features["clip_sim"] = F.cosine_similarity(
            #     raw_clip_text_feat,
            #     raw_clip_image_feat,
            #     dim=-1,
            #     eps=1e-8,
            # ).unsqueeze(-1)
            
            features["clip_sim"] = F.cosine_similarity(
                clip_text_feat,
                features["image"],
                dim=-1,
                eps=1e-8,
            ).unsqueeze(-1)

        # -------------------------
        # Auxiliary user/location description residual features
        # -------------------------
        if self.use_user_desc_aux:
            if user_desc is None:
                features["user_desc"] = torch.zeros(batch_size, self.hidden_dim, device=device)
            else:
                user_desc = user_desc.to(device).float()
                has_user_desc = (user_desc.abs().sum(dim=-1, keepdim=True) > 0).float()
                features["user_desc"] = (
                    self.user_desc_aux_encoder(user_desc)
                    * has_user_desc
                    * self.user_desc_aux_scale
                )

        if self.use_loc_desc_aux:
            if loc_desc is None:
                features["loc_desc"] = torch.zeros(batch_size, self.hidden_dim, device=device)
            else:
                loc_desc = loc_desc.to(device).float()
                has_loc_desc = (loc_desc.abs().sum(dim=-1, keepdim=True) > 0).float()
                features["loc_desc"] = (
                    self.loc_desc_aux_encoder(loc_desc)
                    * has_loc_desc
                    * self.loc_desc_aux_scale
                )

        return features

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        meta_num: Optional[torch.Tensor] = None,
        meta_cat: Optional[torch.Tensor] = None,
        meta_bin: Optional[torch.Tensor] = None,
        image_tensor: Optional[torch.Tensor] = None,
        glove_tokens: Optional[Sequence[Sequence[str]]] = None,
        glove_text: Optional[Sequence[str]] = None,
        glove_token_count: Optional[torch.Tensor] = None,
        user_desc: Optional[torch.Tensor] = None,
        loc_desc: Optional[torch.Tensor] = None,
        return_features: bool = False,
        modality_mask: Optional[Dict[str, bool]] = None,
    ):
        features = self.extract_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            meta_num=meta_num,
            meta_cat=meta_cat,
            meta_bin=meta_bin,
            image_tensor=image_tensor,
            glove_tokens=glove_tokens,
            glove_text=glove_text,
            glove_token_count=glove_token_count,
            user_desc=user_desc,
            loc_desc=loc_desc,
        )

        if modality_mask is not None:
            for name in ["text", "meta", "image", "user_desc", "loc_desc"]:
                if modality_mask.get(name, False) and features.get(name) is not None:
                    features[name] = torch.zeros_like(features[name])
            if modality_mask.get("text", False) or modality_mask.get("image", False):
                if features.get("clip_sim") is not None:
                    features["clip_sim"] = torch.zeros_like(features["clip_sim"])

        fused_core = self.fusion(features)
        fused = fused_core
        if features.get("user_desc") is not None:
            fused = fused + features["user_desc"]
        if features.get("loc_desc") is not None:
            fused = fused + features["loc_desc"]

        B = fused.size(0)
        device = fused.device
        zero = lambda: torch.zeros(B, self.hidden_dim, device=device)

        # Stable image_v2_sim_v1-style head:
        # 4 experts = full / meta / text / image, optionally conditioned on clip_sim.
        # user_desc / loc_desc are already added as small residuals to fused above;
        # they do NOT enter the head gate as separate availability flags.
        output = self.ModalityAwareMoEHead(
            fused=fused,
            text_feat=features["text"] if features["text"] is not None else zero(),
            meta_feat=features["meta"] if features["meta"] is not None else zero(),
            image_feat=features["image"] if features["image"] is not None else zero(),
            clip_sim=features.get("clip_sim", None),
        )

        if return_features:
            return {
                "output": output,
                "fused": fused,
                "fused_core": fused_core,
                "features": features,
            }

        return output