from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from src.models.text_encoder import TextEncoder
from src.models.meta_encoder import MetaEncoder
from src.models.image_encoder import build_image_encoder
from src.models.fusion import ConcatFusion, CrossFeatureFusion, PairwiseGatedFusion
from src.models.head import RegressionHead


class SMPFusionModel(nn.Module):
    """
    Main multimodal model for SMP popularity prediction.

    Score-first version:
    - keep CLIP text / image encoders frozen by default
    - keep each modality projected to the same hidden_dim
    - support stronger fusion via PairwiseGatedFusion
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
    ) -> None:
        super().__init__()

        if not any([use_text, use_glove, use_meta, use_image]):
            raise ValueError("At least one modality must be enabled.")

        self.use_text = use_text
        self.use_glove = use_glove
        self.use_meta = use_meta
        self.use_image = use_image
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type.lower()

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
            )
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        print(f"[DEBUG] fusion_type = {self.fusion_type}, fusion_class = {self.fusion.__class__.__name__}")

        self.head = RegressionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation="relu",
            use_layernorm=False,
            use_skip=True,
        )

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
        }

        clip_text_feat: Optional[torch.Tensor] = None
        if self.use_text:
            if input_ids is None or attention_mask is None:
                raise ValueError("CLIP text branch enabled but input_ids/attention_mask is missing.")
            clip_text_feat = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            features["clip_text"] = clip_text_feat

        glove_text_feat: Optional[torch.Tensor] = None
        if self.use_glove:
            if glove_tokens is None:
                raise ValueError("GloVe branch enabled but glove_tokens is missing.")
            glove_text_feat = self.glove_encoder(glove_tokens=glove_tokens)
            features["glove_text"] = glove_text_feat

        if self.text_branch_enabled:
            if clip_text_feat is not None and glove_text_feat is not None:
                glove_delta = glove_text_feat
                if self.glove_proj is not None:
                    glove_delta = self.glove_proj(glove_delta)
                features["text"] = self.text_dropout(clip_text_feat + self.glove_alpha * glove_delta)
            elif clip_text_feat is not None:
                features["text"] = clip_text_feat
            elif glove_text_feat is not None:
                glove_only = glove_text_feat
                if glove_only.size(-1) != self.hidden_dim:
                    if self.glove_only_proj is None:
                        raise ValueError(
                            f"GloVe-only text feature dim mismatch: expected {self.hidden_dim}, got {glove_only.size(-1)}."
                        )
                    glove_only = self.glove_only_proj(glove_only)
                features["text"] = glove_only

        if self.use_meta:
            if meta_num is None and meta_cat is None and meta_bin is None:
                raise ValueError("Meta branch enabled but meta_num/meta_cat/meta_bin are all missing.")
            features["meta"] = self.meta_encoder(
                meta_num=meta_num,
                meta_cat=meta_cat,
                meta_bin=meta_bin,
            )

        if self.use_image:
            if image_tensor is None:
                raise ValueError("Image branch enabled but image_tensor is missing.")
            image_feat = self.image_encoder(image_tensor)
            features["image"] = image_feat
        else:
            features["image"] = self.image_encoder(
                image_tensor=None,
                batch_size=batch_size,
                device=device,
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
        return_features: bool = False,
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
        )

        fused = self.fusion(features)
        output = self.head(fused)

        if return_features:
            return {
                "output": output,
                "fused": fused,
                "features": features,
            }

        return output