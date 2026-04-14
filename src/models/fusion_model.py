from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.text_encoder import TextEncoder
from src.models.meta_encoder import MetaEncoder
from src.models.image_encoder import build_image_encoder
from src.models.fusion import ConcatFusion, CrossFeatureFusion
from src.models.head import RegressionHead


class SMPFusionModel(nn.Module):
    """
    Main multimodal model for SMP popularity prediction.

    Supported branches:
    - text
    - metadata
    - image

    Recommended fusion for current multimodal stage:
    - CrossFeatureFusion for text + metadata + image
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
        use_meta: bool = True,
        use_image: bool = True,
        image_model_name: str = "openai/clip-vit-base-patch32",
        text_pooling: str = "clip",
        text_trainable: bool = False,
        image_pretrained: bool = True,   # kept for interface compatibility
        image_trainable: bool = False,
        fusion_type: str = "cross_feature",
        meta_branch_dim: int = 128,
    ) -> None:
        super().__init__()

        if not any([use_text, use_meta, use_image]):
            raise ValueError("At least one modality must be enabled.")

        self.use_text = use_text
        self.use_meta = use_meta
        self.use_image = use_image
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type.lower()

        self.text_encoder: Optional[nn.Module] = None
        self.meta_encoder: Optional[nn.Module] = None
        self.image_encoder: Optional[nn.Module] = None

        fusion_input_dims: Dict[str, int] = {}

        # -------------------------
        # text branch
        # -------------------------
        if self.use_text:
            self.text_encoder = TextEncoder(
                model_name=text_model_name,
                output_dim=hidden_dim,
                pooling=text_pooling,
                dropout=dropout,
                trainable=text_trainable,
            )
            fusion_input_dims["text"] = hidden_dim

        # -------------------------
        # metadata branch
        # -------------------------
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

        # -------------------------
        # image branch
        # -------------------------
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

        # -------------------------
        # fusion
        # -------------------------
        if self.fusion_type == "concat":
            self.fusion = ConcatFusion(
                input_dims=fusion_input_dims,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
                activation="relu",
                use_layernorm=True,
            )
        elif self.fusion_type == "cross_feature":
            if not (self.use_text and self.use_meta and self.use_image):
                raise ValueError(
                    "cross_feature fusion requires "
                    "use_text=True, use_meta=True, and use_image=True."
                )
            self.fusion = CrossFeatureFusion(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
                activation="relu",
                use_layernorm=True,
            )
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        # -------------------------
        # prediction head
        # -------------------------
        self.head = RegressionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation="relu",
            use_layernorm=True,
        )

    def _infer_batch_size(
        self,
        input_ids: Optional[torch.Tensor],
        meta_num: Optional[torch.Tensor],
        meta_cat: Optional[torch.Tensor],
        meta_bin: Optional[torch.Tensor],
        image_tensor: Optional[torch.Tensor],
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
        raise ValueError("Cannot infer batch size from all-None inputs.")

    def _infer_device(
        self,
        input_ids: Optional[torch.Tensor],
        meta_num: Optional[torch.Tensor],
        meta_cat: Optional[torch.Tensor],
        meta_bin: Optional[torch.Tensor],
        image_tensor: Optional[torch.Tensor],
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
        return torch.device("cpu")

    def extract_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        meta_num: Optional[torch.Tensor] = None,
        meta_cat: Optional[torch.Tensor] = None,
        meta_bin: Optional[torch.Tensor] = None,
        image_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        batch_size = self._infer_batch_size(
            input_ids=input_ids,
            meta_num=meta_num,
            meta_cat=meta_cat,
            meta_bin=meta_bin,
            image_tensor=image_tensor,
        )
        device = self._infer_device(
            input_ids=input_ids,
            meta_num=meta_num,
            meta_cat=meta_cat,
            meta_bin=meta_bin,
            image_tensor=image_tensor,
        )

        features: Dict[str, Optional[torch.Tensor]] = {
            "text": None,
            "meta": None,
            "image": None,
        }

        # -------------------------
        # text feature
        # -------------------------
        if self.use_text:
            if input_ids is None or attention_mask is None:
                raise ValueError(
                    "Text branch enabled but input_ids/attention_mask is missing."
                )
            text_feat = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # CLIP text embeddings live in cosine space
            # features["text"] = F.normalize(text_feat, dim=-1)
            features["text"] = self.text_encoder(input_ids, attention_mask)

        # -------------------------
        # metadata feature
        # -------------------------
        if self.use_meta:
            if meta_num is None and meta_cat is None and meta_bin is None:
                raise ValueError(
                    "Meta branch enabled but meta_num/meta_cat/meta_bin are all missing."
                )
            features["meta"] = self.meta_encoder(
                meta_num=meta_num,
                meta_cat=meta_cat,
                meta_bin=meta_bin,
            )

        # -------------------------
        # image feature
        # -------------------------
        if self.use_image:
            if image_tensor is None:
                raise ValueError("Image branch enabled but image_tensor is missing.")
            image_feat = self.image_encoder(image_tensor)
            # CLIP image embeddings should also be normalized to cosine space
            features["image"] = F.normalize(image_feat, dim=-1)
        else:
            # placeholder path for shape compatibility in concat-only experiments
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
        return_features: bool = False,
    ):
        features = self.extract_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            meta_num=meta_num,
            meta_cat=meta_cat,
            meta_bin=meta_bin,
            image_tensor=image_tensor,
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