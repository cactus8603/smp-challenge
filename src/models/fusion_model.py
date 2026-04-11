from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.text_encoder import TextEncoder
from src.models.meta_encoder import MetaEncoder
from src.models.image_encoder import build_image_encoder
from src.models.fusion import ConcatFusion
from src.models.head import RegressionHead


class SMPFusionModel(nn.Module):
    """
    Main multimodal model for SMP popularity prediction.

    Current supported branches:
    - text
    - metadata
    - image (placeholder-ready / timm-ready)

    Design goals:
    - modular
    - easy to expand
    - unified hidden dimension across modalities
    """

    def __init__(
        self,
        text_model_name: str = "distilbert-base-uncased",
        meta_dim: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_text: bool = True,
        use_meta: bool = True,
        use_image: bool = False,
        image_model_name: str = "resnet50",
        text_pooling: str = "cls",
        text_trainable: bool = True,
        image_pretrained: bool = True,
        image_trainable: bool = True,
    ) -> None:
        super().__init__()

        if not any([use_text, use_meta, use_image]):
            raise ValueError("At least one modality must be enabled.")

        self.use_text = use_text
        self.use_meta = use_meta
        self.use_image = use_image
        self.hidden_dim = hidden_dim

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
            if meta_dim <= 0:
                raise ValueError(f"meta_dim must be > 0 when use_meta=True, got {meta_dim}")
            self.meta_encoder = MetaEncoder(
                input_dim=meta_dim,
                output_dim=hidden_dim,
                hidden_dims=[128, hidden_dim],
                dropout=dropout,
                activation="relu",
                use_layernorm=True,
            )
            fusion_input_dims["meta"] = hidden_dim

        # -------------------------
        # image branch
        # -------------------------
        # Even when use_image=False, we still build a placeholder encoder
        # so the interface remains stable and future extension is easy.
        self.image_encoder = build_image_encoder(
            use_image=use_image,
            image_model_name=image_model_name,
            output_dim=hidden_dim,
            pretrained=image_pretrained,
            trainable=image_trainable,
            dropout=dropout,
            placeholder_when_disabled=True,
        )
        fusion_input_dims["image"] = hidden_dim

        # -------------------------
        # fusion
        # -------------------------
        self.fusion = ConcatFusion(
            input_dims=fusion_input_dims,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
            activation="relu",
            use_layernorm=True,
        )

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
        meta_features: Optional[torch.Tensor],
        image_tensor: Optional[torch.Tensor],
    ) -> int:
        if input_ids is not None:
            return input_ids.size(0)
        if meta_features is not None:
            return meta_features.size(0)
        if image_tensor is not None:
            return image_tensor.size(0)
        raise ValueError("Cannot infer batch size from all-None inputs.")

    def _infer_device(
        self,
        input_ids: Optional[torch.Tensor],
        meta_features: Optional[torch.Tensor],
        image_tensor: Optional[torch.Tensor],
    ) -> torch.device:
        if input_ids is not None:
            return input_ids.device
        if meta_features is not None:
            return meta_features.device
        if image_tensor is not None:
            return image_tensor.device
        return torch.device("cpu")

    def extract_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        meta_features: Optional[torch.Tensor] = None,
        image_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Return per-modality encoded features before fusion.
        Useful for debugging / ablation later.
        """
        batch_size = self._infer_batch_size(input_ids, meta_features, image_tensor)
        device = self._infer_device(input_ids, meta_features, image_tensor)

        features: Dict[str, Optional[torch.Tensor]] = {
            "text": None,
            "meta": None,
            "image": None,
        }

        # text
        if self.use_text:
            if input_ids is None or attention_mask is None:
                raise ValueError("Text branch enabled but input_ids/attention_mask is missing.")
            features["text"] = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # meta
        if self.use_meta:
            if meta_features is None:
                raise ValueError("Meta branch enabled but meta_features is missing.")
            features["meta"] = self.meta_encoder(meta_features)

        # image
        if self.use_image:
            if image_tensor is None:
                raise ValueError("Image branch enabled but image_tensor is missing.")
            features["image"] = self.image_encoder(image_tensor)
        else:
            # placeholder zero feature to keep interface stable
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
        meta_features: Optional[torch.Tensor] = None,
        image_tensor: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        features = self.extract_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            meta_features=meta_features,
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