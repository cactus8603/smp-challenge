from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ImageEncoderPlaceholder(nn.Module):
    """
    Placeholder image encoder.

    Use this when the image pipeline is not ready yet.
    It returns a zero tensor with the target output_dim so the rest of the
    multimodal architecture can already be built and tested.
    """

    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()
        self.output_dim = output_dim

    def forward(
        self,
        image_tensor: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if image_tensor is not None:
            batch_size = image_tensor.size(0)
            device = image_tensor.device
        else:
            if batch_size is None:
                raise ValueError("batch_size must be provided when image_tensor is None.")
            if device is None:
                device = torch.device("cpu")

        return torch.zeros(batch_size, self.output_dim, device=device)


class TimmImageEncoder(nn.Module):
    """
    Image encoder based on timm backbone.

    Expected usage later:
        encoder = TimmImageEncoder(
            model_name="resnet50",
            output_dim=256,
            pretrained=True,
            trainable=True,
        )

    Notes:
    - This class requires timm to be installed.
    - The backbone is created with num_classes=0 so it returns features.
    - A projection layer maps backbone output to a unified output_dim.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        output_dim: int = 256,
        pretrained: bool = True,
        trainable: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required for TimmImageEncoder. Please install it with: pip install timm"
            ) from e

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        if hasattr(self.backbone, "num_features"):
            backbone_dim = self.backbone.num_features
        else:
            raise ValueError(f"Cannot infer num_features from timm model: {model_name}")

        if not trainable:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.output_dim = output_dim
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(image_tensor)
        feats = self.proj(feats)
        return feats


def build_image_encoder(
    use_image: bool,
    image_model_name: str = "resnet50",
    output_dim: int = 256,
    pretrained: bool = True,
    trainable: bool = True,
    dropout: float = 0.1,
    placeholder_when_disabled: bool = True,
):
    """
    Factory function for image encoder.

    Rules:
    - if use_image=True: build real timm encoder
    - if use_image=False and placeholder_when_disabled=True:
        build placeholder encoder
    - else: return None
    """
    if use_image:
        return TimmImageEncoder(
            model_name=image_model_name,
            output_dim=output_dim,
            pretrained=pretrained,
            trainable=trainable,
            dropout=dropout,
        )

    if placeholder_when_disabled:
        return ImageEncoderPlaceholder(output_dim=output_dim)

    return None