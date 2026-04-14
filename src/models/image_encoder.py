from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection


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


class CLIPImageEncoder(nn.Module):
    """
    CLIP-based image encoder.

    Design goals:
    - Align image features with CLIP text space
    - Output a unified image representation [B, output_dim]
    - Support optional freezing for lower training cost
    - Keep the interface simple for multimodal fusion

    Recommended model_name:
        "openai/clip-vit-base-patch32"

    Important:
    - The input image tensor should already be processed with the matching
      CLIP image processor / normalization pipeline.
    - Expected input shape: [B, 3, H, W]
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        output_dim: Optional[int] = None,
        trainable: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.model = CLIPVisionModelWithProjection.from_pretrained(
            model_name,
            use_safetensors=True,
        )
        self.hidden_size = self.model.config.hidden_size
        self.projection_dim = self.model.config.projection_dim

        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

        self.output_dim = output_dim if output_dim is not None else self.projection_dim
        self.dropout = nn.Dropout(dropout)

        if self.output_dim != self.projection_dim:
            self.proj = nn.Sequential(
                nn.Linear(self.projection_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
            )
        else:
            self.proj = None

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tensor: [B, 3, H, W], already normalized for CLIP

        Returns:
            image_repr: [B, output_dim]
        """
        outputs = self.model(pixel_values=image_tensor)

        # [B, projection_dim]
        x = outputs.image_embeds
        x = self.dropout(x)

        if self.proj is not None:
            x = self.proj(x)

        return x


def build_image_encoder(
    use_image: bool,
    image_model_name: str = "openai/clip-vit-base-patch32",
    output_dim: int = 256,
    pretrained: bool = True,
    trainable: bool = False,
    dropout: float = 0.1,
    placeholder_when_disabled: bool = True,
):
    """
    Factory function for image encoder.

    Rules:
    - if use_image=True: build real CLIP vision encoder
    - if use_image=False and placeholder_when_disabled=True:
        build placeholder encoder
    - else: return None

    Notes:
    - `pretrained` is kept for interface compatibility with the old version.
      CLIPVisionModelWithProjection.from_pretrained(...) always loads pretrained weights.
    """
    if use_image:
        return CLIPImageEncoder(
            model_name=image_model_name,
            output_dim=output_dim,
            trainable=trainable,
            dropout=dropout,
        )

    if placeholder_when_disabled:
        return ImageEncoderPlaceholder(output_dim=output_dim)

    return None


if __name__ == "__main__":
    batch_size = 2
    height = 224
    width = 224

    model = CLIPImageEncoder(
        model_name="openai/clip-vit-base-patch32",
        output_dim=256,
        trainable=False,
        dropout=0.1,
    )

    image_tensor = torch.randn(batch_size, 3, height, width)
    y = model(image_tensor=image_tensor)
    print("output shape:", y.shape)