from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import CLIPTextModelWithProjection


class TextEncoder(nn.Module):
    """
    CLIP-based text encoder.

    Design goals:
    - Keep the interface close to the old text encoder
    - Output a unified text representation [B, output_dim]
    - Support optional freezing for lower training cost
    - Stay easy to plug into fusion with metadata

    Recommended model_name:
        "openai/clip-vit-base-patch32"

    Important:
    - If you switch to CLIP here, your dataset/tokenizer side should also use
      the matching CLIP tokenizer and usually max_length=77.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        output_dim: Optional[int] = None,
        pooling: str = "clip",
        dropout: float = 0.1,
        trainable: bool = False,
    ) -> None:
        super().__init__()

        self.model = CLIPTextModelWithProjection.from_pretrained(
            model_name,
            use_safetensors=True,
        )
        self.hidden_size = self.model.config.hidden_size
        self.projection_dim = self.model.config.projection_dim

        # Kept for interface compatibility with the old encoder.
        self.pooling = pooling.lower()
        if self.pooling not in {"clip"}:
            raise ValueError(
                f"Unsupported pooling for CLIP text encoder: {pooling}. "
                "Use pooling='clip'."
            )

        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

        self.output_dim = output_dim if output_dim is not None else self.projection_dim

        self.proj = None
        if self.output_dim != self.projection_dim:
            self.proj = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.projection_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # [B, projection_dim]
        x = outputs.text_embeds

        if self.proj is not None:
            x = self.proj(x)

        return x


if __name__ == "__main__":
    batch_size = 2
    seq_len = 16

    model = TextEncoder(
        model_name="openai/clip-vit-base-patch32",
        output_dim=256,
        trainable=False,
    )

    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    y = model(input_ids=input_ids, attention_mask=attention_mask)
    print("output shape:", y.shape)
