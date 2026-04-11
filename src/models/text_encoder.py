from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """
    Generic text encoder wrapper for HuggingFace transformer models.

    Design goals:
    - Easy to swap backbone model later
    - Unified output dimension via optional projection
    - Support common pooling methods
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        
        output_dim: Optional[int] = None,
        pooling: str = "cls",
        dropout: float = 0.1,
        trainable: bool = True,
    ) -> None:
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        self.pooling = pooling.lower()

        if self.pooling not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pooling: {pooling}")

        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

        self.output_dim = output_dim if output_dim is not None else self.hidden_size

        self.proj = None
        if self.output_dim != self.hidden_size:
            self.proj = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, self.output_dim),
                nn.LayerNorm(self.output_dim),
            )

    def _mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean pooling over valid tokens only.
        """
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        masked_hidden = last_hidden_state * mask
        summed = masked_hidden.sum(dim=1)            # [B, H]
        counts = mask.sum(dim=1).clamp(min=1e-6)     # [B, 1]
        return summed / counts

    def _pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0]
        if self.pooling == "mean":
            return self._mean_pool(last_hidden_state, attention_mask)
        raise RuntimeError(f"Unexpected pooling: {self.pooling}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        x = self._pool(outputs.last_hidden_state, attention_mask)

        if self.proj is not None:
            x = self.proj(x)

        return x