# from __future__ import annotations

# from typing import Optional

# import torch
# import torch.nn as nn
# from transformers import CLIPTextModelWithProjection


# class TextEncoder(nn.Module):
#     """
#     CLIP-based text encoder.

#     Design goals:
#     - Keep the interface close to the old text encoder
#     - Output a unified text representation [B, output_dim]
#     - Support optional freezing for lower training cost
#     - Stay easy to plug into fusion with metadata

#     Recommended model_name:
#         "openai/clip-vit-base-patch32"

#     Important:
#     - If you switch to CLIP here, your dataset/tokenizer side should also use
#       the matching CLIP tokenizer and usually max_length=77.
#     """

#     def __init__(
#         self,
#         model_name: str = "openai/clip-vit-base-patch32",
#         output_dim: Optional[int] = None,
#         pooling: str = "clip",
#         dropout: float = 0.1,
#         trainable: bool = False,
#     ) -> None:
#         super().__init__()

#         self.model = CLIPTextModelWithProjection.from_pretrained(
#             model_name,
#             use_safetensors=True,
#         )
#         self.hidden_size = self.model.config.hidden_size
#         self.projection_dim = self.model.config.projection_dim

#         # Kept for interface compatibility with the old encoder.
#         self.pooling = pooling.lower()
#         if self.pooling not in {"clip"}:
#             raise ValueError(
#                 f"Unsupported pooling for CLIP text encoder: {pooling}. "
#                 "Use pooling='clip'."
#             )

#         if not trainable:
#             for param in self.model.parameters():
#                 param.requires_grad = False

#         self.output_dim = output_dim if output_dim is not None else self.projection_dim

#         # Dropout 永遠存在，避免 output_dim == projection_dim 時沒有正則化
#         self.dropout = nn.Dropout(dropout)

#         if self.output_dim != self.projection_dim:
#             self.proj = nn.Sequential(
#                 nn.Linear(self.projection_dim, self.output_dim),
#                 nn.LayerNorm(self.output_dim),
#             )
#         else:
#             self.proj = None

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: torch.Tensor,
#     ) -> torch.Tensor:
#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )

#         # [B, projection_dim]
#         x = outputs.text_embeds
#         x = self.dropout(x)  # 無論是否有 proj 都做 dropout

#         if self.proj is not None:
#             x = self.proj(x)

#         return x


# if __name__ == "__main__":
#     batch_size = 2
#     seq_len = 16

#     model = TextEncoder(
#         model_name="openai/clip-vit-base-patch32",
#         output_dim=256,
#         trainable=False,
#     )

#     input_ids = torch.randint(0, 100, (batch_size, seq_len))
#     attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

#     y = model(input_ids=input_ids, attention_mask=attention_mask)
#     print("output shape:", y.shape)


from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import CLIPTextModelWithProjection


class TextEncoder(nn.Module):
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
        self.output_dim = output_dim if output_dim is not None else self.projection_dim

        self.pooling = pooling.lower()
        if self.pooling not in {"clip"}:
            raise ValueError(f"Unsupported pooling: {pooling}. Use pooling='clip'.")

        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

        self.token_attn = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 1),
        )

        self.token_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
        )

        if self.output_dim != self.projection_dim:
            self.global_proj = nn.Sequential(
                nn.Linear(self.projection_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
            )
            self.token_out_proj = nn.Sequential(
                nn.Linear(self.projection_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
            )
        else:
            self.global_proj = nn.Identity()
            self.token_out_proj = nn.Identity()

        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim * 2, self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim, self.output_dim),
        )

        self.adapter_scale = nn.Parameter(torch.tensor(0.02))
        self.final_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_raw_clip: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        raw_clip_feat = outputs.text_embeds  # [B, projection_dim]

        token_states = outputs.last_hidden_state  # [B, L, hidden_size]
        token_mask = attention_mask.bool().unsqueeze(-1)  # [B, L, 1]

        attn_logits = self.token_attn(token_states)  # [B, L, 1]
        attn_logits = attn_logits.masked_fill(~token_mask, -1e4)
        attn_weights = torch.softmax(attn_logits, dim=1)

        token_feat = (token_states * attn_weights).sum(dim=1)
        token_feat = self.token_proj(token_feat)

        global_out = self.global_proj(raw_clip_feat)
        token_out = self.token_out_proj(token_feat)

        adapter_input = torch.cat([global_out, token_out], dim=-1)
        adapter_delta = self.adapter(adapter_input)

        x = global_out + self.adapter_scale * adapter_delta
        x = self.final_norm(x)
        x = self.dropout(x)

        if return_raw_clip:
            return x, raw_clip_feat

        return x


if __name__ == "__main__":
    model = TextEncoder(
        model_name="openai/clip-vit-base-patch32",
        output_dim=256,
        trainable=False,
    )

    input_ids = torch.randint(0, 100, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)

    y, raw = model(input_ids, attention_mask, return_raw_clip=True)
    print("output shape:", y.shape)
    print("raw shape:", raw.shape)