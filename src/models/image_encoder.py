# from __future__ import annotations

# from typing import Optional

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import CLIPVisionModelWithProjection


# class ImageEncoderPlaceholder(nn.Module):
#     def __init__(self, output_dim: int = 256) -> None:
#         super().__init__()
#         self.output_dim = output_dim

#     def forward(
#         self,
#         image_tensor: Optional[torch.Tensor] = None,
#         batch_size: Optional[int] = None,
#         device: Optional[torch.device] = None,
#     ) -> torch.Tensor:
#         if image_tensor is not None:
#             batch_size = image_tensor.size(0)
#             device = image_tensor.device
#         else:
#             if batch_size is None:
#                 raise ValueError("batch_size must be provided when image_tensor is None.")
#             if device is None:
#                 device = torch.device("cpu")

#         return torch.zeros(batch_size, self.output_dim, device=device)


# class CLIPImageEncoder(nn.Module):
#     def __init__(
#         self,
#         model_name: str = "openai/clip-vit-base-patch32",
#         output_dim: Optional[int] = None,
#         trainable: bool = False,
#         dropout: float = 0.1,
#     ) -> None:
#         super().__init__()

#         self.model = CLIPVisionModelWithProjection.from_pretrained(
#             model_name,
#             use_safetensors=True,
#         )

#         self.hidden_size = self.model.config.hidden_size
#         self.projection_dim = self.model.config.projection_dim
#         self.output_dim = output_dim if output_dim is not None else self.projection_dim

#         if not trainable:
#             for param in self.model.parameters():
#                 param.requires_grad = False

#         self.patch_attn = nn.Sequential(
#             nn.LayerNorm(self.hidden_size),
#             nn.Linear(self.hidden_size, self.hidden_size // 2),
#             nn.GELU(),
#             nn.Linear(self.hidden_size // 2, 1),
#         )

#         self.patch_proj = nn.Sequential(
#             nn.Linear(self.hidden_size, self.projection_dim),
#             nn.LayerNorm(self.projection_dim),
#         )

#         if self.output_dim != self.projection_dim:
#             self.global_proj = nn.Sequential(
#                 nn.Linear(self.projection_dim, self.output_dim),
#                 nn.LayerNorm(self.output_dim),
#             )
#             self.patch_out_proj = nn.Sequential(
#                 nn.Linear(self.projection_dim, self.output_dim),
#                 nn.LayerNorm(self.output_dim),
#             )
#         else:
#             self.global_proj = nn.Identity()
#             self.patch_out_proj = nn.Identity()

#         self.adapter = nn.Sequential(
#             nn.Linear(self.output_dim * 2, self.output_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.output_dim, self.output_dim),
#         )

#         self.adapter_scale = nn.Parameter(torch.tensor(0.1))
#         self.final_norm = nn.LayerNorm(self.output_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
#         outputs = self.model(pixel_values=image_tensor)

#         global_feat = outputs.image_embeds
#         patch_tokens = outputs.last_hidden_state[:, 1:, :]

#         attn_logits = self.patch_attn(patch_tokens)
#         attn_weights = torch.softmax(attn_logits, dim=1)
#         patch_feat = (patch_tokens * attn_weights).sum(dim=1)
#         patch_feat = self.patch_proj(patch_feat)

#         global_out = self.global_proj(global_feat)
#         patch_out = self.patch_out_proj(patch_feat)

#         adapter_input = torch.cat([global_out, patch_out], dim=-1)
#         adapter_delta = self.adapter(adapter_input)

#         x = global_out + self.adapter_scale * adapter_delta
#         x = self.final_norm(x)
#         x = self.dropout(x)

#         return x


# def build_image_encoder(
#     use_image: bool,
#     image_model_name: str = "openai/clip-vit-base-patch32",
#     output_dim: int = 256,
#     pretrained: bool = True,
#     trainable: bool = False,
#     dropout: float = 0.1,
#     placeholder_when_disabled: bool = True,
# ):
#     if use_image:
#         return CLIPImageEncoder(
#             model_name=image_model_name,
#             output_dim=output_dim,
#             trainable=trainable,
#             dropout=dropout,
#         )

#     if placeholder_when_disabled:
#         return ImageEncoderPlaceholder(output_dim=output_dim)

#     return None


# if __name__ == "__main__":
#     model = CLIPImageEncoder(
#         model_name="openai/clip-vit-base-patch32",
#         output_dim=256,
#         trainable=False,
#         dropout=0.1,
#     )

#     image_tensor = torch.randn(2, 3, 224, 224)
#     y = model(image_tensor)
#     print("output shape:", y.shape)

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection


class ImageEncoderPlaceholder(nn.Module):
    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()
        self.output_dim = output_dim

    def forward(
        self,
        image_tensor: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        return_raw_clip: bool = False,
    ):
        if image_tensor is not None:
            batch_size = image_tensor.size(0)
            device = image_tensor.device
        else:
            if batch_size is None:
                raise ValueError("batch_size must be provided when image_tensor is None.")
            if device is None:
                device = torch.device("cpu")

        x = torch.zeros(batch_size, self.output_dim, device=device)

        if return_raw_clip:
            return x, None

        return x


class CLIPImageEncoder(nn.Module):
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
        self.output_dim = output_dim if output_dim is not None else self.projection_dim
        self.trainable = bool(trainable)

        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

        self.patch_attn = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 1),
        )

        self.patch_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
        )

        if self.output_dim != self.projection_dim:
            self.global_proj = nn.Sequential(
                nn.Linear(self.projection_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
            )
            self.patch_out_proj = nn.Sequential(
                nn.Linear(self.projection_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
            )
        else:
            self.global_proj = nn.Identity()
            self.patch_out_proj = nn.Identity()

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
        image_tensor: torch.Tensor,
        return_raw_clip: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.trainable:
            outputs = self.model(pixel_values=image_tensor)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(pixel_values=image_tensor)

        raw_clip_feat = outputs.image_embeds  # [B, projection_dim]

        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # remove CLS
        attn_logits = self.patch_attn(patch_tokens)
        attn_weights = torch.softmax(attn_logits, dim=1)

        patch_feat = (patch_tokens * attn_weights).sum(dim=1)
        patch_feat = self.patch_proj(patch_feat)

        global_out = self.global_proj(raw_clip_feat)
        patch_out = self.patch_out_proj(patch_feat)

        adapter_input = torch.cat([global_out, patch_out], dim=-1)
        adapter_delta = self.adapter(adapter_input)

        x = global_out + self.adapter_scale * adapter_delta
        x = self.final_norm(x)
        x = self.dropout(x)

        if return_raw_clip:
            return x, raw_clip_feat

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
    model = CLIPImageEncoder(
        model_name="openai/clip-vit-base-patch32",
        output_dim=256,
        trainable=False,
        dropout=0.1,
    )

    image_tensor = torch.randn(2, 3, 224, 224)
    y, raw = model(image_tensor, return_raw_clip=True)
    print("output shape:", y.shape)
    print("raw shape:", raw.shape)
