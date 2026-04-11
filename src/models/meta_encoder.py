
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    dropout: float = 0.1,
    activation: str = "relu",
    use_layernorm: bool = True,
) -> nn.Sequential:
    """
    Build a configurable MLP block.

    Example:
        input_dim=16, hidden_dims=[128, 256]
        -> Linear(16,128) -> Act -> LN -> Dropout
        -> Linear(128,256) -> Act -> LN -> Dropout
    """
    if activation.lower() == "relu":
        act_layer = nn.ReLU
    elif activation.lower() == "gelu":
        act_layer = nn.GELU
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    if not hidden_dims:
        raise ValueError("build_mlp requires at least one hidden_dim.")

    layers = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))  # Pre-LN: before activation
        layers.append(act_layer())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim

    return nn.Sequential(*layers)


class CategoricalEmbeddingEncoder(nn.Module):
    """
    Encode multiple categorical fields with separate embeddings.

    Args:
        cardinalities:
            Number of categories for each categorical field.
            Each field is expected to already be integer-encoded and include an UNK id.
        embedding_dim:
            If fixed_embedding_dim is None, each field uses:
                min(max_embedding_dim, max(min_embedding_dim, ceil(sqrt(cardinality))))
        output_dim:
            Project concatenated embeddings to this dimension.
    """

    def __init__(
        self,
        cardinalities: Sequence[int],
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        fixed_embedding_dim: Optional[int] = None,
        min_embedding_dim: int = 4,
        max_embedding_dim: int = 32,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        self.cardinalities = list(cardinalities)
        self.num_fields = len(self.cardinalities)

        if self.num_fields == 0:
            self.embeddings = nn.ModuleList()
            self.total_embedding_dim = 0
            self.encoder = None
            self.output_dim = 0
            return

        embedding_dims: List[int] = []
        for card in self.cardinalities:
            if card <= 0:
                raise ValueError(f"Invalid categorical cardinality: {card}")
            if fixed_embedding_dim is not None:
                emb_dim = fixed_embedding_dim
            else:
                emb_dim = int(card ** 0.5)
                emb_dim = max(min_embedding_dim, emb_dim)
                emb_dim = min(max_embedding_dim, emb_dim)
            embedding_dims.append(emb_dim)

        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, emb_dim) for cardinality, emb_dim in zip(self.cardinalities, embedding_dims)]
        )
        self.total_embedding_dim = sum(embedding_dims)

        if hidden_dims is None:
            hidden_dims = [max(output_dim, 64), output_dim]

        self.encoder = build_mlp(
            input_dim=self.total_embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            use_layernorm=use_layernorm,
        )
        self.output_dim = hidden_dims[-1]

        if self.output_dim != output_dim:
            self.proj = nn.Linear(self.output_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.proj = None

    def forward(self, meta_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            meta_cat: [B, D_cat], integer ids for each categorical field

        Returns:
            cat_repr: [B, output_dim]
        """
        if self.num_fields == 0:
            batch_size = meta_cat.size(0)
            return meta_cat.new_zeros((batch_size, 0), dtype=torch.float32)

        if meta_cat.dim() != 2 or meta_cat.size(1) != self.num_fields:
            raise ValueError(
                f"Expected meta_cat shape [B, {self.num_fields}], got {tuple(meta_cat.shape)}"
            )

        embedded_fields = []
        for i, emb in enumerate(self.embeddings):
            field_ids = meta_cat[:, i].long()
            embedded_fields.append(emb(field_ids))

        x = torch.cat(embedded_fields, dim=-1)
        x = self.encoder(x)

        if self.proj is not None:
            x = self.proj(x)

        return x


class MetaEncoder(nn.Module):
    """
    Heterogeneous metadata encoder.

    Inputs:
        - meta_num: [B, D_num]  numeric metadata
        - meta_cat: [B, D_cat]  categorical metadata ids
        - meta_bin: [B, D_bin]  binary metadata

    Design:
        1. Numeric branch: MLP
        2. Categorical branch: per-field embedding + MLP
        3. Binary branch: small MLP
        4. Gated fusion over the three branch representations
        5. Final projection MLP -> meta representation

    This version is intended to be a stronger baseline than a single dense MLP,
    while staying simple enough to train and debug.
    """

    def __init__(
        self,
        num_input_dim: int,
        cat_cardinalities: Optional[Sequence[int]] = None,
        bin_input_dim: int = 0,
        output_dim: int = 256,
        branch_dim: int = 128,
        num_hidden_dims: Optional[List[int]] = None,
        cat_hidden_dims: Optional[List[int]] = None,
        bin_hidden_dims: Optional[List[int]] = None,
        fusion_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layernorm: bool = True,
        fixed_cat_embedding_dim: Optional[int] = None,
        min_cat_embedding_dim: int = 4,
        max_cat_embedding_dim: int = 32,
    ) -> None:
        super().__init__()

        self.num_input_dim = num_input_dim
        self.cat_cardinalities = list(cat_cardinalities or [])
        self.cat_input_dim = len(self.cat_cardinalities)
        self.bin_input_dim = bin_input_dim
        self.output_dim = output_dim
        self.branch_dim = branch_dim

        if self.num_input_dim < 0 or self.bin_input_dim < 0:
            raise ValueError("Input dimensions must be non-negative.")
        if self.num_input_dim == 0 and self.cat_input_dim == 0 and self.bin_input_dim == 0:
            raise ValueError("At least one metadata branch must be non-empty.")

        # -------------------------
        # Numeric branch
        # -------------------------
        self.use_num = self.num_input_dim > 0
        if self.use_num:
            if num_hidden_dims is None:
                num_hidden_dims = [max(branch_dim, 128), branch_dim]
            self.num_encoder = build_mlp(
                input_dim=self.num_input_dim,
                hidden_dims=num_hidden_dims,
                dropout=dropout,
                activation=activation,
                use_layernorm=use_layernorm,
            )
            self.num_out_dim = num_hidden_dims[-1]
            self.num_proj = None
            if self.num_out_dim != branch_dim:
                self.num_proj = nn.Linear(self.num_out_dim, branch_dim)
                self.num_out_dim = branch_dim
        else:
            self.num_encoder = None
            self.num_proj = None
            self.num_out_dim = 0

        # -------------------------
        # Categorical branch
        # -------------------------
        self.use_cat = self.cat_input_dim > 0
        if self.use_cat:
            self.cat_encoder = CategoricalEmbeddingEncoder(
                cardinalities=self.cat_cardinalities,
                output_dim=branch_dim,
                hidden_dims=cat_hidden_dims,
                fixed_embedding_dim=fixed_cat_embedding_dim,
                min_embedding_dim=min_cat_embedding_dim,
                max_embedding_dim=max_cat_embedding_dim,
                dropout=dropout,
                activation=activation,
                use_layernorm=use_layernorm,
            )
            self.cat_out_dim = self.cat_encoder.output_dim
        else:
            self.cat_encoder = None
            self.cat_out_dim = 0

        # -------------------------
        # Binary branch
        # -------------------------
        self.use_bin = self.bin_input_dim > 0
        if self.use_bin:
            if bin_hidden_dims is None:
                hidden = max(branch_dim // 2, 32)
                bin_hidden_dims = [hidden, branch_dim]
            self.bin_encoder = build_mlp(
                input_dim=self.bin_input_dim,
                hidden_dims=bin_hidden_dims,
                dropout=dropout,
                activation=activation,
                use_layernorm=use_layernorm,
            )
            self.bin_out_dim = bin_hidden_dims[-1]
            self.bin_proj = None
            if self.bin_out_dim != branch_dim:
                self.bin_proj = nn.Linear(self.bin_out_dim, branch_dim)
                self.bin_out_dim = branch_dim
        else:
            self.bin_encoder = None
            self.bin_proj = None
            self.bin_out_dim = 0

        # -------------------------
        # Gated branch weighting (只在多 branch 時才有意義)
        # -------------------------
        active_branch_count = int(self.use_num) + int(self.use_cat) + int(self.use_bin)
        self.active_branch_count = active_branch_count

        if active_branch_count > 1:
            self.gate_mlp = nn.Sequential(
                nn.Linear(branch_dim * active_branch_count, branch_dim),
                nn.ReLU(),
                nn.Linear(branch_dim, active_branch_count),
            )
        else:
            self.gate_mlp = None

        # -------------------------
        # Final fusion MLP
        # -------------------------
        if fusion_hidden_dims is None:
            # 避免 output_dim == branch_dim 時產生兩層相同維度 + 多餘 projection
            if output_dim > branch_dim:
                fusion_hidden_dims = [output_dim]
            else:
                fusion_hidden_dims = [max(branch_dim, 256), output_dim]

        self.fusion_mlp = build_mlp(
            input_dim=branch_dim,
            hidden_dims=fusion_hidden_dims,
            dropout=dropout,
            activation=activation,
            use_layernorm=use_layernorm,
        )
        fusion_out_dim = fusion_hidden_dims[-1]
        self.fusion_proj = nn.Linear(fusion_out_dim, output_dim) if fusion_out_dim != output_dim else None

    def _encode_num(self, meta_num: torch.Tensor) -> torch.Tensor:
        x = self.num_encoder(meta_num)
        if self.num_proj is not None:
            x = self.num_proj(x)
        return x

    def _encode_cat(self, meta_cat: torch.Tensor) -> torch.Tensor:
        return self.cat_encoder(meta_cat)

    def _encode_bin(self, meta_bin: torch.Tensor) -> torch.Tensor:
        x = self.bin_encoder(meta_bin)
        if self.bin_proj is not None:
            x = self.bin_proj(x)
        return x

    def forward(
        self,
        meta_num: Optional[torch.Tensor] = None,
        meta_cat: Optional[torch.Tensor] = None,
        meta_bin: Optional[torch.Tensor] = None,
        return_gate_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            meta_num: [B, D_num]
            meta_cat: [B, D_cat]
            meta_bin: [B, D_bin]
            return_gate_weights:
                If True, also return branch gate weights of shape [B, num_active_branches].

        Returns:
            meta_repr: [B, output_dim]
            optionally gate_weights: [B, num_active_branches]
        """
        branch_reprs = []

        if self.use_num:
            if meta_num is None:
                raise ValueError("meta_num is required but num_input_dim > 0")
            branch_reprs.append(self._encode_num(meta_num.float()))

        if self.use_cat:
            if meta_cat is None:
                raise ValueError("meta_cat is required but categorical branch is enabled")
            branch_reprs.append(self._encode_cat(meta_cat.long()))

        if self.use_bin:
            if meta_bin is None:
                raise ValueError("meta_bin is required but bin_input_dim > 0")
            branch_reprs.append(self._encode_bin(meta_bin.float()))

        if len(branch_reprs) == 1:
            fused = branch_reprs[0]
            gate_weights = fused.new_ones((fused.size(0), 1))
        else:
            stacked = torch.stack(branch_reprs, dim=1)            # [B, N, branch_dim]
            gate_input = torch.cat(branch_reprs, dim=-1)          # [B, N * branch_dim]
            gate_logits = self.gate_mlp(gate_input)               # [B, N]
            gate_weights = F.softmax(gate_logits, dim=-1)         # [B, N]
            fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)  # [B, branch_dim]

        out = self.fusion_mlp(fused)
        if self.fusion_proj is not None:
            out = self.fusion_proj(out)

        if return_gate_weights:
            return out, gate_weights
        return out


if __name__ == "__main__":
    # Simple smoke test
    batch_size = 4
    num_input_dim = 12
    cat_cardinalities = [10, 20, 5]
    bin_input_dim = 6

    model = MetaEncoder(
        num_input_dim=num_input_dim,
        cat_cardinalities=cat_cardinalities,
        bin_input_dim=bin_input_dim,
        output_dim=256,
        branch_dim=128,
        dropout=0.1,
    )

    meta_num = torch.randn(batch_size, num_input_dim)
    meta_cat = torch.randint(0, 5, (batch_size, len(cat_cardinalities)))
    meta_bin = torch.randint(0, 2, (batch_size, bin_input_dim)).float()

    meta_repr, gate_weights = model(
        meta_num=meta_num,
        meta_cat=meta_cat,
        meta_bin=meta_bin,
        return_gate_weights=True,
    )

    print("meta_repr:", meta_repr.shape)       # [4, 256]
    print("gate_weights:", gate_weights.shape) # [4, 3]
