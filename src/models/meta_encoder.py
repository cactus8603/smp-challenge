
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


class VectorCompressor(nn.Module):
    """
    Compress a high-dimensional pre-computed vector (e.g. user_description 400-dim)
    into branch_dim via a small bottleneck MLP.

    Architecture:
        Linear(input_dim, bottleneck_dim) -> LayerNorm -> GELU -> Dropout
        -> Linear(bottleneck_dim, output_dim) -> LayerNorm -> GELU -> Dropout

    Missing vectors are represented as all-zero input vectors by the dataset.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        - meta_num:      [B, D_num]   numeric metadata
        - meta_cat:      [B, D_cat]   categorical metadata ids
        - meta_bin:      [B, D_bin]   binary metadata
        - user_desc:     [B, 400]     user_description vector (zeros if unavailable)
        - loc_desc:      [B, 400]     location_description vector (zeros if unavailable)

    Design:
        1. Numeric branch:              MLP → branch_dim
        2. Categorical branch:          Embedding + MLP → branch_dim
        3. Binary branch:               MLP → branch_dim
        4. Gated fusion over structured metadata branches
        5. Final MLP → output_dim

    Note:
        user_description/location_description are supported for backward compatibility,
        but the current recommended architecture keeps them outside MetaEncoder as
        auxiliary residual branches in SMPFusionModel.
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
        # ── vector description branches ──────────────────────
        user_desc_dim: int = 768,       # sentence-transformers all-mpnet-base-v2 output dim
        loc_desc_dim: int = 400,
        desc_bottleneck_dim: int = 64,
        use_user_desc: bool = False,
        use_loc_desc: bool = False,     # 關閉：location_description 官方資料全是 None
        # ── feature gating ────────────────────────────────────
        feature_gate_config: Optional[List[dict]] = None,
    ) -> None:
        super().__init__()

        self.num_input_dim   = num_input_dim
        self.cat_cardinalities = list(cat_cardinalities or [])
        self.cat_input_dim   = len(self.cat_cardinalities)
        self.bin_input_dim   = bin_input_dim

        # feature gate groups: each group gates a set of num/cat dims by one bin flag
        self.feature_gate_config: List[dict] = feature_gate_config or []
        self.output_dim      = output_dim
        self.branch_dim      = branch_dim
        self.use_user_desc   = use_user_desc and user_desc_dim > 0
        self.use_loc_desc    = use_loc_desc  and loc_desc_dim  > 0

        if self.num_input_dim < 0 or self.bin_input_dim < 0:
            raise ValueError("Input dimensions must be non-negative.")
        if (self.num_input_dim == 0 and self.cat_input_dim == 0
                and self.bin_input_dim == 0
                and not self.use_user_desc and not self.use_loc_desc):
            raise ValueError("At least one metadata branch must be non-empty.")

        # ── Numeric branch ────────────────────────────────────
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
            self.num_encoder = self.num_proj = None
            self.num_out_dim = 0

        # ── Categorical branch ────────────────────────────────
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

        # ── Binary branch ─────────────────────────────────────
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
            self.bin_encoder = self.bin_proj = None
            self.bin_out_dim = 0

        # ── user_description branch ───────────────────────────
        if self.use_user_desc:
            self.user_desc_encoder = VectorCompressor(
                input_dim=user_desc_dim,
                bottleneck_dim=desc_bottleneck_dim,
                output_dim=branch_dim,
                dropout=dropout,
            )
        else:
            self.user_desc_encoder = None

        # ── location_description branch ───────────────────────
        if self.use_loc_desc:
            self.loc_desc_encoder = VectorCompressor(
                input_dim=loc_desc_dim,
                bottleneck_dim=desc_bottleneck_dim,
                output_dim=branch_dim,
                dropout=dropout,
            )
        else:
            self.loc_desc_encoder = None

        # ── Gated fusion ──────────────────────────────────────
        active_branch_count = (
            int(self.use_num)
            + int(self.use_cat)
            + int(self.use_bin)
            + int(self.use_user_desc)
            + int(self.use_loc_desc)
        )
        self.active_branch_count = active_branch_count

        if active_branch_count > 1:
            self.gate_mlp = nn.Sequential(
                nn.Linear(branch_dim * active_branch_count, branch_dim),
                nn.ReLU(),
                nn.Linear(branch_dim, active_branch_count),
            )
        else:
            self.gate_mlp = None

        # ── Final fusion MLP ──────────────────────────────────
        if fusion_hidden_dims is None:
            fusion_hidden_dims = [output_dim]
        self.fusion_mlp = build_mlp(
            input_dim=branch_dim,
            hidden_dims=fusion_hidden_dims,
            dropout=dropout,
            activation=activation,
            use_layernorm=use_layernorm,
        )
        fusion_out_dim = fusion_hidden_dims[-1]
        self.fusion_proj = (
            nn.Linear(fusion_out_dim, output_dim)
            if fusion_out_dim != output_dim else None
        )

    # ── helpers ───────────────────────────────────────────────
    def _encode_num(self, x: torch.Tensor) -> torch.Tensor:
        x = self.num_encoder(x)
        if self.num_proj is not None:
            x = self.num_proj(x)
        return x

    def _encode_cat(self, x: torch.Tensor) -> torch.Tensor:
        return self.cat_encoder(x)

    def _encode_bin(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bin_encoder(x)
        if self.bin_proj is not None:
            x = self.bin_proj(x)
        return x

    def _get_batch_size(self, *tensors) -> int:
        for t in tensors:
            if t is not None:
                return t.size(0)
        raise ValueError("All inputs are None.")

    def _zero_vec(self, input_dim: int, batch_size: int) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.zeros(batch_size, input_dim, device=device)

    # ── forward ───────────────────────────────────────────────
    def forward(
        self,
        meta_num: Optional[torch.Tensor] = None,
        meta_cat: Optional[torch.Tensor] = None,
        meta_bin: Optional[torch.Tensor] = None,
        user_desc: Optional[torch.Tensor] = None,
        loc_desc: Optional[torch.Tensor] = None,
        return_gate_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            meta_num:  [B, D_num]
            meta_cat:  [B, D_cat]
            meta_bin:  [B, D_bin]
            user_desc: [B, 768]  float32; zeros for users without description
            loc_desc:  [B, D]    float32; zeros when unavailable
            return_gate_weights: also return [B, N_branches] gate weights

        Returns:
            meta_repr [B, output_dim], optionally gate_weights [B, N_branches]
        """
        branch_reprs: List[torch.Tensor] = []
        B = self._get_batch_size(meta_num, meta_cat, meta_bin, user_desc, loc_desc)

        # ── feature gating ────────────────────────────────────────────────────
        # For each gate group: flag=0 → zero the gated num dims / force cat to UNK(0)
        # Applied before branch MLPs so encoders never see spurious imputed values.
        if self.feature_gate_config and meta_bin is not None:
            num_cloned = False
            cat_cloned = False
            for group in self.feature_gate_config:
                flag = meta_bin[:, group["flag_bin_idx"]].float()  # [B]

                if meta_num is not None and group["num_indices"]:
                    if not num_cloned:
                        meta_num = meta_num.clone()
                        num_cloned = True
                    for idx in group["num_indices"]:
                        meta_num[:, idx] = meta_num[:, idx] * flag

                if meta_cat is not None and group["cat_indices"]:
                    if not cat_cloned:
                        meta_cat = meta_cat.clone()
                        cat_cloned = True
                    flag_long = flag.long()
                    for idx in group["cat_indices"]:
                        meta_cat[:, idx] = meta_cat[:, idx] * flag_long
        # ─────────────────────────────────────────────────────────────────────

        if self.use_num:
            if meta_num is None:
                raise ValueError("meta_num required but not provided")
            branch_reprs.append(self._encode_num(meta_num.float()))

        if self.use_cat:
            if meta_cat is None:
                raise ValueError("meta_cat required but not provided")
            branch_reprs.append(self._encode_cat(meta_cat.long()))

        if self.use_bin:
            if meta_bin is None:
                raise ValueError("meta_bin required but not provided")
            branch_reprs.append(self._encode_bin(meta_bin.float()))

        # ── user_description ──────────────────────────────────
        # Zero vector = no description; VectorCompressor handles it naturally
        if self.use_user_desc:
            if user_desc is None:
                user_desc = self._zero_vec(self.user_desc_encoder.input_dim, B)
            branch_reprs.append(self.user_desc_encoder(user_desc.float()))

        # ── location_description ──────────────────────────────
        if self.use_loc_desc:
            if loc_desc is None:
                loc_desc = self._zero_vec(self.loc_desc_encoder.input_dim, B)
            branch_reprs.append(self.loc_desc_encoder(loc_desc.float()))

        # ── gated fusion ──────────────────────────────────────
        if len(branch_reprs) == 1:
            fused = branch_reprs[0]
            gate_weights = fused.new_ones((B, 1))
        else:
            stacked     = torch.stack(branch_reprs, dim=1)             # [B, N, D]
            gate_input  = torch.cat(branch_reprs, dim=-1)              # [B, N*D]
            gate_logits = self.gate_mlp(gate_input)                    # [B, N]
            gate_weights = F.softmax(gate_logits, dim=-1)              # [B, N]
            fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

        out = self.fusion_mlp(fused)
        if self.fusion_proj is not None:
            out = self.fusion_proj(out)

        if return_gate_weights:
            return out, gate_weights
        return out


if __name__ == "__main__":
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
        use_user_desc=True,
        use_loc_desc=False,
        user_desc_dim=768,
        loc_desc_dim=400,
        desc_bottleneck_dim=64,
    )

    meta_num = torch.randn(batch_size, num_input_dim)
    meta_cat = torch.randint(0, 5, (batch_size, len(cat_cardinalities)))
    meta_bin = torch.randint(0, 2, (batch_size, bin_input_dim)).float()

    # Real vectors; zero vector means unavailable.
    user_desc = torch.randn(batch_size, 768)
    loc_desc = torch.randn(batch_size, 400)
    user_desc[2] = 0.0
    loc_desc[2] = 0.0

    meta_repr, gate_weights = model(
        meta_num=meta_num,
        meta_cat=meta_cat,
        meta_bin=meta_bin,
        user_desc=user_desc,
        loc_desc=loc_desc,
        return_gate_weights=True,
    )

    print("meta_repr   :", meta_repr.shape)       # [4, 256]
    print("gate_weights:", gate_weights.shape)    # [4, 5]
    print("gate_weights sample:", gate_weights[0].detach().tolist())

    print("forward ok")

