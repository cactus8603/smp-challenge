from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    dropout: float = 0.1,
    activation: str = "gelu",
    use_layernorm: bool = True,
) -> nn.Sequential:
    if input_dim <= 0:
        raise ValueError("build_mlp input_dim must be positive.")
    if not hidden_dims:
        raise ValueError("build_mlp requires at least one hidden dim.")

    act_layer = nn.GELU if activation.lower() == "gelu" else nn.ReLU

    layers: List[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if use_layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(act_layer())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    return nn.Sequential(*layers)


class VectorCompressor(nn.Module):
    """
    Compress high-dimensional precomputed vectors, e.g. user_desc embeddings.
    Kept here because fusion_model imports it from meta_encoder.py.
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
        return self.net(x.float())


class FieldwiseCategoricalEncoder(nn.Module):
    def __init__(
        self,
        cardinalities: Sequence[int],
        output_dim: int,
        dropout: float = 0.1,
        fixed_embedding_dim: Optional[int] = None,
        min_embedding_dim: int = 4,
        max_embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.cardinalities = list(cardinalities)
        self.num_fields = len(self.cardinalities)
        self.output_dim = output_dim

        if self.num_fields == 0:
            self.embeddings = nn.ModuleList()
            self.proj = None
            return

        embedding_dims: List[int] = []
        for card in self.cardinalities:
            card = max(int(card), 1)
            if fixed_embedding_dim is not None:
                emb_dim = int(fixed_embedding_dim)
            else:
                emb_dim = int(card ** 0.5)
                emb_dim = max(min_embedding_dim, emb_dim)
                emb_dim = min(max_embedding_dim, emb_dim)
            embedding_dims.append(emb_dim)

        self.embeddings = nn.ModuleList([
            nn.Embedding(max(int(card), 1), int(dim))
            for card, dim in zip(self.cardinalities, embedding_dims)
        ])
        total_dim = int(sum(embedding_dims))
        self.proj = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_fields == 0:
            return x.new_zeros((x.size(0), 0), dtype=torch.float32)

        embs = []
        x = x.long()
        for i, emb in enumerate(self.embeddings):
            ids = x[:, i].clamp(min=0, max=emb.num_embeddings - 1)
            embs.append(emb(ids))
        return self.proj(torch.cat(embs, dim=-1))


def _clean_col_name(name: str) -> str:
    for prefix in ("num__", "cat__", "bin__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


class SemanticMetaGroupEncoder(nn.Module):
    """
    Encode one semantic metadata group.

    This keeps the semantic grouping idea, but avoids a Transformer inside
    metadata. Each group produces one token, then tokens are combined by a
    learned soft gate.
    """

    def __init__(
        self,
        name: str,
        num_indices: Sequence[int],
        cat_indices: Sequence[int],
        bin_indices: Sequence[int],
        cat_cardinalities: Sequence[int],
        group_dim: int,
        dropout: float = 0.1,
        fixed_cat_embedding_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.num_indices = list(num_indices)
        self.cat_indices = list(cat_indices)
        self.bin_indices = list(bin_indices)

        self.register_buffer("_num_idx", torch.tensor(self.num_indices, dtype=torch.long), persistent=False)
        self.register_buffer("_cat_idx", torch.tensor(self.cat_indices, dtype=torch.long), persistent=False)
        self.register_buffer("_bin_idx", torch.tensor(self.bin_indices, dtype=torch.long), persistent=False)

        self.use_num = len(self.num_indices) > 0
        self.use_cat = len(self.cat_indices) > 0
        self.use_bin = len(self.bin_indices) > 0

        part_count = int(self.use_num) + int(self.use_cat) + int(self.use_bin)
        if part_count == 0:
            raise ValueError(f"Semantic group {name} is empty.")

        if self.use_num:
            self.num_encoder = build_mlp(
                input_dim=len(self.num_indices),
                hidden_dims=[max(group_dim, 64), group_dim],
                dropout=dropout,
                activation="gelu",
                use_layernorm=True,
            )
        else:
            self.num_encoder = None

        if self.use_cat:
            selected_cards = [int(cat_cardinalities[i]) for i in self.cat_indices]
            self.cat_encoder = FieldwiseCategoricalEncoder(
                cardinalities=selected_cards,
                output_dim=group_dim,
                dropout=dropout,
                fixed_embedding_dim=fixed_cat_embedding_dim,
            )
        else:
            self.cat_encoder = None

        if self.use_bin:
            self.bin_encoder = build_mlp(
                input_dim=len(self.bin_indices),
                hidden_dims=[max(group_dim // 2, 32), group_dim],
                dropout=dropout,
                activation="gelu",
                use_layernorm=True,
            )
        else:
            self.bin_encoder = None

        if part_count == 1:
            self.merge = nn.Identity()
        else:
            self.merge = nn.Sequential(
                nn.Linear(group_dim * part_count, group_dim),
                nn.LayerNorm(group_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(group_dim, group_dim),
                nn.LayerNorm(group_dim),
            )

    def forward(
        self,
        meta_num: Optional[torch.Tensor],
        meta_cat: Optional[torch.Tensor],
        meta_bin: Optional[torch.Tensor],
    ) -> torch.Tensor:
        parts: List[torch.Tensor] = []

        if self.use_num:
            if meta_num is None:
                raise ValueError(f"{self.name} requires meta_num")
            x = meta_num.index_select(1, self._num_idx.to(meta_num.device)).float()
            parts.append(self.num_encoder(x))

        if self.use_cat:
            if meta_cat is None:
                raise ValueError(f"{self.name} requires meta_cat")
            x = meta_cat.index_select(1, self._cat_idx.to(meta_cat.device)).long()
            parts.append(self.cat_encoder(x))

        if self.use_bin:
            if meta_bin is None:
                raise ValueError(f"{self.name} requires meta_bin")
            x = meta_bin.index_select(1, self._bin_idx.to(meta_bin.device)).float()
            parts.append(self.bin_encoder(x))

        if len(parts) == 1:
            return parts[0]
        return self.merge(torch.cat(parts, dim=-1))


class SemanticGroupGatedFusion(nn.Module):
    """
    Stable semantic group fusion:
        group_tokens -> soft gate -> weighted sum -> output MLP

    No Transformer here. This should be closer to the old stable pairwise model
    while still letting metadata use semantic grouping.
    """

    def __init__(
        self,
        group_dim: int,
        output_dim: int,
        num_groups: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.group_embedding = nn.Parameter(torch.zeros(1, num_groups, group_dim))
        nn.init.normal_(self.group_embedding, mean=0.0, std=0.02)

        self.gate_net = nn.Sequential(
            nn.Linear(group_dim, max(group_dim // 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(group_dim // 2, 32), 1),
        )

        self.out = nn.Sequential(
            nn.Linear(group_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, group_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = group_tokens + self.group_embedding[:, : group_tokens.size(1), :]
        logits = self.gate_net(x).squeeze(-1)
        weights = F.softmax(logits, dim=-1)
        fused = (x * weights.unsqueeze(-1)).sum(dim=1)
        return self.out(fused), weights


class LegacyTypedMetaEncoder(nn.Module):
    """
    Fallback typed encoder: num/cat/bin branches with gated merge.
    """

    def __init__(
        self,
        num_input_dim: int,
        cat_cardinalities: Sequence[int],
        bin_input_dim: int,
        output_dim: int,
        branch_dim: int,
        dropout: float = 0.1,
        fixed_cat_embedding_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.use_num = num_input_dim > 0
        self.use_cat = len(cat_cardinalities) > 0
        self.use_bin = bin_input_dim > 0

        if self.use_num:
            self.num_encoder = build_mlp(
                input_dim=num_input_dim,
                hidden_dims=[max(branch_dim, 128), branch_dim],
                dropout=dropout,
            )
        else:
            self.num_encoder = None

        if self.use_cat:
            self.cat_encoder = FieldwiseCategoricalEncoder(
                cat_cardinalities,
                output_dim=branch_dim,
                dropout=dropout,
                fixed_embedding_dim=fixed_cat_embedding_dim,
            )
        else:
            self.cat_encoder = None

        if self.use_bin:
            self.bin_encoder = build_mlp(
                input_dim=bin_input_dim,
                hidden_dims=[max(branch_dim // 2, 32), branch_dim],
                dropout=dropout,
            )
        else:
            self.bin_encoder = None

        n = int(self.use_num) + int(self.use_cat) + int(self.use_bin)
        self.n = n
        self.gate = nn.Sequential(
            nn.Linear(branch_dim * n, branch_dim),
            nn.GELU(),
            nn.Linear(branch_dim, n),
        ) if n > 1 else None

        self.out = nn.Sequential(
            nn.Linear(branch_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        meta_num: Optional[torch.Tensor],
        meta_cat: Optional[torch.Tensor],
        meta_bin: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reps: List[torch.Tensor] = []
        if self.use_num:
            reps.append(self.num_encoder(meta_num.float()))
        if self.use_cat:
            reps.append(self.cat_encoder(meta_cat.long()))
        if self.use_bin:
            reps.append(self.bin_encoder(meta_bin.float()))

        if len(reps) == 1:
            fused = reps[0]
            gate_weights = fused.new_ones((fused.size(0), 1))
        else:
            stacked = torch.stack(reps, dim=1)
            logits = self.gate(torch.cat(reps, dim=-1))
            gate_weights = F.softmax(logits, dim=-1)
            fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)

        return self.out(fused), gate_weights


class NumericFeatureTokenizer(nn.Module):
    """
    FT-Transformer numeric tokenizer.

    Each numeric feature has its own weight vector and bias:
        token_i = x_i * W_i + b_i
    """

    def __init__(self, num_features: int, token_dim: int) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.token_dim = int(token_dim)
        if self.num_features > 0:
            self.weight = nn.Parameter(torch.empty(self.num_features, self.token_dim))
            self.bias = nn.Parameter(torch.zeros(self.num_features, self.token_dim))
            nn.init.xavier_uniform_(self.weight)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        if self.num_features <= 0:
            if x is None:
                raise ValueError("NumericFeatureTokenizer has no features and cannot infer batch/device.")
            return x.new_zeros((x.size(0), 0, self.token_dim), dtype=torch.float32)
        if x is None:
            raise ValueError("NumericFeatureTokenizer requires meta_num.")
        x = x.float()
        if x.size(1) != self.num_features:
            raise ValueError(f"Expected {self.num_features} numeric features, got {x.size(1)}")
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class BinaryFeatureTokenizer(nn.Module):
    """
    Binary tokenizer. Binary flags are treated like small numeric feature tokens.
    """

    def __init__(self, bin_features: int, token_dim: int) -> None:
        super().__init__()
        self.bin_features = int(bin_features)
        self.token_dim = int(token_dim)
        if self.bin_features > 0:
            self.weight = nn.Parameter(torch.empty(self.bin_features, self.token_dim))
            self.bias = nn.Parameter(torch.zeros(self.bin_features, self.token_dim))
            nn.init.xavier_uniform_(self.weight)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        if self.bin_features <= 0:
            if x is None:
                raise ValueError("BinaryFeatureTokenizer has no features and cannot infer batch/device.")
            return x.new_zeros((x.size(0), 0, self.token_dim), dtype=torch.float32)
        if x is None:
            raise ValueError("BinaryFeatureTokenizer requires meta_bin.")
        x = x.float()
        if x.size(1) != self.bin_features:
            raise ValueError(f"Expected {self.bin_features} binary features, got {x.size(1)}")
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CategoricalFeatureTokenizer(nn.Module):
    """
    Field-wise categorical tokenizer for FT-Transformer.
    Each categorical field owns its embedding table and produces one token.
    """

    def __init__(self, cardinalities: Sequence[int], token_dim: int) -> None:
        super().__init__()
        self.cardinalities = [max(int(c), 1) for c in cardinalities]
        self.num_fields = len(self.cardinalities)
        self.token_dim = int(token_dim)
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, self.token_dim) for card in self.cardinalities
        ])
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, x: Optional[torch.Tensor], batch_size: Optional[int] = None, device=None) -> torch.Tensor:
        if self.num_fields <= 0:
            if x is not None:
                batch_size = x.size(0)
                device = x.device
            if batch_size is None or device is None:
                raise ValueError("CategoricalFeatureTokenizer cannot infer empty output shape.")
            return torch.zeros(batch_size, 0, self.token_dim, device=device, dtype=torch.float32)
        if x is None:
            raise ValueError("CategoricalFeatureTokenizer requires meta_cat.")
        x = x.long()
        if x.size(1) != self.num_fields:
            raise ValueError(f"Expected {self.num_fields} categorical fields, got {x.size(1)}")
        tokens = []
        for i, emb in enumerate(self.embeddings):
            ids = x[:, i].clamp(min=0, max=emb.num_embeddings - 1)
            tokens.append(emb(ids))
        return torch.stack(tokens, dim=1)


class FTTransformerMetaEncoder(nn.Module):
    """
    Feature Tokenizer + Transformer metadata encoder.

    It tokenizes every numeric/categorical/binary feature as an individual token,
    prepends a CLS token, runs a compact TransformerEncoder, then projects CLS
    to output_dim.
    """

    def __init__(
        self,
        num_input_dim: int,
        cat_cardinalities: Sequence[int],
        bin_input_dim: int,
        output_dim: int,
        token_dim: int = 192,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_mult: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.num_input_dim = int(num_input_dim)
        self.cat_cardinalities = list(cat_cardinalities or [])
        self.cat_input_dim = len(self.cat_cardinalities)
        self.bin_input_dim = int(bin_input_dim)
        self.output_dim = int(output_dim)
        self.token_dim = int(token_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)

        if self.token_dim % self.num_heads != 0:
            raise ValueError(f"ft_token_dim={self.token_dim} must be divisible by ft_num_heads={self.num_heads}.")
        if self.num_input_dim + self.cat_input_dim + self.bin_input_dim <= 0:
            raise ValueError("FTTransformerMetaEncoder requires at least one metadata feature.")

        self.num_tokenizer = NumericFeatureTokenizer(self.num_input_dim, self.token_dim) if self.num_input_dim > 0 else None
        self.cat_tokenizer = CategoricalFeatureTokenizer(self.cat_cardinalities, self.token_dim) if self.cat_input_dim > 0 else None
        self.bin_tokenizer = BinaryFeatureTokenizer(self.bin_input_dim, self.token_dim) if self.bin_input_dim > 0 else None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        # Lightweight field-type embeddings help the transformer distinguish
        # numeric/categorical/binary tokens without hard-coded groups.
        self.type_embeddings = nn.Parameter(torch.zeros(1, 4, self.token_dim))  # cls, num, cat, bin
        nn.init.normal_(self.type_embeddings, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=self.num_heads,
            dim_feedforward=max(self.token_dim * int(ffn_mult), self.token_dim),
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.final_norm = nn.LayerNorm(self.token_dim) if use_layernorm else nn.Identity()

        self.out = nn.Sequential(
            nn.Linear(self.token_dim, output_dim),
            nn.LayerNorm(output_dim) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim) if use_layernorm else nn.Identity(),
        )

    def forward(
        self,
        meta_num: Optional[torch.Tensor],
        meta_cat: Optional[torch.Tensor],
        meta_bin: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = None
        device = None
        for x in (meta_num, meta_cat, meta_bin):
            if x is not None:
                batch_size = x.size(0)
                device = x.device
                break
        if batch_size is None or device is None:
            raise ValueError("FTTransformerMetaEncoder cannot infer batch/device from empty metadata.")

        tokens: List[torch.Tensor] = []
        cls = self.cls_token.expand(batch_size, -1, -1) + self.type_embeddings[:, 0:1, :]
        tokens.append(cls)

        if self.num_tokenizer is not None:
            num_tok = self.num_tokenizer(meta_num) + self.type_embeddings[:, 1:2, :]
            tokens.append(num_tok)
        if self.cat_tokenizer is not None:
            cat_tok = self.cat_tokenizer(meta_cat, batch_size=batch_size, device=device) + self.type_embeddings[:, 2:3, :]
            tokens.append(cat_tok)
        if self.bin_tokenizer is not None:
            bin_tok = self.bin_tokenizer(meta_bin) + self.type_embeddings[:, 3:4, :]
            tokens.append(bin_tok)

        x = torch.cat(tokens, dim=1)
        x = self.transformer(x)
        cls_out = self.final_norm(x[:, 0, :])
        out = self.out(cls_out)

        # Placeholder info tensor for compatibility with MetaEncoder(return_gate_weights=True).
        # Shape [B, 1] means there is no soft gate here.
        info = out.new_ones((out.size(0), 1))
        return out, info


class MetaEncoder(nn.Module):
    """
    Semantic grouped metadata encoder without Transformer.

    Groups:
    - taxonomy
    - temporal_user
    - geo
    - content_profile_availability
    - other

    Each group can contain num/cat/bin subsets. Final fusion is soft gated sum.
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
        activation: str = "gelu",
        use_layernorm: bool = True,
        fixed_cat_embedding_dim: Optional[int] = None,
        min_cat_embedding_dim: int = 4,
        max_cat_embedding_dim: int = 32,
        user_desc_dim: int = 768,
        loc_desc_dim: int = 400,
        desc_bottleneck_dim: int = 64,
        use_user_desc: bool = False,
        use_loc_desc: bool = False,
        meta_num_cols: Optional[Sequence[str]] = None,
        meta_cat_cols: Optional[Sequence[str]] = None,
        meta_bin_cols: Optional[Sequence[str]] = None,
        use_semantic_groups: bool = True,
        encoder_type: str = "legacy",  # legacy | semantic_groups | ft_transformer
        ft_token_dim: int = 192,
        ft_num_layers: int = 3,
        ft_num_heads: int = 8,
        ft_ffn_mult: int = 2,
        ft_dropout: Optional[float] = None,
        semantic_group_layers: int = 0,
        semantic_group_heads: int = 4,
        semantic_group_ffn_mult: int = 4,
    ) -> None:
        super().__init__()

        self.num_input_dim = int(num_input_dim)
        self.cat_cardinalities = list(cat_cardinalities or [])
        self.cat_input_dim = len(self.cat_cardinalities)
        self.bin_input_dim = int(bin_input_dim)
        self.output_dim = int(output_dim)
        self.branch_dim = int(branch_dim)

        self.meta_num_cols = list(meta_num_cols or [])
        self.meta_cat_cols = list(meta_cat_cols or [])
        self.meta_bin_cols = list(meta_bin_cols or [])

        self.encoder_type = str(encoder_type or "legacy").lower()
        # Backward compatible alias: older configs used use_semantic_groups=true.
        if self.encoder_type in {"semantic", "semantic_group", "semantic_groups"}:
            self.encoder_type = "semantic_groups"
        if self.encoder_type in {"ft", "fttransformer", "ft-transformer"}:
            self.encoder_type = "ft_transformer"
        if self.encoder_type not in {"legacy", "semantic_groups", "ft_transformer"}:
            raise ValueError(f"Unsupported meta encoder_type={encoder_type}")
        if use_semantic_groups and self.encoder_type == "legacy":
            self.encoder_type = "semantic_groups"

        self.use_semantic_groups = bool(
            self.encoder_type == "semantic_groups"
            and (
                len(self.meta_num_cols) == self.num_input_dim
                and len(self.meta_cat_cols) == self.cat_input_dim
                and len(self.meta_bin_cols) == self.bin_input_dim
            )
        )
        self.use_ft_transformer = self.encoder_type == "ft_transformer"

        self.ft_encoder = None
        self.group_names = ["num", "cat", "bin"]
        self.group_encoders = None
        self.group_fusion = None
        self.legacy_encoder = None

        if self.use_ft_transformer:
            self.ft_encoder = FTTransformerMetaEncoder(
                num_input_dim=self.num_input_dim,
                cat_cardinalities=self.cat_cardinalities,
                bin_input_dim=self.bin_input_dim,
                output_dim=self.output_dim,
                token_dim=int(ft_token_dim),
                num_layers=int(ft_num_layers),
                num_heads=int(ft_num_heads),
                ffn_mult=int(ft_ffn_mult),
                dropout=dropout if ft_dropout is None else float(ft_dropout),
                activation=activation,
                use_layernorm=use_layernorm,
            )
        elif self.use_semantic_groups:
            self.group_names, group_defs = self._build_group_defs()
            self.group_encoders = nn.ModuleDict()
            for group_name, defs in group_defs.items():
                self.group_encoders[group_name] = SemanticMetaGroupEncoder(
                    name=group_name,
                    num_indices=defs["num"],
                    cat_indices=defs["cat"],
                    bin_indices=defs["bin"],
                    cat_cardinalities=self.cat_cardinalities,
                    group_dim=self.branch_dim,
                    dropout=dropout,
                    fixed_cat_embedding_dim=fixed_cat_embedding_dim,
                )

            self.group_fusion = SemanticGroupGatedFusion(
                group_dim=self.branch_dim,
                output_dim=self.output_dim,
                num_groups=len(self.group_names),
                dropout=dropout,
            )
        else:
            self.legacy_encoder = LegacyTypedMetaEncoder(
                num_input_dim=self.num_input_dim,
                cat_cardinalities=self.cat_cardinalities,
                bin_input_dim=self.bin_input_dim,
                output_dim=self.output_dim,
                branch_dim=self.branch_dim,
                dropout=dropout,
                fixed_cat_embedding_dim=fixed_cat_embedding_dim,
            )

    def _build_group_defs(self) -> Tuple[List[str], Dict[str, Dict[str, List[int]]]]:
        group_order = ["taxonomy", "temporal_user", "geo", "content_profile_availability", "other"]
        group_defs: Dict[str, Dict[str, List[int]]] = {
            g: {"num": [], "cat": [], "bin": []} for g in group_order
        }

        used_num, used_cat, used_bin = set(), set(), set()

        def add(kind: str, group: str, idx: int) -> None:
            group_defs[group][kind].append(idx)
            if kind == "num":
                used_num.add(idx)
            elif kind == "cat":
                used_cat.add(idx)
            else:
                used_bin.add(idx)

        taxonomy_cat = {
            "category", "subcategory", "concept",
            "category_subcategory_combo", "category_concept_combo",
        }
        geo_cat = {"geo_cluster", "timezone_id", "city", "state", "country"}

        geo_num_keywords = {
            "latitude", "longitude", "geoaccuracy",
            "country_target_enc", "state_target_enc", "city_target_enc",
            "location_text_target_enc",
            "country_freq", "state_freq", "city_freq", "location_text_freq",
        }

        temporal_user_num_keywords = {
            # Time features
            "hour", "weekday", "year", "month", "day", "weekofyear",
            "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
            "month_sin", "month_cos",

            # Fold-safe user aggregates (highest-signal features)
            "user_prev_post_count", "user_mean_label", "user_median_label",
            "user_std_label", "user_category_nunique", "user_active_hour_mean",

            # User profile (only fields that actually exist in official data)
            "photo_count_log1p",
        }

        content_profile_num_keywords = {
            "title_len", "tags_len", "full_text_len",
            "title_word_count", "full_text_word_count",
            "tag_count", "avg_tag_len",
            "title_digit_ratio", "title_upper_ratio", "title_punct_ratio",
            "full_text_digit_ratio", "full_text_punct_ratio",
            "clip_token_count", "clip_token_count_raw",
            "tag_token_count",
            "user_desc_len", "user_desc_word_count", "user_desc_keyword_count",
        }

        temporal_user_bin = {"is_weekend", "is_night", "is_workhour", "ispro", "canbuypro", "ispublic"}
        geo_bin = {"has_geo", "has_location_text", "has_city", "has_state", "has_country"}
        availability_bin = {
            "has_title", "has_tags", "has_full_text", "has_user_description", "has_image",
            "user_desc_kw_photo", "user_desc_kw_travel", "user_desc_kw_art",
            "user_desc_kw_nature", "user_desc_kw_camera", "user_desc_kw_pro",
            "user_desc_kw_social", "user_desc_kw_location",
        }

        for i, col in enumerate(self.meta_cat_cols):
            raw = _clean_col_name(col)
            if raw in taxonomy_cat:
                add("cat", "taxonomy", i)
            elif raw in geo_cat:
                add("cat", "geo", i)

        for i, col in enumerate(self.meta_num_cols):
            raw = _clean_col_name(col)
            if raw in geo_num_keywords:
                add("num", "geo", i)
            elif raw in temporal_user_num_keywords:
                add("num", "temporal_user", i)
            elif raw in content_profile_num_keywords:
                add("num", "content_profile_availability", i)

        for i, col in enumerate(self.meta_bin_cols):
            raw = _clean_col_name(col)
            if raw in temporal_user_bin:
                add("bin", "temporal_user", i)
            elif raw in geo_bin:
                add("bin", "geo", i)
            elif raw in availability_bin:
                add("bin", "content_profile_availability", i)

        for i in range(self.num_input_dim):
            if i not in used_num:
                add("num", "other", i)
        for i in range(self.cat_input_dim):
            if i not in used_cat:
                add("cat", "other", i)
        for i in range(self.bin_input_dim):
            if i not in used_bin:
                add("bin", "other", i)

        non_empty_names = [
            g for g in group_order
            if group_defs[g]["num"] or group_defs[g]["cat"] or group_defs[g]["bin"]
        ]
        return non_empty_names, {g: group_defs[g] for g in non_empty_names}

    def forward(
        self,
        meta_num: Optional[torch.Tensor] = None,
        meta_cat: Optional[torch.Tensor] = None,
        meta_bin: Optional[torch.Tensor] = None,
        user_desc: Optional[torch.Tensor] = None,
        loc_desc: Optional[torch.Tensor] = None,
        return_gate_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.use_ft_transformer:
            out, info = self.ft_encoder(meta_num, meta_cat, meta_bin)
            return (out, info) if return_gate_weights else out

        if not self.use_semantic_groups:
            out, gates = self.legacy_encoder(meta_num, meta_cat, meta_bin)
            return (out, gates) if return_gate_weights else out

        group_tokens = []
        for name in self.group_names:
            group_tokens.append(self.group_encoders[name](meta_num, meta_cat, meta_bin))
        tokens = torch.stack(group_tokens, dim=1)
        out, weights = self.group_fusion(tokens)
        return (out, weights) if return_gate_weights else out


if __name__ == "__main__":
    B = 4
    num_cols = ["num__hour", "num__city_target_enc", "num__user_desc_len"]
    cat_cols = ["cat__category", "cat__city"]
    bin_cols = ["bin__has_geo", "bin__has_user_description"]

    model = MetaEncoder(
        num_input_dim=len(num_cols),
        cat_cardinalities=[10, 100],
        bin_input_dim=len(bin_cols),
        output_dim=256,
        branch_dim=128,
        meta_num_cols=num_cols,
        meta_cat_cols=cat_cols,
        meta_bin_cols=bin_cols,
        use_semantic_groups=True,
    )
    y, w = model(
        meta_num=torch.randn(B, len(num_cols)),
        meta_cat=torch.randint(0, 5, (B, len(cat_cols))),
        meta_bin=torch.randint(0, 2, (B, len(bin_cols))).float(),
        return_gate_weights=True,
    )
    print(y.shape, w.shape, model.group_names)
