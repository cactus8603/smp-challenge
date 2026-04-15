from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn


class GloveEncoder(nn.Module):
    """
    Simple GloVe encoder for lexical text features.

    Design goals:
    - Input: batch-level `glove_tokens`, i.e. List[List[str]]
    - Lookup token embeddings from a frozen embedding table
    - Mean-pool valid token embeddings into a fixed-size vector [B, embed_dim]
    - Optional projection to `output_dim`
    - Safe fallback to zero vector when a sample has no valid tokens

    Typical usage:
        token_to_idx, embeddings = GloveEncoder.load_glove_txt("glove.6B.300d.txt")
        encoder = GloveEncoder(
            token_to_idx=token_to_idx,
            embedding_matrix=embeddings,
            output_dim=256,
            dropout=0.1,
            trainable=False,
        )
        feat = encoder(glove_tokens=batch["glove_tokens"])   # [B, 256]
    """

    def __init__(
        self,
        token_to_idx: Dict[str, int],
        embedding_matrix: torch.Tensor,
        output_dim: Optional[int] = None,
        dropout: float = 0.1,
        trainable: bool = False,
        normalize_output: bool = False,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
    ) -> None:
        super().__init__()

        if not isinstance(token_to_idx, dict):
            raise TypeError("token_to_idx must be a dict[str, int].")
        if not isinstance(embedding_matrix, torch.Tensor):
            raise TypeError("embedding_matrix must be a torch.Tensor.")
        if embedding_matrix.ndim != 2:
            raise ValueError(
                f"embedding_matrix must be 2D [vocab_size, embed_dim], got shape={tuple(embedding_matrix.shape)}"
            )

        self.token_to_idx = token_to_idx
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.normalize_output = normalize_output

        if self.pad_token not in self.token_to_idx:
            raise ValueError(f"pad token '{self.pad_token}' not found in token_to_idx.")
        if self.unk_token not in self.token_to_idx:
            raise ValueError(f"unk token '{self.unk_token}' not found in token_to_idx.")

        self.pad_idx = int(self.token_to_idx[self.pad_token])
        self.unk_idx = int(self.token_to_idx[self.unk_token])

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=not trainable,
            padding_idx=self.pad_idx,
        )

        self.embed_dim = int(embedding_matrix.shape[1])
        self.output_dim = int(output_dim) if output_dim is not None else self.embed_dim

        self.dropout = nn.Dropout(dropout)

        if self.output_dim != self.embed_dim:
            self.proj = nn.Sequential(
                nn.Linear(self.embed_dim, self.output_dim),
                nn.LayerNorm(self.output_dim),
            )
        else:
            self.proj = None

    def tokens_to_indices(
        self,
        glove_tokens: Sequence[Sequence[str]],
        device: Optional[torch.device | str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert List[List[str]] into:
        - token_ids: [B, T]
        - mask: [B, T] where 1 means valid token, 0 means padding
        """
        if glove_tokens is None:
            raise ValueError("glove_tokens cannot be None.")

        batch_size = len(glove_tokens)
        max_len = max((len(tokens) for tokens in glove_tokens), default=0)

        if max_len == 0:
            token_ids = torch.full((batch_size, 1), self.pad_idx, dtype=torch.long, device=device)
            mask = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
            return token_ids, mask

        token_ids = torch.full((batch_size, max_len), self.pad_idx, dtype=torch.long, device=device)
        mask = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)

        for i, tokens in enumerate(glove_tokens):
            for j, token in enumerate(tokens[:max_len]):
                idx = self.token_to_idx.get(str(token).lower(), self.unk_idx)
                token_ids[i, j] = idx
                mask[i, j] = 1.0

        return token_ids, mask

    def forward(
        self,
        glove_tokens: Sequence[Sequence[str]],
    ) -> torch.Tensor:
        """
        Args:
            glove_tokens: List[List[str]] from dataset/collate

        Returns:
            feat: [B, output_dim]
        """
        device = self.embedding.weight.device
        token_ids, mask = self.tokens_to_indices(glove_tokens=glove_tokens, device=device)

        # [B, T, D]
        emb = self.embedding(token_ids)

        # mean pooling over valid tokens
        mask_expanded = mask.unsqueeze(-1)  # [B, T, 1]
        summed = (emb * mask_expanded).sum(dim=1)  # [B, D]
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]
        feat = summed / counts

        feat = self.dropout(feat)

        if self.proj is not None:
            feat = self.proj(feat)

        if self.normalize_output:
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)

        return feat

    @staticmethod
    def load_glove_txt(
        glove_path: str | Path,
        max_vectors: Optional[int] = None,
        add_special_tokens: bool = True,
        lowercase: bool = True,
        encoding: str = "utf-8",
    ) -> tuple[Dict[str, int], torch.Tensor]:
        """
        Load a standard GloVe-style text file, e.g.:
            word 0.123 0.456 ...

        Returns:
            token_to_idx, embedding_matrix

        Notes:
        - Adds <pad> and <unk> rows by default
        - <pad> is all zeros
        - <unk> is mean of all loaded vectors
        """
        glove_path = Path(glove_path)
        if not glove_path.exists():
            raise FileNotFoundError(f"GloVe file not found: {glove_path}")

        words: List[str] = []
        vectors: List[List[float]] = []

        with glove_path.open("r", encoding=encoding, errors="ignore") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) <= 2:
                    continue

                word = parts[0].lower() if lowercase else parts[0]
                try:
                    vector = [float(x) for x in parts[1:]]
                except ValueError:
                    continue

                words.append(word)
                vectors.append(vector)

                if max_vectors is not None and len(words) >= max_vectors:
                    break

        if not vectors:
            raise ValueError(f"No valid vectors loaded from {glove_path}")

        vector_tensor = torch.tensor(vectors, dtype=torch.float32)
        embed_dim = int(vector_tensor.shape[1])

        token_to_idx: Dict[str, int] = {}
        embedding_rows: List[torch.Tensor] = []

        if add_special_tokens:
            token_to_idx["<pad>"] = 0
            embedding_rows.append(torch.zeros(embed_dim, dtype=torch.float32))

            token_to_idx["<unk>"] = 1
            embedding_rows.append(vector_tensor.mean(dim=0))

        start_idx = len(token_to_idx)
        for i, word in enumerate(words):
            if word in token_to_idx:
                continue
            token_to_idx[word] = start_idx + len(embedding_rows) - (2 if add_special_tokens else 0)
            embedding_rows.append(vector_tensor[i])

        embedding_matrix = torch.stack(embedding_rows, dim=0)
        return token_to_idx, embedding_matrix

    @staticmethod
    def save_glove_cache(
        token_to_idx: Dict[str, int],
        embedding_matrix: torch.Tensor,
        cache_path: str | Path,
    ) -> None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "token_to_idx": token_to_idx,
                "embedding_matrix": embedding_matrix.cpu(),
            },
            cache_path,
        )

    @staticmethod
    def load_glove_cache(
        cache_path: str | Path,
    ) -> tuple[Dict[str, int], torch.Tensor]:
        cache = torch.load(Path(cache_path), map_location="cpu")
        token_to_idx = cache["token_to_idx"]
        embedding_matrix = cache["embedding_matrix"]
        if not isinstance(token_to_idx, dict):
            raise TypeError("Cached token_to_idx must be a dict.")
        if not isinstance(embedding_matrix, torch.Tensor):
            raise TypeError("Cached embedding_matrix must be a torch.Tensor.")
        return token_to_idx, embedding_matrix


if __name__ == "__main__":
    # Minimal smoke test
    token_to_idx = {
        "<pad>": 0,
        "<unk>": 1,
        "hello": 2,
        "world": 3,
        "cat": 4,
    }
    embedding_matrix = torch.randn(5, 50)
    embedding_matrix[0].zero_()

    encoder = GloveEncoder(
        token_to_idx=token_to_idx,
        embedding_matrix=embedding_matrix,
        output_dim=32,
        dropout=0.1,
        trainable=False,
    )

    glove_tokens = [
        ["hello", "world"],
        ["cat", "unknown_token"],
        [],
    ]
    y = encoder(glove_tokens=glove_tokens)
    print("output shape:", y.shape)
