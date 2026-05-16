from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastTextTagEncoder(nn.Module):
    """
    Lightweight fastText-style tag/keyword encoder.

    Input:
        tag_tokens: List[List[str]]
            In our SMP pipeline this can reuse batch["tag_tokens"] because those
            are already tokenized high-quality keyword/tag tokens.

    Output:
        [B, output_dim]

    Why this exists:
    - Flickr tags are short, noisy, sparse, and often keyword-like.
    - fastText pretrained word vectors are usually a better fit than sentence encoders
      for this kind of token bag.
    - This class uses pretrained .vec/.txt vectors or a saved .pt cache.

    Supported vector formats:
    - .pt cache created by save_cache()
    - text .vec/.txt format:
        first line can be either:
            <vocab_size> <dim>
        or a normal token vector line:
            word 0.1 0.2 ...
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

        if pad_token not in token_to_idx:
            raise ValueError(f"pad token '{pad_token}' not found in token_to_idx.")
        if unk_token not in token_to_idx:
            raise ValueError(f"unk token '{unk_token}' not found in token_to_idx.")
        if not isinstance(embedding_matrix, torch.Tensor) or embedding_matrix.ndim != 2:
            raise ValueError("embedding_matrix must be a 2D torch.Tensor.")

        self.token_to_idx = token_to_idx
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_idx = int(token_to_idx[pad_token])
        self.unk_idx = int(token_to_idx[unk_token])
        self.normalize_output = normalize_output

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.float(),
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
        tag_tokens: Sequence[Sequence[str]],
        device: Optional[torch.device | str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tag_tokens is None:
            raise ValueError("tag_tokens cannot be None.")

        batch_size = len(tag_tokens)
        max_len = max((len(tokens) for tokens in tag_tokens), default=0)

        if max_len == 0:
            token_ids = torch.full((batch_size, 1), self.pad_idx, dtype=torch.long, device=device)
            mask = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
            return token_ids, mask

        token_ids = torch.full((batch_size, max_len), self.pad_idx, dtype=torch.long, device=device)
        mask = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)

        for i, tokens in enumerate(tag_tokens):
            for j, token in enumerate(tokens[:max_len]):
                key = str(token).lower().strip()
                if not key:
                    idx = self.pad_idx
                    valid = 0.0
                else:
                    idx = self.token_to_idx.get(key, self.unk_idx)
                    valid = 1.0
                token_ids[i, j] = idx
                mask[i, j] = valid

        return token_ids, mask

    def forward(
        self,
        tag_tokens: Sequence[Sequence[str]],
    ) -> torch.Tensor:
        """
        Encode tokenized tags / keyword tokens.
        """
        if tag_tokens is None:
            raise ValueError("FastTextTagEncoder.forward requires tag_tokens.")

        device = self.embedding.weight.device
        token_ids, mask = self.tokens_to_indices(tag_tokens=tag_tokens, device=device)

        emb = self.embedding(token_ids)  # [B, T, D]
        mask_expanded = mask.unsqueeze(-1)

        summed = (emb * mask_expanded).sum(dim=1)
        counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        feat = summed / counts

        feat = self.dropout(feat)

        if self.proj is not None:
            feat = self.proj(feat)

        if self.normalize_output:
            feat = F.normalize(feat, p=2, dim=-1)

        return feat

    @staticmethod
    def load_vec_txt(
        vec_path: str | Path,
        max_vectors: Optional[int] = None,
        lowercase: bool = True,
        add_special_tokens: bool = True,
        encoding: str = "utf-8",
    ) -> Tuple[Dict[str, int], torch.Tensor]:
        """
        Load fastText .vec/.txt text vectors.

        Handles optional header line:
            1000000 300
        """
        vec_path = Path(vec_path)
        if not vec_path.exists():
            raise FileNotFoundError(f"fastText vector file not found: {vec_path}")

        words: List[str] = []
        vectors: List[List[float]] = []
        expected_dim: Optional[int] = None

        with vec_path.open("r", encoding=encoding, errors="ignore") as f:
            for line_num, line in enumerate(f):
                line = line.rstrip("\n")
                if not line:
                    continue

                parts = line.split()
                if len(parts) <= 2:
                    continue

                # Header line: "<vocab> <dim>"
                if line_num == 0 and len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    expected_dim = int(parts[1])
                    continue

                word = parts[0].lower() if lowercase else parts[0]

                try:
                    vector = [float(x) for x in parts[1:]]
                except ValueError:
                    continue

                if expected_dim is not None and len(vector) != expected_dim:
                    continue

                if expected_dim is None:
                    expected_dim = len(vector)

                words.append(word)
                vectors.append(vector)

                if max_vectors is not None and len(words) >= int(max_vectors):
                    break

        if not vectors:
            raise ValueError(f"No valid vectors loaded from {vec_path}")

        vector_tensor = torch.tensor(vectors, dtype=torch.float32)
        embed_dim = int(vector_tensor.shape[1])

        token_to_idx: Dict[str, int] = {}
        embedding_rows: List[torch.Tensor] = []

        if add_special_tokens:
            token_to_idx["<pad>"] = 0
            embedding_rows.append(torch.zeros(embed_dim, dtype=torch.float32))

            token_to_idx["<unk>"] = 1
            embedding_rows.append(vector_tensor.mean(dim=0))

        for i, word in enumerate(words):
            if word in token_to_idx:
                continue
            token_to_idx[word] = len(embedding_rows)
            embedding_rows.append(vector_tensor[i])

        embedding_matrix = torch.stack(embedding_rows, dim=0)
        return token_to_idx, embedding_matrix

    @staticmethod
    def save_cache(
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
    def load_cache(
        cache_path: str | Path,
    ) -> Tuple[Dict[str, int], torch.Tensor]:
        cache = torch.load(Path(cache_path), map_location="cpu")
        token_to_idx = cache["token_to_idx"]
        embedding_matrix = cache["embedding_matrix"]
        if not isinstance(token_to_idx, dict):
            raise TypeError("Cached token_to_idx must be a dict.")
        if not isinstance(embedding_matrix, torch.Tensor):
            raise TypeError("Cached embedding_matrix must be a torch.Tensor.")
        return token_to_idx, embedding_matrix
