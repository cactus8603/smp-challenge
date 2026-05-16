from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    s = str(x).strip()
    return s if s else default


def _resolve_path(path: str | Path, project_root: str | Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if project_root is None:
        return p.resolve()
    return (Path(project_root) / p).resolve()


def _safe_cache_name(model_name: str, cache_name: Optional[str] = None) -> str:
    if cache_name:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cache_name)).strip("_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")


def _select_user_texts(
    df: pd.DataFrame,
    uid_col: str,
    text_col: str,
    fallback_text_col: Optional[str] = None,
    max_chars: int = 1024,
) -> tuple[list[str], list[str]]:
    if uid_col not in df.columns:
        raise ValueError(f"uid_col={uid_col!r} not found in dataframe.")

    if text_col not in df.columns:
        if fallback_text_col and fallback_text_col in df.columns:
            text_col = fallback_text_col
        else:
            raise ValueError(
                f"text_col={text_col!r} not found, and fallback_text_col={fallback_text_col!r} is unavailable."
            )

    work_cols = [uid_col, text_col]
    if fallback_text_col and fallback_text_col in df.columns and fallback_text_col not in work_cols:
        work_cols.append(fallback_text_col)

    work = df[work_cols].copy()
    work[uid_col] = work[uid_col].map(lambda x: _safe_str(x, ""))
    work = work[work[uid_col] != ""]

    primary = work[text_col].map(lambda x: _safe_str(x, ""))
    if fallback_text_col and fallback_text_col in work.columns:
        fallback = work[fallback_text_col].map(lambda x: _safe_str(x, ""))
        text = primary.where(primary.str.len() > 0, fallback)
    else:
        text = primary

    work["__text__"] = text.map(lambda x: x[:max_chars] if max_chars and len(x) > max_chars else x)
    work["__len__"] = work["__text__"].str.len()

    # Use the longest available description per user. This is deterministic and
    # robust when rows from the same user contain duplicate or partially empty descriptions.
    work = work.sort_values([uid_col, "__len__"], ascending=[True, False])
    user_df = work.drop_duplicates(uid_col, keep="first")[[uid_col, "__text__"]]
    user_df = user_df.sort_values(uid_col).reset_index(drop=True)

    uids = user_df[uid_col].astype(str).tolist()
    texts = user_df["__text__"].astype(str).tolist()
    return uids, texts


def build_user_desc_embedding_cache(
    df: pd.DataFrame,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    uid_col: str = "Uid",
    text_col: str = "user_description_clean",
    fallback_text_col: str | None = "user_description",
    cache_dir: str | Path = "data/cache/user_desc_embeddings",
    cache_name: str | None = None,
    batch_size: int = 128,
    normalize_embeddings: bool = True,
    device: str | None = None,
    max_chars: int = 1024,
    force_rebuild: bool = False,
    project_root: str | Path | None = None,
) -> Tuple[Path, Path]:
    """
    Build or reuse cached user_description sentence embeddings.

    Returns:
        emb_path: .npy path, shape [num_users, dim]
        idx_path: .json path, maps uid string -> row index in emb array
    """
    cache_dir = _resolve_path(cache_dir, project_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_cache_name(model_name, cache_name)
    emb_path = cache_dir / f"{safe_name}.npy"
    idx_path = cache_dir / f"{safe_name}_uid_index.json"
    meta_path = cache_dir / f"{safe_name}_meta.json"

    if emb_path.exists() and idx_path.exists() and not force_rebuild:
        print(f"[USER_DESC_EMB] reuse cache: {emb_path}")
        print(f"[USER_DESC_EMB] reuse index: {idx_path}")
        return emb_path, idx_path

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError(
            "user_desc_encoder.enabled=true requires sentence-transformers.\n"
            "Install it with: pip install sentence-transformers"
        ) from e

    uids, texts = _select_user_texts(
        df=df,
        uid_col=uid_col,
        text_col=text_col,
        fallback_text_col=fallback_text_col,
        max_chars=max_chars,
    )

    non_empty = sum(1 for t in texts if t.strip())
    print(
        "[USER_DESC_EMB] building cache | "
        f"model={model_name} | users={len(uids)} | non_empty={non_empty}/{len(uids)} | "
        f"batch_size={batch_size} | normalize={normalize_embeddings}"
    )

    model_kwargs = {}
    if device:
        model_kwargs["device"] = device
    model = SentenceTransformer(model_name, **model_kwargs)

    emb = model.encode(
        texts,
        batch_size=int(batch_size),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=bool(normalize_embeddings),
    ).astype(np.float32)

    uid_index = {str(uid): int(i) for i, uid in enumerate(uids)}

    np.save(str(emb_path), emb)
    idx_path.write_text(json.dumps(uid_index, ensure_ascii=False), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "uid_col": uid_col,
                "text_col": text_col,
                "fallback_text_col": fallback_text_col,
                "num_users": len(uids),
                "non_empty_texts": non_empty,
                "embedding_shape": list(emb.shape),
                "normalize_embeddings": bool(normalize_embeddings),
                "max_chars": int(max_chars),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[USER_DESC_EMB] saved embeddings: {emb_path} shape={emb.shape}")
    print(f"[USER_DESC_EMB] saved uid index: {idx_path}")
    return emb_path, idx_path


def maybe_prepare_user_desc_embeddings(
    cfg: Dict[str, Any],
    df: pd.DataFrame,
    project_root: str | Path | None = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    Read cfg['user_desc_encoder']; if enabled, build/reuse a HF cache.
    Otherwise return cfg['data'].user_desc_emb_path / user_desc_idx_path.
    """
    data_cfg = cfg.get("data", {}) or {}
    enc_cfg = cfg.get("user_desc_encoder", {}) or {}

    if not bool(enc_cfg.get("enabled", False)):
        return data_cfg.get("user_desc_emb_path"), data_cfg.get("user_desc_idx_path")

    model_name = str(enc_cfg.get("model_name", "sentence-transformers/all-mpnet-base-v2"))
    emb_path, idx_path = build_user_desc_embedding_cache(
        df=df,
        model_name=model_name,
        uid_col=str(enc_cfg.get("uid_col", "Uid")),
        text_col=str(enc_cfg.get("text_col", "user_description_clean")),
        fallback_text_col=enc_cfg.get("fallback_text_col", "user_description"),
        cache_dir=enc_cfg.get("cache_dir", "data/cache/user_desc_embeddings"),
        cache_name=enc_cfg.get("cache_name", None),
        batch_size=int(enc_cfg.get("batch_size", 128)),
        normalize_embeddings=bool(enc_cfg.get("normalize_embeddings", True)),
        device=enc_cfg.get("device", None),
        max_chars=int(enc_cfg.get("max_chars", 1024)),
        force_rebuild=bool(enc_cfg.get("force_rebuild", False)),
        project_root=project_root,
    )

    # Keep cfg internally consistent for code that reads cfg['data'] later.
    data_cfg["user_desc_emb_path"] = str(emb_path)
    data_cfg["user_desc_idx_path"] = str(idx_path)
    cfg["data"] = data_cfg

    return str(emb_path), str(idx_path)
