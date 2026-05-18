from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

import torch

from src.models.fasttext_tag_encoder import FastTextTagEncoder
from src.utils.criterion import HybridLoss, PairwiseRankingLoss


def build_loss(loss_cfg: Dict[str, Any]):
    name = str(loss_cfg.get("name", "hybrid")).lower()

    if name == "mse":
        return torch.nn.MSELoss()
    if name in {"mae", "l1"}:
        return torch.nn.L1Loss()
    if name == "smoothl1":
        return torch.nn.SmoothL1Loss()
    if name == "ranking":
        max_weight = loss_cfg.get("max_weight", loss_cfg.get("rank_max_weight", 3.0))
        if max_weight is not None:
            max_weight = float(max_weight)
        return PairwiseRankingLoss(
            margin=float(loss_cfg.get("margin", loss_cfg.get("rank_margin", 0.0))),
            min_target_diff=float(loss_cfg.get("min_target_diff", 0.1)),
            weight_by_target_diff=bool(loss_cfg.get("weight_by_target_diff", True)),
            max_weight=max_weight,
            max_pairs=loss_cfg.get("rank_max_pairs", None),
        )
    if name == "hybrid":
        return HybridLoss(
            alpha=float(loss_cfg.get("alpha", 1.0)),
            beta=float(loss_cfg.get("beta", 0.45)),
            gamma=float(loss_cfg.get("gamma", 0.15)),
            delta=float(loss_cfg.get("delta", 0.10)),
            eta=float(loss_cfg.get("eta", 0.08)),
            zeta=float(loss_cfg.get("zeta", 0.20)),
            hard_scale=float(loss_cfg.get("hard_scale", 1.2)),
            high_target_scale=float(loss_cfg.get("high_target_scale", 0.25)),
            reg_max_weight=float(loss_cfg.get("reg_max_weight", 4.0)),
            min_target_diff=float(loss_cfg.get("min_target_diff", 0.15)),
            rank_max_weight=float(loss_cfg.get("rank_max_weight", 4.0)),
            rank_max_pairs=loss_cfg.get("rank_max_pairs", None),
            contrast_max_pairs=loss_cfg.get("contrast_max_pairs", None),
            variance_floor_ratio=float(loss_cfg.get("variance_floor_ratio", 0.60)),
            focal_gamma=float(loss_cfg.get("focal_gamma", 1.5)),
        )

    raise ValueError(f"Unsupported loss: {name}")


def snapshot_python_code(project_root: Path, exp_dir: Path) -> None:
    snapshot_dir = exp_dir / "code_snapshot"

    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for folder_name in ["src", "scripts"]:
        src_root = project_root / folder_name
        if not src_root.exists():
            print(f"[CODE SNAPSHOT] skip missing folder: {src_root}")
            continue

        for py_file in src_root.rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue
            rel_path = py_file.relative_to(project_root)
            dst = snapshot_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(py_file, dst)
            copied += 1

    print(f"[CODE SNAPSHOT] saved {copied} .py files to: {snapshot_dir}")


def build_fasttext_tag_encoder(
    tag_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    use_tag_embedding: bool,
    hidden_dim: int,
    dropout: float,
):
    if not use_tag_embedding:
        print("[TAG_EMB] disabled by config: tag_embedding.use=false")
        return None

    cache_path = (
        tag_cfg.get("cache_path")
        or tag_cfg.get("fasttext_cache_path")
        or data_cfg.get("fasttext_tag_cache_path")
    )
    vec_path = (
        tag_cfg.get("vec_path")
        or tag_cfg.get("fasttext_vec_path")
        or data_cfg.get("fasttext_vec_path")
    )

    max_vectors = tag_cfg.get("max_vectors", 300000)
    if max_vectors is not None:
        max_vectors = int(max_vectors)

    token_to_idx = None
    embedding_matrix = None

    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            token_to_idx, embedding_matrix = FastTextTagEncoder.load_cache(cache_path)
            print(f"[TAG_EMB] loaded fastText tag cache: {cache_path}")
        elif not vec_path:
            print(f"[TAG_EMB] cache_path not found, will build/download: {cache_path}")

    if token_to_idx is None and vec_path:
        vec_path = Path(vec_path)
        if not vec_path.exists():
            raise FileNotFoundError(f"fastText vec_path not found: {vec_path}")

        token_to_idx, embedding_matrix = FastTextTagEncoder.load_vec_txt(
            vec_path,
            max_vectors=max_vectors,
            lowercase=True,
            add_special_tokens=True,
        )
        print(
            f"[TAG_EMB] loaded fastText vectors: {vec_path} | "
            f"max_vectors={max_vectors}"
        )

        if cache_path:
            FastTextTagEncoder.save_cache(token_to_idx, embedding_matrix, cache_path)
            print(f"[TAG_EMB] saved cache: {cache_path}")

    if token_to_idx is None:
        raise FileNotFoundError(
            "[TAG_EMB] tag_embedding.use=true but no usable fastText vectors/cache were found.\n"
            f"  cache_path={cache_path}\n"
            f"  vec_path={vec_path}\n\n"
            "Fix one of these:\n"
            "  1) Set tag_embedding.vec_path to your local wiki-news-300d-1M.vec\n"
            "  2) Set tag_embedding.cache_path to an existing fastText cache .pt\n"
            "  3) Set tag_embedding.use: false\n\n"
            "This version intentionally does NOT use torchtext because torchtext often "
            "breaks with PyTorch ABI/version mismatches."
        )

    tag_encoder = FastTextTagEncoder(
        token_to_idx=token_to_idx,
        embedding_matrix=embedding_matrix,
        output_dim=hidden_dim,
        dropout=dropout,
        trainable=bool(tag_cfg.get("trainable", False)),
        normalize_output=bool(tag_cfg.get("normalize_output", False)),
    )

    print(
        "[TAG_EMB] enabled | "
        f"vocab={len(token_to_idx)} | "
        f"embed_dim={tag_encoder.embed_dim} | "
        f"output_dim={tag_encoder.output_dim} | "
        f"trainable={bool(tag_cfg.get('trainable', False))}"
    )

    return tag_encoder
