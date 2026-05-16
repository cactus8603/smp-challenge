from __future__ import annotations

import argparse
import json
import random
import sys
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# Allow scripts/ to import from src/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.datasets.metadata_preprocessor import MetadataPreprocessor
from src.datasets.smp_dataset import SMPDataset, smp_collate_fn
from src.engine.trainer import Trainer
from src.models.fusion_model import SMPFusionModel
from src.models.fasttext_tag_encoder import FastTextTagEncoder
from src.utils.criterion import HybridLoss
from src.utils.user_desc_embeddings import maybe_prepare_user_desc_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Which fold to train. Example: 0 ~ n_folds-1",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of GroupKFold splits.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping.")
    return data


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge_dict(merged[k], v)
        else:
            merged[k] = deepcopy(v)
    return merged


def load_config_with_base(config_path: Path) -> Dict[str, Any]:
    cfg = load_yaml(config_path)

    base_key = cfg.pop("base", None)
    if base_key is None:
        return cfg

    base_path = Path(base_key)
    if not base_path.is_absolute():
        base_path = (config_path.parent / base_key).resolve()

    base_cfg = load_config_with_base(base_path)
    return deep_merge_dict(base_cfg, cfg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loss(loss_cfg: Dict[str, Any]):
    name = str(loss_cfg.get("name", "hybrid")).lower()

    if name == "mse":
        return torch.nn.MSELoss()
    elif name in {"mae", "l1"}:
        return torch.nn.L1Loss()
    elif name == "smoothl1":
        return torch.nn.SmoothL1Loss()
    elif name == "ranking":
        return HybridLoss(
            alpha=float(loss_cfg.get("alpha", 1.0)),
            beta=float(loss_cfg.get("beta", 0.3)),
            margin=float(loss_cfg.get("margin", 0.0)),
            min_target_diff=float(loss_cfg.get("min_target_diff", 0.1)),
            weight_by_target_diff=bool(loss_cfg.get("weight_by_target_diff", True)),
            max_weight=float(loss_cfg.get("max_weight", loss_cfg.get("rank_max_weight", 3.0))),
        )
    elif name == "hybrid":
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

            variance_floor_ratio=float(loss_cfg.get("variance_floor_ratio", 0.60)),
            focal_gamma=float(loss_cfg.get("focal_gamma", 1.5)),
        )
    else:
        raise ValueError(f"Unsupported loss: {name}")

def snapshot_python_code(project_root: Path, exp_dir: Path) -> None:
    """
    Save a source-code snapshot for reproducibility.

    Copies only .py files from:
      - project_root/src/**/*.py
      - project_root/scripts/**/*.py

    into:
      exp_dir/code_snapshot/

    Existing code_snapshot is removed first so repeated runs do not keep stale files.
    """
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
    """
    Build optional fastText tag encoder.

    Priority:
    1. tag_embedding.cache_path / data.fasttext_tag_cache_path
    2. tag_embedding.vec_path / data.fasttext_vec_path
    3. online pretrained fastText via torchtext.vocab.FastText

    Config examples:

    Existing cache:
      tag_embedding:
        use: true
        cache_path: /local/smp/fasttext_tag_cache_en.pt

    Local .vec:
      tag_embedding:
        use: true
        vec_path: /local/smp/wiki-news-300d-1M.vec
        cache_path: /local/smp/fasttext_tag_cache_en.pt
        max_vectors: 300000

    Online pretrained:
      tag_embedding:
        use: true
        source: torchtext
        language: en
        cache_dir: /local/smp/torchtext_fasttext
        cache_path: /local/smp/torchtext_fasttext/fasttext_en_cache.pt
        max_vectors: 300000

    Notes:
    - This encoder consumes batch["tag_tokens"], which are already tokenized
      keyword/tag tokens from SMPDataset.
    - It does NOT use tag. The old field name `tag_tokens` is reused only
      because the dataset already provides that token list.
    """
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

    # ---------------------------------------------------------
    # 1. Load existing cache
    # ---------------------------------------------------------
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            token_to_idx, embedding_matrix = FastTextTagEncoder.load_cache(cache_path)
            print(f"[TAG_EMB] loaded fastText tag cache: {cache_path}")
        elif vec_path:
            # Build from local .vec below, then save cache.
            pass
        else:
            # No local cache yet; fall back to online pretrained.
            print(f"[TAG_EMB] cache_path not found, will build/download: {cache_path}")

    # ---------------------------------------------------------
    # 2. Build from local .vec if provided
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 3. No torchtext fallback
    # ---------------------------------------------------------
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

def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".jsonl":
        return pd.read_json(p, lines=True)

    raise ValueError(f"Unsupported data file format: {p.suffix}")


def make_fold_split(
    df: pd.DataFrame,
    fold: int,
    n_folds: int,
    group_col: str = "Uid",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in dataframe.")

    if "label" not in df.columns:
        raise ValueError("official_train dataframe must contain 'label' column.")

    groups = df[group_col].fillna("UNK_GROUP").astype(str).to_numpy()
    gkf = GroupKFold(n_splits=n_folds)

    splits = list(gkf.split(df, groups=groups))
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold must be in [0, {len(splits) - 1}], got {fold}")

    train_idx, val_idx = splits[fold]
    train_df = df.iloc[train_idx].reset_index(drop=True).copy()
    val_df = df.iloc[val_idx].reset_index(drop=True).copy()

    train_df["split"] = "train"
    val_df["split"] = "val"

    return train_df, val_df


def _safe_nunique(series: pd.Series) -> int:
    return int(series.dropna().nunique())


def add_user_aggregate_features_fold(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build label-derived user aggregate features.

    Two modes:
    - target_df IS train_df → use leave-one-out aggregation to avoid label leakage
      (each row sees the user statistics computed from all OTHER rows of the same user)
    - target_df IS val_df / test_df → use standard groupby aggregation from train_df

    Without LOO, train rows would see user_mean_label that includes their own label,
    causing massive train/val gap (train ≈ 1.0, val ≈ 0.37).
    """
    if train_df.empty or "Uid" not in train_df.columns or "Uid" not in target_df.columns:
        return target_df

    work = train_df.copy()
    work["label"] = pd.to_numeric(work["label"], errors="coerce")

    if "hour" in work.columns:
        work["hour"] = pd.to_numeric(work["hour"], errors="coerce")
    else:
        work["hour"] = np.nan

    is_train_self = (target_df is train_df) or (
        len(target_df) == len(train_df)
        and "post_id" in target_df.columns
        and "post_id" in train_df.columns
        and target_df["post_id"].equals(train_df["post_id"])
    )

    agg_cols_to_drop = [
        "user_prev_post_count",
        "user_mean_label",
        "user_median_label",
        "user_std_label",
        "user_category_nunique",
        "user_active_hour_mean",
    ]

    if is_train_self:
        # ── LEAVE-ONE-OUT for train ──────────────────────────────
        # For each row i, compute aggregates over the user's OTHER rows.
        out = train_df.drop(
            columns=[c for c in agg_cols_to_drop if c in train_df.columns],
            errors="ignore",
        ).copy()

        # Compute per-user totals first
        grp = work.groupby("Uid", dropna=True)
        user_count = grp["post_id"].transform("count")
        label_sum  = grp["label"].transform("sum")
        label_count_nonnull = grp["label"].transform(lambda s: s.notna().sum())
        hour_sum   = grp["hour"].transform("sum")
        hour_count_nonnull = grp["hour"].transform(lambda s: s.notna().sum())

        # Leave-one-out mean: (sum - self) / (count - 1)
        own_label = work["label"]
        own_hour  = work["hour"]

        loo_count = (user_count - 1).clip(lower=0)
        out["user_prev_post_count"] = loo_count.astype(np.float32)

        # label mean: subtract own label if non-null
        label_sum_loo = label_sum - own_label.fillna(0.0)
        label_count_loo = (label_count_nonnull - own_label.notna().astype(int)).clip(lower=0)
        out["user_mean_label"] = np.where(
            label_count_loo > 0,
            label_sum_loo / label_count_loo.replace(0, np.nan),
            np.nan,
        )

        # hour mean: subtract own hour if non-null
        hour_sum_loo = hour_sum - own_hour.fillna(0.0)
        hour_count_loo = (hour_count_nonnull - own_hour.notna().astype(int)).clip(lower=0)
        out["user_active_hour_mean"] = np.where(
            hour_count_loo > 0,
            hour_sum_loo / hour_count_loo.replace(0, np.nan),
            np.nan,
        )

        # median / std / nunique: not easy to LOO-rewrite efficiently.
        # Use full-user values but they're weaker signals than mean; less leakage risk.
        median_full = grp["label"].transform("median")
        std_full    = grp["label"].transform("std")
        cat_nunique = (
            grp["category"].transform(lambda s: s.dropna().nunique())
            if "category" in work.columns else np.nan
        )
        out["user_median_label"]    = median_full.astype(np.float32)
        out["user_std_label"]       = std_full.astype(np.float32)
        out["user_category_nunique"] = pd.to_numeric(cat_nunique, errors="coerce").astype(np.float32)

        return out

    # ── Standard aggregation for val / test ─────────────────────
    agg = (
        work.groupby("Uid", dropna=True)
        .agg(
            user_prev_post_count=("post_id", "count"),
            user_mean_label=("label", "mean"),
            user_median_label=("label", "median"),
            user_std_label=("label", "std"),
            user_category_nunique=("category", _safe_nunique),
            user_active_hour_mean=("hour", "mean"),
        )
        .reset_index()
    )

    agg_cols = [c for c in agg.columns if c != "Uid"]
    out = target_df.drop(
        columns=[c for c in agg_cols if c in target_df.columns],
        errors="ignore",
    )
    out = out.merge(agg, on="Uid", how="left")
    return out


def ensure_user_aggregate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure aggregate columns exist even if they are absent in official_train.
    This keeps schema stable for downstream preprocessing.
    """
    out = df.copy()
    required_cols = [
        "user_prev_post_count",
        "user_mean_label",
        "user_median_label",
        "user_std_label",
        "user_category_nunique",
        "user_active_hour_mean",
    ]
    for c in required_cols:
        if c not in out.columns:
            out[c] = None
    return out




# ---------------------------------------------------------------------
# Location text parsing
# ---------------------------------------------------------------------
_LOCATION_MISSING_VALUES = {"", "nan", "none", "null", "unknown", "unk", "0", "0.0"}


def _clean_location_component(x: Any) -> str:
    """Normalize one location component from raw location_text/city/state/country."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if not s:
        return ""
    s = " ".join(s.split())
    if s.lower() in _LOCATION_MISSING_VALUES:
        return ""
    # Light canonicalization for common country aliases.
    alias = {
        "usa": "United States",
        "u.s.a.": "United States",
        "u.s.a": "United States",
        "us": "United States",
        "u.s.": "United States",
        "united states of america": "United States",
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
        "brasil": "Brazil",
    }
    return alias.get(s.lower(), s)


def _is_missing_location_value(x: Any) -> bool:
    return _clean_location_component(x) == ""


def _parse_location_text_value(x: Any) -> tuple[str, str, str]:
    """
    Heuristic parser for Flickr-style location_text.

    Examples:
      "London, United Kingdom" -> city=London, country=United Kingdom
      "Buffalo, New York, United States of America" -> city=Buffalo, state=New York, country=United States
      "France" -> country=France
    """
    s = _clean_location_component(x)
    if not s:
        return "", "", ""

    parts = [_clean_location_component(p) for p in s.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return "", "", ""

    if len(parts) == 1:
        return "", "", parts[0]
    if len(parts) == 2:
        return parts[0], "", parts[1]

    city = parts[0]
    state = ", ".join(parts[1:-1])
    country = parts[-1]
    return city, state, country


def enrich_location_from_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill empty city/state/country from location_text before geo features are built.

    The processed parquet currently has useful location_text but empty city/state/country.
    This function makes those fields usable for categorical features and fold-safe
    target/frequency encoding. Existing non-empty city/state/country values are kept.
    """
    out = df.copy()
    if "location_text" not in out.columns:
        out["location_text"] = ""
    for col in ["city", "state", "country"]:
        if col not in out.columns:
            out[col] = ""

    parsed = out["location_text"].map(_parse_location_text_value)
    parsed_city = parsed.map(lambda t: t[0])
    parsed_state = parsed.map(lambda t: t[1])
    parsed_country = parsed.map(lambda t: t[2])

    for col, parsed_col in [
        ("city", parsed_city),
        ("state", parsed_state),
        ("country", parsed_country),
    ]:
        missing = out[col].map(_is_missing_location_value)
        out.loc[missing, col] = parsed_col[missing]
        out[col] = out[col].map(_clean_location_component)

    # Compact debug summary; useful in logs/check script.
    for col in ["location_text", "city", "state", "country"]:
        non_empty = int(out[col].fillna("").astype(str).str.strip().ne("").sum())
        print(f"[GEO] {col}: non_empty={non_empty}/{len(out)}")

    return out


def add_geo_encoding_features(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    smoothing: float = 30.0,
) -> pd.DataFrame:
    """
    Add fold-safe geo/location target encoding and frequency features.

    IMPORTANT:
    - For validation/test target_df: statistics are computed from train_df only.
    - For training target_df == train_df: target encodings are leave-one-out,
      so each row does NOT see its own label.

    This prevents train-only target encoding leakage while keeping the useful
    location priors for validation/test.
    """
    import numpy as np

    out = target_df.copy()
    if "label" not in train_df.columns:
        raise ValueError("add_geo_encoding_features requires train_df['label'].")

    train_label = pd.to_numeric(train_df["label"], errors="coerce")
    global_mean = float(train_label.mean()) if train_label.notna().any() else 0.0
    smoothing = float(smoothing)

    # Is target_df the same fold-train frame? If yes, use leave-one-out.
    is_train_self = (target_df is train_df) or (
        len(target_df) == len(train_df)
        and "post_id" in target_df.columns
        and "post_id" in train_df.columns
        and target_df["post_id"].equals(train_df["post_id"])
    )

    for col in ["country", "state", "city", "location_text"]:
        enc_col = f"{col}_target_enc"
        freq_col = f"{col}_freq"

        if col not in train_df.columns or col not in out.columns:
            out[enc_col] = np.float32(global_mean)
            out[freq_col] = np.float32(0.0)
            continue

        work = train_df[[col]].copy()
        work["_label"] = train_label

        # Frequency is always computed from fold-train only.
        freq = work[col].value_counts(dropna=True).rename(freq_col)
        out[freq_col] = np.log1p(out[col].map(freq).fillna(0)).astype(np.float32)

        if is_train_self:
            # Leave-one-out smoothed target encoding for train rows:
            #   (group_sum - own_label + global_mean * smoothing)
            # / (group_count - self_count + smoothing)
            grp_sum = work.groupby(col, dropna=True)["_label"].transform("sum")
            grp_count = work.groupby(col, dropna=True)["_label"].transform("count")

            own_label = train_label.fillna(0.0)
            own_count = train_label.notna().astype(int)

            loo_sum = grp_sum - own_label
            loo_count = (grp_count - own_count).clip(lower=0)

            loo_enc = (loo_sum + global_mean * smoothing) / (loo_count + smoothing)
            out[enc_col] = pd.to_numeric(loo_enc, errors="coerce").fillna(global_mean).astype(np.float32)
        else:
            # Smoothed target encoding for val/test using fold-train only.
            stats = (
                work.groupby(col, dropna=True)["_label"]
                .agg(["sum", "count"])
            )
            smoothed = (
                (stats["sum"] + global_mean * smoothing)
                / (stats["count"] + smoothing)
            )
            out[enc_col] = out[col].map(smoothed).fillna(global_mean).astype(np.float32)

    return out


def main():
    args = parse_args()
    cfg = load_config_with_base(Path(args.config).resolve())

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # paths / experiment dirs
    # -------------------------
    exp_name = str(cfg["exp_name"])
    output_root = Path(cfg["output"]["root_dir"])

    fold_name = f"fold_{args.fold}"
    exp_dir = output_root / exp_name / fold_name
    ckpt_dir = exp_dir / "checkpoints"
    tb_dir = exp_dir / "tensorboard"
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg = deepcopy(cfg)
    resolved_cfg["runtime"] = {
        "fold": args.fold,
        "n_folds": args.n_folds,
    }

    with (exp_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(resolved_cfg, f, ensure_ascii=False, indent=2)

    # -------------------------
    # config shortcuts
    # -------------------------
    model_cfg = cfg["model"]
    text_cfg = cfg["text"]
    image_cfg = cfg["image"]
    meta_cfg = cfg["meta"]
    train_cfg = cfg["train"]
    preprocess_cfg = cfg["preprocess"]
    fusion_cfg = cfg["fusion"]
    loss_cfg = cfg["loss"]
    data_cfg = cfg["data"]
    tag_embedding_cfg = cfg.get("tag_embedding", {}) or {}
    code_snapshot_cfg = cfg.get("code_snapshot", {}) or {}
    
    # user_desc paths are prepared below from cfg.user_desc_encoder.
    user_desc_emb_path = None
    user_desc_idx_path = None
    loc_desc_emb_path = data_cfg.get("loc_desc_emb_path")
    loc_desc_idx_path = data_cfg.get("loc_desc_idx_path")

    use_text = bool(model_cfg["use_text"])
    use_meta = bool(model_cfg["use_meta"])
    use_image = bool(model_cfg["use_image"])
    use_tag_embedding = bool(tag_embedding_cfg.get("use", tag_embedding_cfg.get("enabled", False)))

    text_model_name = text_cfg["model_name"]
    image_model_name = image_cfg["model_name"]
    image_path_col = image_cfg.get("path_col", "image_path")

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"])
    pin_memory = bool(train_cfg.get("pin_memory", True))
    persistent_workers = bool(train_cfg.get("persistent_workers", num_workers > 0))
    drop_last = bool(train_cfg.get("drop_last", False))

    if bool(code_snapshot_cfg.get("enabled", True)):
        snapshot_python_code(ROOT, exp_dir)

    # -------------------------
    # data loading: KFold from official_train only
    # -------------------------
    official_train_path = data_cfg.get("official_train_path")
    if official_train_path is None:
        raise ValueError(
            "For KFold training, config.data.official_train_path is required."
        )

    official_train_df = load_dataframe(official_train_path)

    # Optional reproducible Hugging Face / sentence-transformers user_description embeddings.
    # If cfg.user_desc_encoder.enabled=true, this builds/reuses a cache and updates
    # user_desc_emb_path / user_desc_idx_path for SMPDataset below.
    user_desc_emb_path, user_desc_idx_path = maybe_prepare_user_desc_embeddings(
        cfg=cfg,
        df=official_train_df,
        project_root=ROOT,
    )
    data_cfg = cfg["data"]

    if "Uid" not in official_train_df.columns:
        raise ValueError("official_train dataframe must contain 'Uid' for GroupKFold.")

    # make sure missing aggregate cols exist; they will be recomputed after fold split
    official_train_df = ensure_user_aggregate_columns(official_train_df)

    train_df, val_df = make_fold_split(
        official_train_df,
        fold=args.fold,
        n_folds=args.n_folds,
        group_col="Uid",
    )

    # Fill empty city/state/country from location_text before geo encoding.
    train_df = enrich_location_from_text(train_df)
    val_df = enrich_location_from_text(val_df)

    # ---------------------------------------------------------
    # IMPORTANT: recompute label-derived user aggregates
    # using ONLY fold-train, then apply to train/val
    # ---------------------------------------------------------
    train_df = add_user_aggregate_features_fold(train_df, train_df)
    val_df = add_user_aggregate_features_fold(train_df, val_df)

    # geo target encoding + frequency (computed from fold-train, applied to train/val)
    # Train rows use leave-one-out target encoding to avoid seeing their own label.
    target_encoding_smoothing = float(preprocess_cfg.get("target_encoding_smoothing", 30.0))
    train_df = add_geo_encoding_features(train_df, train_df, smoothing=target_encoding_smoothing)
    val_df   = add_geo_encoding_features(train_df, val_df, smoothing=target_encoding_smoothing)

    print(f"[INFO] Fold {args.fold}/{args.n_folds}")
    print(f"[INFO] Train rows: {len(train_df)} | Val rows: {len(val_df)}")
    print(f"[INFO] Train users: {train_df['Uid'].nunique()} | Val users: {val_df['Uid'].nunique()}")

    label_mean = train_df["label"].mean()
    label_std = train_df["label"].std()

    if use_image:
        if image_path_col not in train_df.columns:
            raise ValueError(f"train_df missing image path column: {image_path_col}")
        if image_path_col not in val_df.columns:
            raise ValueError(f"val_df missing image path column: {image_path_col}")

    # -------------------------
    # preprocessor: fit on fold-train only
    # -------------------------
    preprocessor = MetadataPreprocessor(
        num_cols=preprocess_cfg.get("num_cols"),
        cat_cols=preprocess_cfg.get("cat_cols"),
        bin_cols=preprocess_cfg.get("bin_cols"),
        log1p_cols=preprocess_cfg.get("log1p_cols"),
        normalize_numeric=bool(preprocess_cfg.get("normalize_numeric", True)),
        use_user_desc_lite=bool(preprocess_cfg.get("use_user_desc_lite", True)),
        user_desc_lite_debug=bool(preprocess_cfg.get("user_desc_lite_debug", True)),
        user_desc_lite_max_keywords=int(preprocess_cfg.get("user_desc_lite_max_keywords", 5)),
    )

    train_df = preprocessor.fit_transform(train_df)
    val_df = preprocessor.transform(val_df)

    preprocessor.save(exp_dir / "metadata_preprocessor.json")

    image_root_dir = image_cfg.get("root_dir", None)

    # -------------------------
    # datasets
    # -------------------------
    train_dataset = SMPDataset(
        df=train_df,
        preprocessor=preprocessor,
        text_model_name=text_model_name,
        image_model_name=image_model_name,
        user_desc_emb_path=user_desc_emb_path,
        user_desc_idx_path=user_desc_idx_path,
        # loc_desc_emb_path=loc_desc_emb_path,
        # loc_desc_idx_path=loc_desc_idx_path,
        normalize_label=True,
        label_mean=label_mean,
        label_std=label_std,
        max_length=int(text_cfg["max_length"]),
        use_text=use_text,
        use_meta=use_meta,
        use_caption=bool(text_cfg.get("use_caption", False)),
        caption_max_freq=text_cfg.get("caption_max_freq", 50),
        use_image=use_image,
        image_path_col=image_path_col,
        image_root_dir=image_root_dir,
        is_train=True,
    )

    val_dataset = SMPDataset(
        df=val_df,
        preprocessor=preprocessor,
        text_model_name=text_model_name,
        image_model_name=image_model_name,
        user_desc_emb_path=user_desc_emb_path,
        user_desc_idx_path=user_desc_idx_path,
        # loc_desc_emb_path=loc_desc_emb_path,
        # loc_desc_idx_path=loc_desc_idx_path,
        normalize_label=True,
        label_mean=label_mean,
        label_std=label_std,
        max_length=int(text_cfg["max_length"]),
        use_text=use_text,
        use_meta=use_meta,
        use_caption=bool(text_cfg.get("use_caption", False)),
        caption_max_freq=text_cfg.get("caption_max_freq", 50),
        use_image=use_image,
        image_path_col=image_path_col,
        image_root_dir=image_root_dir,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=smp_collate_fn,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=smp_collate_fn,
        drop_last=False,
    )

    # -------------------------
    # model
    # -------------------------
    cat_cardinalities = [
        int(preprocessor.cat_cardinalities[col])
        for col in preprocessor.cat_cols
    ]

    # -------------------------
    # optional fastText tag encoder
    # -------------------------
    tag_encoder = build_fasttext_tag_encoder(
        tag_cfg=tag_embedding_cfg,
        data_cfg=data_cfg,
        use_tag_embedding=use_tag_embedding,
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
    )

    model = SMPFusionModel(
        text_model_name=text_model_name,
        meta_num_dim=len(preprocessor.transformed_num_cols),
        meta_cat_cardinalities=cat_cardinalities,
        meta_bin_dim=len(preprocessor.transformed_bin_cols),
        meta_num_cols=preprocessor.transformed_num_cols,
        meta_cat_cols=preprocessor.transformed_cat_cols,
        meta_bin_cols=preprocessor.transformed_bin_cols,
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        use_text=use_text,
        use_tag_embedding=use_tag_embedding,
        tag_encoder=tag_encoder,
        use_meta=use_meta,
        use_image=use_image,
        image_model_name=image_model_name,
        text_pooling=text_cfg["pooling"],
        text_trainable=bool(text_cfg["trainable"]),
        image_pretrained=bool(image_cfg.get("pretrained", True)),
        image_trainable=bool(image_cfg["trainable"]),
        fusion_type=str(fusion_cfg.get("type", "pairwise_gated")),
        use_clip_similarity=bool(fusion_cfg.get("use_clip_similarity", True)),
        clip_similarity_mode=str(fusion_cfg.get("clip_similarity_mode", "raw")),
        head_type=str(model_cfg.get("head_type", "moe")),
        use_meta_aux_head=bool(model_cfg.get("use_meta_aux_head", False)),
        meta_aux_scale=float(model_cfg.get("meta_aux_scale", 0.0)),
        use_meta_residual=bool(model_cfg.get("use_meta_residual", True)),
        meta_residual_init=float(model_cfg.get("meta_residual_init", 0.0)),
        meta_residual_dropout=float(model_cfg.get("meta_residual_dropout", 0.0)),
        head_hidden_mult=float(model_cfg.get("head_hidden_mult", 1.0)),
        head_num_layers=int(model_cfg.get("head_num_layers", 2)),
        fusion_num_heads=int(fusion_cfg.get("num_heads", 4)),
        fusion_meta_res_scale=float(fusion_cfg.get("meta_res_scale", 0.75)),
        fusion_text_res_scale=float(fusion_cfg.get("text_res_scale", 0.50)),
        fusion_image_res_scale=float(fusion_cfg.get("image_res_scale", 0.50)),
        fusion_tm_res_scale=float(fusion_cfg.get("tm_res_scale", 0.75)),
        fusion_order=str(fusion_cfg.get("order", "text_meta_first")),
        normalize_image_feature=bool(image_cfg.get("normalize_feature", False)),
        meta_branch_dim=int(meta_cfg["branch_dim"]),
        use_semantic_meta_groups=bool(meta_cfg.get("use_semantic_groups", False)),
        meta_encoder_type=str(meta_cfg.get("encoder_type", "semantic_groups" if meta_cfg.get("use_semantic_groups", False) else "legacy")),
        ft_token_dim=int(meta_cfg.get("ft_token_dim", 192)),
        ft_num_layers=int(meta_cfg.get("ft_num_layers", 3)),
        ft_num_heads=int(meta_cfg.get("ft_num_heads", 8)),
        ft_ffn_mult=int(meta_cfg.get("ft_ffn_mult", 2)),
        ft_dropout=meta_cfg.get("ft_dropout", None),
        use_user_desc=bool(meta_cfg.get("use_user_desc", False)),
        user_desc_dim=int(meta_cfg.get("user_desc_dim", getattr(train_dataset, "user_desc_emb_dim", 0) or 768)),
        use_loc_desc=bool(meta_cfg.get("use_loc_desc", False)),
        loc_desc_dim=int(meta_cfg.get("loc_desc_dim", getattr(train_dataset, "loc_desc_emb_dim", 0) or 400)),
        desc_bottleneck_dim=int(meta_cfg.get("desc_bottleneck_dim", 32)),
        user_desc_scale=float(meta_cfg.get("user_desc_scale", 0.0)),
        loc_desc_scale=float(meta_cfg.get("loc_desc_scale", 0.0))
    ).to(device)

    print(
        f"[TAG_EMB] config.use_tag_embedding={use_tag_embedding} | "
        f"model.use_tag_embedding={getattr(model, 'use_tag_embedding', None)} | "
        f"text_branch_enabled={getattr(model, 'text_branch_enabled', None)} | "
        f"tag_encoder={'yes' if getattr(model, 'tag_encoder', None) is not None else 'no'}"
    )

    # -------------------------
    # optimization
    # -------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    criterion = build_loss(loss_cfg)

    print(f"[CONFIG] text={text_cfg}")
    print(f"[CONFIG] fusion={fusion_cfg}")
    print(f"[CONFIG] model.head_type={model_cfg.get('head_type', 'moe')}")
    print(f"[CONFIG] model.use_meta_aux_head={model_cfg.get('use_meta_aux_head', False)} | meta_aux_scale={model_cfg.get('meta_aux_scale', 0.0)}")
    print(f"[CONFIG] fusion.type={fusion_cfg.get('type', 'pairwise_gated')} | num_heads={fusion_cfg.get('num_heads', 4)}")
    print(f"[CONFIG] meta={meta_cfg}")
    print(f"[CONFIG] preprocess={preprocess_cfg}")
    print(f"[CONFIG] loss={loss_cfg}")
    print(f"[CONFIG] tag_embedding={tag_embedding_cfg}")

    total_steps = len(train_loader) * int(train_cfg["epochs"])
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=str(device),
        exp_name=f"{exp_name}_{fold_name}",
        exp_dir=exp_dir,
        ckpt_dir=ckpt_dir,
        tb_dir=tb_dir,
        grad_clip_norm=train_cfg.get("grad_clip_norm"),
        meta_warmup_epochs=int(train_cfg.get("meta_warmup_epochs", 0)),
    )

    monitor_cfg = cfg.get("monitor", {})

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(train_cfg["epochs"]),
        monitor_gbdt=bool(monitor_cfg.get("gbdt", False)),
        gbdt_interval=int(monitor_cfg.get("gbdt_interval", 1)),
        gbdt_max_train_batches=monitor_cfg.get("gbdt_max_train_batches", None),
        gbdt_max_val_batches=monitor_cfg.get("gbdt_max_val_batches", None),
    )


if __name__ == "__main__":
    main()

# Example:
# python3 scripts/train.py --config configs/text_meta_image_v2.yaml --fold 0 --n_folds 5
# python3 scripts/train.py --config configs/text_meta_image_v2.yaml --fold 1 --n_folds 5
# nohup python3 scripts/train.py --config configs/cross_attention_v2_fix.yaml --fold 0 --n_folds 5 > cross_attention_v2_fix.log 2>&1 &

"""
=== Modality Ablation Summary ===
      full | loss=0.9288 | mae=1.7273 | spearman=0.5963
 mask_text | loss=0.9809 | mae=1.7514 | spearman=0.5506
 mask_meta | loss=1.0550 | mae=1.8767 | spearman=0.4321
mask_image | loss=0.9371 | mae=1.7072 | spearman=0.5688

=== Spearman Drop From Full ===
      text: 0.0458
      meta: 0.1642
     image: 0.0276
"""