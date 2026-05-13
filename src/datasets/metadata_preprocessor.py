from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

@dataclass
class NumericStats:
    median: float
    mean: float
    std: float

EPS = 1e-8


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any, default: int = 0) -> int:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    try:
        return int(x)
    except Exception:
        return default


def _safe_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    x = str(x).strip()
    return x if x else default


def _has_text_value(x: Any) -> int:
    """Return 1 when a raw text-like value is present and non-empty."""
    return int(bool(_safe_str(x, "")))


def _has_numeric_value(x: Any) -> int:
    """Return 1 when a raw numeric-like value is finite."""
    v = _safe_float(x)
    return int(v is not None and np.isfinite(v))


def _parse_vector(x: Any, expected_dim: int) -> Optional[list]:
    """
    Parse a comma-separated vector string into a list of floats.
    Returns None if parsing fails or value is missing.

    Handles:
        "0.123,0.456,..."    → [0.123, 0.456, ...]
        [0.123, 0.456, ...]  → same (already a list)
        None / NaN / ""      → None
    """
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, (list, np.ndarray)):
        arr = [float(v) for v in x]
        if len(arr) < expected_dim:
            arr += [0.0] * (expected_dim - len(arr))
        return arr[:expected_dim]
    s = str(x).strip()
    if not s or s.lower() in ("none", "nan", "null"):
        return None
    try:
        parts = s.split(",")
        arr = [float(p.strip()) for p in parts if p.strip()]
        if not arr:
            return None
        if len(arr) < expected_dim:
            arr += [0.0] * (expected_dim - len(arr))
        return arr[:expected_dim]
    except Exception:
        return None


class MetadataPreprocessor:
    """
    Paper-style metadata preprocessing for SMP v3 tables.

    Main ideas:
    - categorical: fill missing with "UNK", map to integer ids
    - binary: fill missing with 0
    - numeric: optional log1p for heavy-tail columns, median imputation, z-score normalization
    - user history style columns: context-appropriate defaults
    - has_* presence flags are kept as ordinary binary metadata features.
      They let the metadata encoder learn missingness patterns, but they are
      not used for hard-coded model routing/index lookup.
    """

    def __init__(
        self,
        num_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        bin_cols: Optional[List[str]] = None,
        text_cols: Optional[List[str]] = None,
        log1p_cols: Optional[List[str]] = None,
        normalize_numeric: bool = True,
        user_desc_dim: int = 400,
        loc_desc_dim: int = 400,
    ) -> None:
        self.num_cols = num_cols or [
            "hour",
            "weekday",
            "year",
            "month",
            "day",
            "weekofyear",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
            "latitude",
            "longitude",
            "geoaccuracy",
            "title_len",
            "tags_len",
            "full_text_len",
            "title_word_count",
            "full_text_word_count",
            "tag_count",
            "avg_tag_len",
            "title_digit_ratio",
            "title_upper_ratio",
            "title_punct_ratio",
            "full_text_digit_ratio",
            "full_text_punct_ratio",
            "timezone_offset",
            "photo_count_log1p",
            "follower_count_log1p",
            "following_count_log1p",
            "total_views_log1p",
            "total_favorites_log1p",
            "mean_views_log1p",
            "mean_favorites_log1p",
            "mean_tags_log1p",
            "account_age_days_log1p",
            "camera_age_days_log1p",
            "follower_following_ratio",
            "views_per_photo",
            "favorites_per_photo",
            "user_prev_post_count",
            "user_mean_label",
            "user_active_hour_mean",
            "user_category_nunique",
            # user_description text stats
            "user_desc_len",
            "user_desc_word_count",
            # geo target encoding + frequency (computed in train.py per fold)
            "country_target_enc",
            "city_target_enc",
            "location_text_target_enc",
            "country_freq",
            "city_freq",
            "location_text_freq",
        ]
        self.cat_cols = cat_cols or [
            "category",
            "subcategory",
            "concept",
            "category_subcategory_combo",
            "category_concept_combo",
            "geo_cluster",
            "timezone_id",
            "mediastatus",
            "mediatype",
            # geocode enrichment
            "city",
            "country",
        ]
        # Keep has_* as ordinary binary metadata features.
        # These are useful missingness / availability signals for the metadata
        # encoder.  They should NOT be used through brittle hard-coded index
        # logic such as model.set_bin_col_idx(...).
        self.bin_cols = bin_cols or [
            "is_weekend",
            "is_night",
            "is_workhour",
            "ispro",
            "canbuypro",
            "ispublic",
            "has_geo",
            "has_title",
            "has_tags",
            "has_full_text",
            "has_user_description",
            "has_location_text",
            "has_city",
            "has_country",
            "has_image",
        ]
        self.text_cols = text_cols or ["title", "alltags", "full_text"]
        self.log1p_cols = log1p_cols or []
        self.normalize_numeric = normalize_numeric
        self.user_desc_dim = user_desc_dim
        self.loc_desc_dim  = loc_desc_dim

        self.num_stats: Dict[str, NumericStats] = {}
        self.cat_vocab: Dict[str, Dict[str, int]] = {}
        self.cat_cardinalities: Dict[str, int] = {}
        self.fitted: bool = False

    @property
    def transformed_num_cols(self) -> List[str]:
        return [f"num__{c}" for c in self.num_cols]

    @property
    def transformed_cat_cols(self) -> List[str]:
        return [f"cat__{c}" for c in self.cat_cols]

    @property
    def transformed_bin_cols(self) -> List[str]:
        return [f"bin__{c}" for c in self.bin_cols]

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Ensure raw fields used to derive presence flags exist before we create
        # has_* columns.  This keeps old datasets and newer enriched datasets
        # compatible with the same config.
        raw_needed = [
            "title",
            "alltags",
            "full_text",
            "user_description",
            "location_description",
            "location_text",
            "city",
            "country",
            "latitude",
            "longitude",
            "image_path",
        ]
        for c in raw_needed:
            if c not in out.columns:
                out[c] = None

        # Derive missingness / availability features only when they are part of
        # bin_cols.  If a config explicitly provides a different bin_cols list,
        # we respect that list and create only the requested flags.
        if "has_geo" in self.bin_cols:
            out["has_geo"] = (
                out["latitude"].map(_has_numeric_value).astype(bool)
                & out["longitude"].map(_has_numeric_value).astype(bool)
            ).astype(np.int64)
        if "has_title" in self.bin_cols:
            out["has_title"] = out["title"].map(_has_text_value).astype(np.int64)
        if "has_tags" in self.bin_cols:
            out["has_tags"] = out["alltags"].map(_has_text_value).astype(np.int64)
        if "has_full_text" in self.bin_cols:
            out["has_full_text"] = out["full_text"].map(_has_text_value).astype(np.int64)
        if "has_user_description" in self.bin_cols:
            out["has_user_description"] = out["user_description"].map(_has_text_value).astype(np.int64)
        if "has_location_text" in self.bin_cols:
            out["has_location_text"] = out["location_text"].map(_has_text_value).astype(np.int64)
        if "has_city" in self.bin_cols:
            out["has_city"] = out["city"].map(_has_text_value).astype(np.int64)
        if "has_country" in self.bin_cols:
            out["has_country"] = out["country"].map(_has_text_value).astype(np.int64)
        if "has_image" in self.bin_cols:
            out["has_image"] = out["image_path"].map(_has_text_value).astype(np.int64)

        for c in self.num_cols + self.cat_cols + self.bin_cols + self.text_cols:
            if c not in out.columns:
                out[c] = None

        # Raw description text may still exist for upstream feature engineering,
        # and user_desc embeddings are loaded separately by SMPDataset.
        for c in ("user_description", "location_description"):
            if c not in out.columns:
                out[c] = ""
        if "label" not in out.columns:
            out["label"] = 0.0
        if "post_id" not in out.columns:
            out["post_id"] = ""
        if "Uid" not in out.columns:
            out["Uid"] = ""
        if "Pid" not in out.columns:
            out["Pid"] = ""
        return out

    def fit(self, train_df: pd.DataFrame) -> "MetadataPreprocessor":
        df = self._ensure_columns(train_df)

        for col in self.num_cols:
            s_raw = pd.to_numeric(df[col], errors="coerce")

            median = float(s_raw.median()) if s_raw.notna().any() else 0.0
            mean = float(s_raw.mean()) if s_raw.notna().any() else 0.0
            std = float(s_raw.std(ddof=0)) if s_raw.notna().any() else 1.0
            if not np.isfinite(std) or std < EPS:
                std = 1.0
            self.num_stats[col] = NumericStats(median=median, mean=mean, std=std)

        for col in self.cat_cols:
            values = df[col].map(lambda x: _safe_str(x, "UNK") or "UNK").fillna("UNK")
            uniq = values.astype(str).unique().tolist()
            ordered = ["UNK"] + sorted([u for u in uniq if u != "UNK"])
            vocab = {v: i for i, v in enumerate(ordered)}
            self.cat_vocab[col] = vocab
            self.cat_cardinalities[col] = len(vocab)

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("MetadataPreprocessor must be fitted before transform().")

        out = self._ensure_columns(df)

        for col in self.num_cols:
            s = pd.to_numeric(out[col], errors="coerce")
            stats = self.num_stats[col]

            if col in ("user_prev_post_count", "user_category_nunique"):
                fill_value = 0.0
            elif col == "user_mean_label":
                fill_value = self.num_stats["user_mean_label"].median
            elif col == "user_active_hour_mean":
                fill_value = self.num_stats["user_active_hour_mean"].median
            else:
                fill_value = stats.median

            s = s.fillna(fill_value).astype(np.float32)

            if col in self.log1p_cols:
                s = np.log1p(np.clip(s, a_min=0.0, a_max=None))

            if self.normalize_numeric:
                s = (s - stats.mean) / stats.std

            out[f"num__{col}"] = s.astype(np.float32)

        for col in self.bin_cols:
            s = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int64)
            s = s.clip(lower=0, upper=1)
            out[f"bin__{col}"] = s

        for col in self.cat_cols:
            vocab = self.cat_vocab[col]
            unk_id = vocab.get("UNK", 0)
            values = out[col].map(lambda x: _safe_str(x, "UNK") or "UNK").fillna("UNK")
            mapped = values.map(lambda x: vocab.get(str(x), unk_id))
            out[f"cat__{col}"] = mapped.astype(np.int64)

        for col in self.text_cols + ["post_id", "Uid", "Pid"]:
            out[col] = out[col].map(lambda x: _safe_str(x, ""))

        out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0.0).astype(np.float32)

        # Description embeddings are loaded by SMPDataset from .npy/.json files.
        # Presence flags such as has_user_description remain ordinary binary
        # metadata features and do not control any hard-coded routing here.

        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    def get_feature_info(self) -> Dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("MetadataPreprocessor must be fitted before get_feature_info().")
        return {
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "bin_cols": self.bin_cols,
            "transformed_num_cols": self.transformed_num_cols,
            "transformed_cat_cols": self.transformed_cat_cols,
            "transformed_bin_cols": self.transformed_bin_cols,
            "cat_cardinalities": self.cat_cardinalities,
            "normalize_numeric": self.normalize_numeric,
            "log1p_cols": self.log1p_cols,
        }

    def save(self, path: str | Path) -> None:
        if not self.fitted:
            raise RuntimeError("MetadataPreprocessor must be fitted before save().")
        payload = {
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "bin_cols": self.bin_cols,
            "text_cols": self.text_cols,
            "log1p_cols": self.log1p_cols,
            "normalize_numeric": self.normalize_numeric,
            "user_desc_dim": self.user_desc_dim,
            "loc_desc_dim":  self.loc_desc_dim,
            "num_stats": {k: asdict(v) for k, v in self.num_stats.items()},
            "cat_vocab": self.cat_vocab,
            "cat_cardinalities": self.cat_cardinalities,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MetadataPreprocessor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(
            num_cols=payload["num_cols"],
            cat_cols=payload["cat_cols"],
            bin_cols=payload["bin_cols"],
            text_cols=payload.get("text_cols"),
            log1p_cols=payload.get("log1p_cols"),
            normalize_numeric=payload.get("normalize_numeric", True),
            user_desc_dim=payload.get("user_desc_dim", 400),
            loc_desc_dim=payload.get("loc_desc_dim", 400),
        )
        obj.num_stats = {k: NumericStats(**v) for k, v in payload["num_stats"].items()}
        obj.cat_vocab = {
            k: {str(kk): int(vv) for kk, vv in vocab.items()}
            for k, vocab in payload["cat_vocab"].items()
        }
        obj.cat_cardinalities = {
            str(k): int(v) for k, v in payload.get("cat_cardinalities", {}).items()
        }
        if not obj.cat_cardinalities and obj.cat_vocab:
            obj.cat_cardinalities = {k: len(v) for k, v in obj.cat_vocab.items()}
        obj.fitted = True
        return obj
