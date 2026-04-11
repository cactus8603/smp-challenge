from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


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


@dataclass
class NumericStats:
    median: float
    mean: float
    std: float


class MetadataPreprocessor:
    """
    Paper-style metadata preprocessing for SMP v3 tables.

    Main ideas:
    - categorical: fill missing with "UNK", map to integer ids
    - binary: fill missing with 0
    - numeric: optional log1p for heavy-tail columns, median imputation, z-score normalization
    - user history style columns: context-appropriate defaults
    """

    def __init__(
        self,
        num_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        bin_cols: Optional[List[str]] = None,
        text_cols: Optional[List[str]] = None,
        log1p_cols: Optional[List[str]] = None,
        normalize_numeric: bool = True,
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
        ]
        self.bin_cols = bin_cols or [
            "is_weekend",
            "is_night",
            "is_workhour",
            "has_geo",
            "has_title",
            "has_tags",
            "has_user_description",
            "has_location_description",
            "ispro",
            "canbuypro",
            "ispublic",
        ]
        self.text_cols = text_cols or ["title", "alltags", "full_text"]
        self.log1p_cols = log1p_cols or []
        self.normalize_numeric = normalize_numeric

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
        for c in self.num_cols + self.cat_cols + self.bin_cols + self.text_cols:
            if c not in out.columns:
                out[c] = None
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
