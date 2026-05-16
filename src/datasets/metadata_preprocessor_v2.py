from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


EPS = 1e-8

# Columns whose values are mathematically bounded (sin/cos ∈ [-1,1]) or
# already compressed to a fixed range — IQR clipping would be a no-op but
# we skip them explicitly to make intent clear in saved stats.
_DEFAULT_IQR_SKIP = frozenset([
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "month_sin", "month_cos",
    "user_description_sentiment",   # VADER compound: always in [-1, 1]
    "geoaccuracy",                   # discrete 0-16
])


@dataclass
class NumericStats:
    median: float
    mean: float
    std: float
    iqr_lower: float = field(default=-float("inf"))  # pre-clip lower bound
    iqr_upper: float = field(default=float("inf"))   # pre-clip upper bound


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
    return int(bool(_safe_str(x, "")))


def _has_numeric_value(x: Any) -> int:
    v = _safe_float(x)
    return int(v is not None and np.isfinite(v))


class MetadataPreprocessorV2:
    """
    v2 of MetadataPreprocessor.

    Changes over v1:
    - Adds has_timezone to default bin_cols (v3 dataset feature).
    - Adds user_description_sentiment to default num_cols (v3 dataset feature).
    - IQR-based outlier clipping for numeric columns:
        * Bounds are computed on training data in fit() using
          [Q1 - iqr_multiplier*IQR, Q3 + iqr_multiplier*IQR].
        * Values are winsorized (clipped) — no rows are dropped.
        * median / mean / std are computed on the post-clip data so that
          normalization statistics are robust to outliers.
        * Columns in iqr_skip_cols are left untouched (e.g. cyclic features).
    """

    def __init__(
        self,
        num_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        bin_cols: Optional[List[str]] = None,
        text_cols: Optional[List[str]] = None,
        log1p_cols: Optional[List[str]] = None,
        normalize_numeric: bool = True,
        iqr_multiplier: float = 1.5,
        iqr_skip_cols: Optional[List[str]] = None,
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
            # v3 新增
            "user_description_sentiment",
            "user_description_clean_len",
            "user_description_clean_word_count",
            # geo target encoding + frequency (computed in train.py per fold)
            "location_text_target_enc",
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
            # v3 reverse geocoding（has_geo=1 才有值）
            "city",
            "country",
        ]
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
            "has_user_description",
            "has_location_text",
            # v3 新增
            "has_timezone",
            "has_city",
            "has_country",
        ]
        self.text_cols = text_cols or ["title", "alltags", "full_text"]
        self.log1p_cols = log1p_cols or []
        self.normalize_numeric = normalize_numeric
        self.iqr_multiplier = iqr_multiplier
        self.iqr_skip_cols: frozenset = frozenset(iqr_skip_cols) if iqr_skip_cols is not None else _DEFAULT_IQR_SKIP
        self.user_desc_dim = user_desc_dim
        self.loc_desc_dim = loc_desc_dim

        self.num_stats: Dict[str, NumericStats] = {}
        self.cat_vocab: Dict[str, Dict[str, int]] = {}
        self.cat_cardinalities: Dict[str, int] = {}
        self.fitted: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def transformed_num_cols(self) -> List[str]:
        return [f"num__{c}" for c in self.num_cols]

    @property
    def transformed_cat_cols(self) -> List[str]:
        return [f"cat__{c}" for c in self.cat_cols]

    @property
    def transformed_bin_cols(self) -> List[str]:
        return [f"bin__{c}" for c in self.bin_cols]

    @property
    def feature_gate_config(self) -> List[Dict]:
        """
        Returns a list of feature gate groups for MetaEncoder.
        Each group defines one binary flag that gates a set of num/cat features.

        Structure of each group:
            {
              "flag_bin_idx":  int,         # index of the gate flag in meta_bin
              "num_indices":   List[int],   # indices in meta_num to zero when flag=0
              "cat_indices":   List[int],   # indices in meta_cat to force UNK when flag=0
            }
        """
        groups = []

        # ── Geo gate ─────────────────────────────────────────────────────────
        # has_geo=0 → zero out latitude/longitude/geoaccuracy; city/country → UNK
        if "has_geo" in self.bin_cols:
            groups.append({
                "flag_bin_idx": self.bin_cols.index("has_geo"),
                "num_indices": [i for i, c in enumerate(self.num_cols)
                                if c in {"latitude", "longitude", "geoaccuracy"}],
                "cat_indices": [i for i, c in enumerate(self.cat_cols)
                                if c in {"city", "country"}],
            })

        # ── User description gate ─────────────────────────────────────────────
        # has_user_description=0 → zero out sentiment/len/word_count
        if "has_user_description" in self.bin_cols:
            groups.append({
                "flag_bin_idx": self.bin_cols.index("has_user_description"),
                "num_indices": [i for i, c in enumerate(self.num_cols)
                                if c in {"user_description_sentiment",
                                         "user_description_clean_len",
                                         "user_description_clean_word_count"}],
                "cat_indices": [],
            })

        # ── Location text gate ────────────────────────────────────────────────
        # has_location_text=0 → zero out location target encoding / frequency
        if "has_location_text" in self.bin_cols:
            groups.append({
                "flag_bin_idx": self.bin_cols.index("has_location_text"),
                "num_indices": [i for i, c in enumerate(self.num_cols)
                                if c in {"location_text_target_enc",
                                         "location_text_freq"}],
                "cat_indices": [],
            })

        return groups

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        raw_needed = [
            "title", "alltags", "full_text",
            "user_description", "location_description", "location_text",
            "city", "country", "latitude", "longitude", "image_path",
        ]
        for c in raw_needed:
            if c not in out.columns:
                out[c] = None

        # Recompute has_* flags that aren't already in the DataFrame.
        flag_sources = {
            "has_geo": lambda o: (
                o["latitude"].map(_has_numeric_value).astype(bool)
                & o["longitude"].map(_has_numeric_value).astype(bool)
            ).astype(np.int64),
            "has_title":            lambda o: o["title"].map(_has_text_value).astype(np.int64),
            "has_tags":             lambda o: o["alltags"].map(_has_text_value).astype(np.int64),
            "has_user_description": lambda o: o["user_description"].map(_has_text_value).astype(np.int64),
            "has_location_text":    lambda o: o["location_text"].map(_has_text_value).astype(np.int64),
            "has_city":             lambda o: o["city"].map(_has_text_value).astype(np.int64),
            "has_country":          lambda o: o["country"].map(_has_text_value).astype(np.int64),
        }
        for flag, fn in flag_sources.items():
            if flag in self.bin_cols and flag not in out.columns:
                out[flag] = fn(out)

        # has_timezone comes from build_dataset_v3; if absent default to 0.
        if "has_timezone" in self.bin_cols and "has_timezone" not in out.columns:
            out["has_timezone"] = 0

        for c in self.num_cols + self.cat_cols + self.bin_cols + self.text_cols:
            if c not in out.columns:
                out[c] = None

        for c in ("user_description", "location_description"):
            if c not in out.columns:
                out[c] = ""
        for c in ("label", ):
            if c not in out.columns:
                out[c] = 0.0
        for c in ("post_id", "Uid", "Pid"):
            if c not in out.columns:
                out[c] = ""

        return out

    def _compute_iqr_bounds(self, s: pd.Series, col: str) -> Tuple[float, float]:
        """Return (lower, upper) IQR clip bounds for a numeric series.

        Returns (-inf, +inf) when the column is in iqr_skip_cols or when
        the series has fewer than 4 valid values.
        """
        if col in self.iqr_skip_cols:
            return (-float("inf"), float("inf"))
        valid = s.dropna()
        if len(valid) < 4:
            return (-float("inf"), float("inf"))
        q1 = float(valid.quantile(0.25))
        q3 = float(valid.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr
        return (lower, upper)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "MetadataPreprocessorV2":
        df = self._ensure_columns(train_df)

        for col in self.num_cols:
            s_raw = pd.to_numeric(df[col], errors="coerce")

            # Step 1: IQR bounds from raw training data
            lower, upper = self._compute_iqr_bounds(s_raw, col)

            # Step 2: winsorize before computing stats so mean/std are robust
            s_clipped = s_raw.clip(lower=lower, upper=upper)

            median = float(s_clipped.median()) if s_clipped.notna().any() else 0.0
            mean   = float(s_clipped.mean())   if s_clipped.notna().any() else 0.0
            std    = float(s_clipped.std(ddof=0)) if s_clipped.notna().any() else 1.0
            if not np.isfinite(std) or std < EPS:
                std = 1.0

            self.num_stats[col] = NumericStats(
                median=median,
                mean=mean,
                std=std,
                iqr_lower=lower,
                iqr_upper=upper,
            )

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
            raise RuntimeError("MetadataPreprocessorV2 must be fitted before transform().")

        out = self._ensure_columns(df)

        for col in self.num_cols:
            s = pd.to_numeric(out[col], errors="coerce")
            stats = self.num_stats[col]

            # Step 1: IQR winsorize (uses bounds learned from training data)
            if np.isfinite(stats.iqr_lower) or np.isfinite(stats.iqr_upper):
                lo = stats.iqr_lower if np.isfinite(stats.iqr_lower) else None
                hi = stats.iqr_upper if np.isfinite(stats.iqr_upper) else None
                s = s.clip(lower=lo, upper=hi)

            # Step 2: fill NaN with appropriate default
            if col in ("user_prev_post_count", "user_category_nunique"):
                fill_value = 0.0
            elif col == "user_mean_label":
                fill_value = stats.median
            elif col == "user_active_hour_mean":
                fill_value = stats.median
            else:
                fill_value = stats.median

            s = s.fillna(fill_value).astype(np.float32)

            # Step 3: optional log1p
            if col in self.log1p_cols:
                s = np.log1p(np.clip(s, a_min=0.0, a_max=None))

            # Step 4: z-score normalization
            if self.normalize_numeric:
                s = (s - stats.mean) / stats.std

            out[f"num__{col}"] = s.astype(np.float32)

        # 地理特徵遮罩：has_geo=0 時強制歸零，避免 median impute 的值誤導模型
        _GEO_MASKED_COLS = {"latitude", "longitude", "geoaccuracy"}
        if "has_geo" in self.bin_cols:
            geo_mask = pd.to_numeric(out.get("has_geo", 0), errors="coerce").fillna(0).clip(0, 1)
            for col in _GEO_MASKED_COLS:
                key = f"num__{col}"
                if key in out.columns:
                    out[key] = (out[key] * geo_mask).astype(np.float32)

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
            raise RuntimeError("MetadataPreprocessorV2 must be fitted before get_feature_info().")
        return {
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "bin_cols": self.bin_cols,
            "transformed_num_cols": self.transformed_num_cols,
            "transformed_cat_cols": self.transformed_cat_cols,
            "transformed_bin_cols": self.transformed_bin_cols,
            "cat_cardinalities": self.cat_cardinalities,
            "normalize_numeric": self.normalize_numeric,
            "iqr_multiplier": self.iqr_multiplier,
            "iqr_skip_cols": sorted(self.iqr_skip_cols),
            "log1p_cols": self.log1p_cols,
            # IQR bounds per column for inspection
            "iqr_bounds": {
                col: {
                    "lower": self.num_stats[col].iqr_lower,
                    "upper": self.num_stats[col].iqr_upper,
                }
                for col in self.num_cols
            },
        }

    def save(self, path: str | Path) -> None:
        if not self.fitted:
            raise RuntimeError("MetadataPreprocessorV2 must be fitted before save().")

        def _to_json_float(v: float) -> Any:
            """Convert ±inf to strings so JSON doesn't choke."""
            if v == float("inf"):
                return "inf"
            if v == float("-inf"):
                return "-inf"
            return v

        payload = {
            "version": 2,
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "bin_cols": self.bin_cols,
            "text_cols": self.text_cols,
            "log1p_cols": self.log1p_cols,
            "normalize_numeric": self.normalize_numeric,
            "iqr_multiplier": self.iqr_multiplier,
            "iqr_skip_cols": sorted(self.iqr_skip_cols),
            "user_desc_dim": self.user_desc_dim,
            "loc_desc_dim": self.loc_desc_dim,
            "num_stats": {
                k: {
                    "median": v.median,
                    "mean": v.mean,
                    "std": v.std,
                    "iqr_lower": _to_json_float(v.iqr_lower),
                    "iqr_upper": _to_json_float(v.iqr_upper),
                }
                for k, v in self.num_stats.items()
            },
            "cat_vocab": self.cat_vocab,
            "cat_cardinalities": self.cat_cardinalities,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MetadataPreprocessorV2":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))

        def _from_json_float(v: Any) -> float:
            if v == "inf":
                return float("inf")
            if v == "-inf":
                return float("-inf")
            return float(v)

        obj = cls(
            num_cols=payload["num_cols"],
            cat_cols=payload["cat_cols"],
            bin_cols=payload["bin_cols"],
            text_cols=payload.get("text_cols"),
            log1p_cols=payload.get("log1p_cols"),
            normalize_numeric=payload.get("normalize_numeric", True),
            iqr_multiplier=payload.get("iqr_multiplier", 1.5),
            iqr_skip_cols=payload.get("iqr_skip_cols"),
            user_desc_dim=payload.get("user_desc_dim", 400),
            loc_desc_dim=payload.get("loc_desc_dim", 400),
        )
        obj.num_stats = {
            k: NumericStats(
                median=v["median"],
                mean=v["mean"],
                std=v["std"],
                iqr_lower=_from_json_float(v.get("iqr_lower", "-inf")),
                iqr_upper=_from_json_float(v.get("iqr_upper", "inf")),
            )
            for k, v in payload["num_stats"].items()
        }
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
