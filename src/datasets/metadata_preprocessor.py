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

USER_DESC_LITE_DEBUG_PRINTED = False
USER_DESC_LITE_DEBUG_ENABLED = True

USER_DESC_LITE_KEYWORD_GROUPS = {
    "photo": [
        "photo", "photos", "photography", "photographer", "fotograf", "fotografia",
        "portrait", "landscape", "streetphoto", "streetphotography",
    ],
    "travel": [
        "travel", "traveler", "traveller", "trip", "journey", "world", "wander",
        "backpack", "explore", "explorer",
    ],
    "art": [
        "art", "artist", "design", "designer", "creative", "illustration",
        "drawing", "painting", "visual",
    ],
    "nature": [
        "nature", "wildlife", "bird", "birds", "animal", "animals", "flower",
        "forest", "mountain", "ocean", "sea", "landscape",
    ],
    "camera": [
        "camera", "canon", "nikon", "sony", "fuji", "fujifilm", "leica",
        "olympus", "pentax", "lens", "dslr", "mirrorless",
    ],
    "pro": [
        "professional", "freelance", "wedding", "commercial", "studio",
        "portfolio", "available", "booking", "hire",
    ],
    "social": [
        "instagram", "facebook", "twitter", "website", "blog", "contact",
        "email", "www", "http", "https",
    ],
    "location": [
        "taiwan", "japan", "usa", "uk", "france", "germany", "italy",
        "canada", "australia", "london", "tokyo", "paris", "new york",
    ],
}


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


def _word_count(x: Any) -> int:
    s = _safe_str(x, "")
    return len(s.split()) if s else 0


def _normalize_for_keyword_match(x: Any) -> str:
    s = _safe_str(x, "").lower()
    if not s:
        return ""
    # Keep this simple and robust. We do not need heavy NLP here.
    for ch in "\n\r\t,.;:!?()[]{}<>/\\|@#$%^&*_+=~`\"'":
        s = s.replace(ch, " ")
    return " ".join(s.split())


def _extract_user_desc_lite_keywords(x: Any, max_keywords: int = 5) -> tuple[str, dict[str, int]]:
    """
    Small semantic extraction for user_description.

    Returns:
      profile_keywords: compact string, e.g. "photo travel camera"
      flags: dict like {"user_desc_kw_photo": 1, ...}

    This intentionally avoids large 768-d embeddings.
    """
    s = _normalize_for_keyword_match(x)
    flags: dict[str, int] = {}

    if not s:
        for group in USER_DESC_LITE_KEYWORD_GROUPS:
            flags[f"user_desc_kw_{group}"] = 0
        return "", flags

    words = set(s.split())
    matched_groups: list[str] = []

    for group, keywords in USER_DESC_LITE_KEYWORD_GROUPS.items():
        hit = 0
        for kw in keywords:
            kw_norm = kw.lower()
            if " " in kw_norm:
                if kw_norm in s:
                    hit = 1
                    break
            else:
                if kw_norm in words or kw_norm in s:
                    hit = 1
                    break

        flags[f"user_desc_kw_{group}"] = hit
        if hit:
            matched_groups.append(group)

    profile_keywords = " ".join(matched_groups[:max_keywords])
    return profile_keywords, flags


def _derive_user_desc_lite_features(out: pd.DataFrame) -> pd.DataFrame:
    """
    Derive tiny user_description features.

    Features:
      - user_desc_keyword_count        numeric
      - user_desc_profile_keywords     debug / optional text-use column
      - user_desc_kw_*                 binary semantic flags

    Important:
      user_description is read directly from dataframe loaded by train.py
      from config.data.official_train_path. There is no separate file path.
    """
    global USER_DESC_LITE_DEBUG_PRINTED

    if "user_description_clean" in out.columns:
        src = out["user_description_clean"]
    elif "user_description" in out.columns:
        src = out["user_description"]
    else:
        src = pd.Series([""] * len(out), index=out.index)

    extracted = src.map(_extract_user_desc_lite_keywords)
    out["user_desc_profile_keywords"] = extracted.map(lambda t: t[0])
    out["user_desc_keyword_count"] = out["user_desc_profile_keywords"].map(
        lambda x: len(_safe_str(x, "").split())
    ).astype(np.float32)

    for group in USER_DESC_LITE_KEYWORD_GROUPS:
        col = f"user_desc_kw_{group}"
        out[col] = extracted.map(lambda t, c=col: int(t[1].get(c, 0))).astype(np.int64)

    # Debug once per process. This helps verify we are reading the correct raw field.
    if USER_DESC_LITE_DEBUG_ENABLED and not USER_DESC_LITE_DEBUG_PRINTED:
        USER_DESC_LITE_DEBUG_PRINTED = True
        non_empty = int(src.map(_has_text_value).sum())
        print(
            f"[USER_DESC_LITE] source={'user_description_clean' if 'user_description_clean' in out.columns else 'user_description'} "
            f"| non_empty={non_empty}/{len(out)}"
        )
        flag_cols = [f"user_desc_kw_{g}" for g in USER_DESC_LITE_KEYWORD_GROUPS]
        flag_sums = {c: int(out[c].sum()) for c in flag_cols if c in out.columns}
        print(f"[USER_DESC_LITE] flag_sums={flag_sums}")

        sample_mask = src.map(_has_text_value).astype(bool)
        sample_df = out.loc[sample_mask, [c for c in ["user_description", "user_description_clean", "user_desc_profile_keywords", "user_desc_keyword_count"] + flag_cols if c in out.columns]]
        if len(sample_df) > 0:
            sample_df = sample_df.head(8)
            print("[USER_DESC_LITE] debug samples:")
            for i, row in sample_df.iterrows():
                raw = row.get("user_description_clean", row.get("user_description", ""))
                raw = _safe_str(raw, "")
                if len(raw) > 180:
                    raw = raw[:180] + " ..."
                kws = row.get("user_desc_profile_keywords", "")
                active = [c.replace("user_desc_kw_", "") for c in flag_cols if c in row and int(row[c]) == 1]
                print(f"  row={i} | keywords='{kws}' | active={active} | desc='{raw}'")
        else:
            print("[USER_DESC_LITE] no non-empty user_description samples found.")

    return out


def _derive_text_length_features(out: pd.DataFrame) -> pd.DataFrame:
    """
    Derive text length metadata features from existing columns.

    This intentionally works with already-built parquet files:
    - loc_desc_len / loc_desc_word_count are derived from
      location_description_clean, falling back to location_text.
    - user_desc_len / user_desc_word_count are derived from
      user_description_clean, falling back to user_description.
    """
    if "user_description_clean" not in out.columns:
        out["user_description_clean"] = out.get("user_description", "").map(_safe_str) if "user_description" in out.columns else ""
    if "location_description_clean" not in out.columns:
        if "location_text" in out.columns:
            out["location_description_clean"] = out["location_text"].map(_safe_str)
        elif "location_description" in out.columns:
            out["location_description_clean"] = out["location_description"].map(_safe_str)
        else:
            out["location_description_clean"] = ""

    if "user_desc_len" not in out.columns:
        out["user_desc_len"] = out["user_description_clean"].map(lambda x: len(_safe_str(x)))
    if "user_desc_word_count" not in out.columns:
        out["user_desc_word_count"] = out["user_description_clean"].map(_word_count)

    if "loc_desc_len" not in out.columns:
        out["loc_desc_len"] = out["location_description_clean"].map(lambda x: len(_safe_str(x)))
    if "loc_desc_word_count" not in out.columns:
        out["loc_desc_word_count"] = out["location_description_clean"].map(_word_count)

    out = _derive_user_desc_lite_features(out)

    return out


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

    Default geo setup:
    - VERSION A = geo target/frequency encodings + has_city/has_state/has_country
    - VERSION B = VERSION A + city/state/country categorical ids
    Current default in this file is VERSION B.
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
        use_user_desc_lite: bool = True,
        user_desc_lite_debug: bool = True,
        user_desc_lite_max_keywords: int = 5,
    ) -> None:
        # Ensure all col specs are lists regardless of input type (e.g. tuple from YAML)
        if num_cols is not None:
            num_cols = list(num_cols)
        if cat_cols is not None:
            cat_cols = list(cat_cols)
        if bin_cols is not None:
            bin_cols = list(bin_cols)
        if text_cols is not None:
            text_cols = list(text_cols)
        if log1p_cols is not None:
            log1p_cols = list(log1p_cols)
        self.use_user_desc_lite = bool(use_user_desc_lite)
        self.user_desc_lite_debug = bool(user_desc_lite_debug)
        self.user_desc_lite_max_keywords = int(user_desc_lite_max_keywords)

        self.num_cols = num_cols or [
            # ── Time ──────────────────────────────────────────
            "hour", "weekday", "year", "month", "day", "weekofyear",
            "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos",

            # ── Geo numeric ───────────────────────────────────
            "latitude", "longitude", "geoaccuracy",

            # ── Text stats ────────────────────────────────────
            "title_len", "tags_len", "full_text_len",
            "title_word_count", "full_text_word_count", "tag_count", "avg_tag_len",
            "title_digit_ratio", "title_upper_ratio", "title_punct_ratio",
            "full_text_digit_ratio", "full_text_punct_ratio",

            # ── User profile ──────────────────────────────────
            "photo_count_log1p",

            # ── Fold-safe user aggregate ──────────────────────
            "user_prev_post_count", "user_mean_label",
            "user_active_hour_mean", "user_category_nunique",

            # ── user_description features ─────────────────────
            "user_desc_len", "user_desc_word_count", "user_desc_keyword_count",

            # ── Location encoding ─────────────────────────────
            "location_text_target_enc", "location_text_freq",
            "country_target_enc", "state_target_enc", "city_target_enc",
            "country_freq", "state_freq", "city_freq",

            # ── REMOVED (constant / null in official data) ────
            # "timezone_offset",        # all 0
            # "account_age_days_log1p", # all 0
            # "camera_age_days_log1p",  # all 0
            # "follower_count_log1p",   # not in official data
            # "following_count_log1p",  # not in official data
            # "total_views_log1p",      # not in official data
            # "total_favorites_log1p",  # not in official data
            # "mean_views_log1p",       # not in official data
            # "mean_favorites_log1p",   # not in official data
            # "mean_tags_log1p",        # not in official data
            # "follower_following_ratio",
            # "views_per_photo",
            # "favorites_per_photo",
        ]
        self.cat_cols = cat_cols or [
            "category", "subcategory", "concept",
            "category_subcategory_combo", "category_concept_combo",
            "geo_cluster",
            "mediatype",
            "city", "state", "country",

            # ── REMOVED (constant / null in official data) ────
            # "timezone_id",   # cardinality=1
            # "mediastatus",   # cardinality=1
        ]
        self.bin_cols = bin_cols or [
            "is_weekend", "is_night", "is_workhour",
            "ispro", "ispublic",
            "has_geo", "has_title", "has_tags",
            "has_user_description", "has_location_text",
            "user_desc_kw_photo", "user_desc_kw_travel", "user_desc_kw_art",
            "user_desc_kw_nature", "user_desc_kw_camera", "user_desc_kw_pro",
            "user_desc_kw_social", "user_desc_kw_location",
            "has_city", "has_state", "has_country",

            # ── REMOVED (constant) ────────────────────────────
            # "canbuypro",     # all 0
            # "has_full_text", # all 1
            # "has_image",     # all 1
        ]
        self.text_cols = text_cols or ["title", "alltags", "full_text"]
        self.log1p_cols = log1p_cols or []
        self.normalize_numeric = normalize_numeric
        self.user_desc_dim = user_desc_dim
        self.loc_desc_dim = loc_desc_dim

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
        # has_* columns. This keeps old datasets and newer enriched datasets
        # compatible with the same config.
        raw_needed = [
            "title", "alltags", "full_text",
            "user_description", "location_description", "location_description_clean", "location_text",
            "city", "state", "country", "latitude", "longitude", "image_path",
        ]
        for c in raw_needed:
            if c not in out.columns:
                out[c] = None

        # Derive missingness / availability features only when they are part of
        # bin_cols. If a config explicitly provides a different bin_cols list,
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
        if "has_state" in self.bin_cols:
            out["has_state"] = out["state"].map(_has_text_value).astype(np.int64)
        if "has_country" in self.bin_cols:
            out["has_country"] = out["country"].map(_has_text_value).astype(np.int64)
        if "has_image" in self.bin_cols:
            out["has_image"] = out["image_path"].map(_has_text_value).astype(np.int64)

        out = _derive_text_length_features(out)

        # YAML-controlled user_desc_lite:
        # if disabled, keep schema stable but zero out the lightweight features.
        if not self.use_user_desc_lite:
            if "user_desc_keyword_count" in out.columns:
                out["user_desc_keyword_count"] = 0.0
            for group in USER_DESC_LITE_KEYWORD_GROUPS:
                c = f"user_desc_kw_{group}"
                if c in out.columns:
                    out[c] = 0

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
        global USER_DESC_LITE_DEBUG_ENABLED
        USER_DESC_LITE_DEBUG_ENABLED = bool(self.user_desc_lite_debug)
        df = self._ensure_columns(train_df)

        if self.use_user_desc_lite and self.user_desc_lite_debug:
            lite_num_cols = [c for c in self.num_cols if "user_desc" in c]
            lite_bin_cols = [c for c in self.bin_cols if c.startswith("user_desc_kw_")]
            print(f"[USER_DESC_LITE] enabled num/bin cols | num={lite_num_cols} | bin={lite_bin_cols}")

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

        global USER_DESC_LITE_DEBUG_ENABLED
        USER_DESC_LITE_DEBUG_ENABLED = bool(self.user_desc_lite_debug)
        out = self._ensure_columns(df)

        # Avoid pandas PerformanceWarning:
        # repeatedly assigning transformed columns fragments the DataFrame.
        # Collect all num/bin/cat outputs first, then concatenate once.
        new_cols: Dict[str, Any] = {}

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

            new_cols[f"num__{col}"] = pd.Series(
                s.astype(np.float32),
                index=out.index,
                name=f"num__{col}",
            )

        for col in self.bin_cols:
            s = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int64)
            s = s.clip(lower=0, upper=1)
            new_cols[f"bin__{col}"] = pd.Series(
                s,
                index=out.index,
                name=f"bin__{col}",
            )

        for col in self.cat_cols:
            vocab = self.cat_vocab[col]
            unk_id = vocab.get("UNK", 0)
            values = out[col].map(lambda x: _safe_str(x, "UNK") or "UNK").fillna("UNK")
            mapped = values.map(lambda x: vocab.get(str(x), unk_id))
            new_cols[f"cat__{col}"] = pd.Series(
                mapped.astype(np.int64),
                index=out.index,
                name=f"cat__{col}",
            )

        if new_cols:
            out = out.drop(columns=[c for c in new_cols if c in out.columns], errors="ignore")
            out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

        # Defragment once. This also helps if upstream feature engineering
        # has already fragmented the raw dataframe.
        out = out.copy()

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
            "loc_desc_dim": self.loc_desc_dim,
            "use_user_desc_lite": self.use_user_desc_lite,
            "user_desc_lite_debug": self.user_desc_lite_debug,
            "user_desc_lite_max_keywords": self.user_desc_lite_max_keywords,
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
            use_user_desc_lite=payload.get("use_user_desc_lite", True),
            user_desc_lite_debug=payload.get("user_desc_lite_debug", False),
            user_desc_lite_max_keywords=payload.get("user_desc_lite_max_keywords", 5),
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