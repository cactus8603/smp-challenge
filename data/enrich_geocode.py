#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enrich_geocode.py

離線 reverse geocoding：把所有 lat/lon 查成 city/state/country，
結果存成 cache JSON，供 build_dataset_v2.py 直接讀取。

支援來源：
  - 官方 train/test 的 *_temporalspatial_information.json
  - 爬蟲 extra data 的 extra_temporalspatial_information.jsonl

用法：
  # 查官方 train + test
  python3 enrich_geocode.py \
      --official_dir /local/smp/data \
      --cache /local/smp/geocode_cache.json

  # 加上 extra data
  python3 enrich_geocode.py \
      --official_dir /local/smp/data \
      --extra_dir    /local/smp/extra_data_v2 \
      --cache        /local/smp/geocode_cache.json

  # 只補 extra data（官方已查過，resume 會跳過）
  python3 enrich_geocode.py \
      --extra_dir /local/smp/extra_data_v2 \
      --cache     /local/smp/geocode_cache.json

  # 乾跑，不打 API，只統計有多少座標要查
  python3 enrich_geocode.py \
      --official_dir /local/smp/data \
      --extra_dir    /local/smp/extra_data_v2 \
      --cache        /local/smp/geocode_cache.json \
      --dry_run

Cache 格式（key = "lat_2dp,lon_2dp"）：
  {
    "25.04,121.51": {"city": "Taipei", "state": "Taiwan", "country": "Taiwan"},
    "0.0,0.0":      null,   ← 查不到或無效座標
    ...
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Coord helpers
# ──────────────────────────────────────────────
INVALID_COORDS = {None, "", "0.0", "0", "0.00", 0, 0.0}


def to_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(str(v).strip())
        return None if f != f else f   # NaN check
    except Exception:
        return None


def is_valid_coord(lat, lon) -> bool:
    lat = to_float(lat)
    lon = to_float(lon)
    if lat is None or lon is None:
        return False
    if lat == 0.0 and lon == 0.0:
        return False
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return False
    return True


def coord_key(lat, lon, precision: int = 2) -> str:
    """Round to `precision` decimal places and make a cache key."""
    return f"{round(float(lat), precision)},{round(float(lon), precision)}"


# ──────────────────────────────────────────────
# Coord extraction from data files
# ──────────────────────────────────────────────
def extract_coords_from_records(records: List[Dict]) -> List[Tuple[float, float]]:
    coords = []
    for r in records:
        lat = r.get("Latitude") or r.get("latitude")
        lon = r.get("Longitude") or r.get("longitude")
        if is_valid_coord(lat, lon):
            coords.append((float(str(lat)), float(str(lon))))
    return coords


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def load_json(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "items", "records", "results"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return list(data.values()) if all(isinstance(v, dict) for v in data.values()) else []
    return []


def collect_all_coords(
    official_dir: Optional[Path],
    extra_dir: Optional[Path],
) -> List[Tuple[float, float]]:
    all_coords: List[Tuple[float, float]] = []

    # Official train + test
    if official_dir and official_dir.exists():
        for split in ("train", "test"):
            fname = f"{split}_temporalspatial_information.json"
            path  = official_dir / split / fname
            if not path.exists():
                # try flat structure
                path = official_dir / fname
            records = load_json(path)
            coords  = extract_coords_from_records(records)
            log.info("Official %s: %d coords with geo", split, len(coords))
            all_coords.extend(coords)

    # Extra data JSONL
    if extra_dir and extra_dir.exists():
        jsonl_files = list(extra_dir.rglob("extra_temporalspatial_information.jsonl"))
        log.info("Extra data: found %d JSONL files", len(jsonl_files))
        for jf in jsonl_files:
            records = load_jsonl(jf)
            coords  = extract_coords_from_records(records)
            all_coords.extend(coords)
        log.info("Extra data: %d coords with geo total", len(all_coords))

    return all_coords


# ──────────────────────────────────────────────
# Cache I/O
# ──────────────────────────────────────────────
def load_cache(path: Path) -> Dict[str, Optional[Dict]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.warning("Failed to load cache %s, starting fresh", path)
        return {}


def save_cache(cache: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ──────────────────────────────────────────────
# Geocoding
# ──────────────────────────────────────────────
def build_geocoder():
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeocoderServiceError
        geocoder = Nominatim(user_agent="smp-geocoder/1.0", timeout=10)
        return geocoder, GeocoderTimedOut, GeocoderServiceError
    except ImportError:
        raise SystemExit(
            "geopy not installed. Run: pip install geopy"
        )


def reverse_geocode(
    geocoder,
    lat: float,
    lon: float,
    GeocoderTimedOut,
    GeocoderServiceError,
    retries: int = 3,
) -> Optional[Dict[str, Optional[str]]]:
    """
    Returns {"city": ..., "state": ..., "country": ...} or None on failure.
    """
    for attempt in range(retries):
        try:
            location = geocoder.reverse(
                (lat, lon),
                exactly_one=True,
                language="en",
            )
            if location is None:
                return None

            addr = location.raw.get("address", {})

            city = (
                addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or addr.get("municipality")
                or addr.get("county")
            )
            state   = addr.get("state") or addr.get("region")
            country = addr.get("country")

            return {
                "city":    city    or None,
                "state":   state   or None,
                "country": country or None,
            }

        except GeocoderTimedOut:
            wait = 2 ** attempt
            log.debug("Timeout lat=%.4f lon=%.4f attempt=%d, retrying in %ds",
                      lat, lon, attempt + 1, wait)
            time.sleep(wait)
        except GeocoderServiceError as e:
            log.warning("Service error lat=%.4f lon=%.4f: %s", lat, lon, e)
            time.sleep(5)
        except Exception as e:
            log.warning("Unexpected error lat=%.4f lon=%.4f: %s", lat, lon, e)
            return None

    return None


# ──────────────────────────────────────────────
# Enrich files and save to output_dir
# ──────────────────────────────────────────────
def enrich_records_with_geo(
    records: List[Dict],
    cache: Dict[str, Optional[Dict]],
    precision: int,
) -> List[Dict]:
    """Attach city/state/country to each record from cache."""
    enriched = []
    for r in records:
        r = dict(r)   # don't mutate original
        lat = r.get("Latitude") or r.get("latitude")
        lon = r.get("Longitude") or r.get("longitude")
        if is_valid_coord(lat, lon):
            key = coord_key(float(str(lat)), float(str(lon)), precision)
            geo = cache.get(key)
            if geo:
                r["city"]    = geo.get("city")
                r["state"]   = geo.get("state")
                r["country"] = geo.get("country")
            else:
                r["city"] = r["state"] = r["country"] = None
        else:
            r["city"] = r["state"] = r["country"] = None
        enriched.append(r)
    return enriched


def save_enriched_file(
    records: List[Dict],
    src_path: Path,
    output_dir: Path,
    suffix: str = "_with_geo",
) -> Path:
    """
    Save enriched records to output_dir with suffix appended to filename.
    Preserves original extension (.json or .jsonl).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = src_path.stem + suffix
    ext  = src_path.suffix   # .json or .jsonl
    out_path = output_dir / (stem + ext)

    if ext == ".jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    log.info("Saved %s (%d records)", out_path, len(records))
    return out_path


def write_enriched_files(
    official_dir: Optional[Path],
    extra_dir: Optional[Path],
    output_dir: Path,
    cache: Dict[str, Optional[Dict]],
    precision: int,
    suffix: str,
) -> None:
    """
    Read all temporal files, attach geo from cache, write enriched copies.
    Does NOT modify any original file.
    """
    # Official train + test
    if official_dir and official_dir.exists():
        for split in ("train", "test"):
            fname = f"{split}_temporalspatial_information.json"
            src   = official_dir / split / fname
            if not src.exists():
                src = official_dir / fname
            if not src.exists():
                log.warning("Official %s temporal file not found: %s", split, src)
                continue
            records  = load_json(src)
            enriched = enrich_records_with_geo(records, cache, precision)
            save_enriched_file(enriched, src, output_dir / split, suffix)

    # Extra data JSONL
    if extra_dir and extra_dir.exists():
        jsonl_files = list(extra_dir.rglob("extra_temporalspatial_information.jsonl"))
        for jf in jsonl_files:
            # Mirror the subdirectory structure under output_dir
            try:
                rel = jf.parent.relative_to(extra_dir)
            except ValueError:
                rel = Path(jf.parent.name)
            records  = load_jsonl(jf)
            enriched = enrich_records_with_geo(records, cache, precision)
            save_enriched_file(enriched, jf, output_dir / rel, suffix)


# ──────────────────────────────────────────────
# Main enrichment loop
# ──────────────────────────────────────────────
def enrich(
    all_coords: List[Tuple[float, float]],
    cache: Dict[str, Optional[Dict]],
    cache_path: Path,
    precision: int,
    sleep_sec: float,
    save_every: int,
    dry_run: bool,
) -> Dict[str, Optional[Dict]]:

    # De-duplicate and filter already cached
    unique_keys: Dict[str, Tuple[float, float]] = {}
    for lat, lon in all_coords:
        key = coord_key(lat, lon, precision)
        if key not in cache and key not in unique_keys:
            unique_keys[key] = (lat, lon)

    total     = len(unique_keys)
    log.info("Unique coords to query: %d  (already cached: %d)", total, len(cache))

    if dry_run:
        est_minutes = total * sleep_sec / 60
        log.info("DRY RUN — would query %d coords (~%.0f min at %.1fs/req)",
                 total, est_minutes, sleep_sec)
        return cache

    if total == 0:
        log.info("Nothing new to query.")
        return cache

    geocoder, GeocoderTimedOut, GeocoderServiceError = build_geocoder()

    done = 0
    for key, (lat, lon) in unique_keys.items():
        result = reverse_geocode(
            geocoder, lat, lon,
            GeocoderTimedOut, GeocoderServiceError,
        )
        cache[key] = result
        done += 1

        if result:
            log.debug("[%d/%d] %.4f,%.4f → %s, %s, %s",
                      done, total, lat, lon,
                      result.get("city"), result.get("state"), result.get("country"))
        else:
            log.debug("[%d/%d] %.4f,%.4f → not found", done, total, lat, lon)

        if done % 100 == 0:
            found = sum(1 for v in cache.values() if v is not None)
            log.info("Progress: %d/%d queried | cache size=%d | found rate=%.1f%%",
                     done, total, len(cache), 100 * found / max(len(cache), 1))

        if done % save_every == 0:
            save_cache(cache, cache_path)
            log.info("Cache saved (%d entries)", len(cache))

        time.sleep(sleep_sec)

    save_cache(cache, cache_path)

    found = sum(1 for v in cache.values() if v is not None)
    log.info("Done. Cache: %d entries, %.1f%% found",
             len(cache), 100 * found / max(len(cache), 1))
    return cache


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reverse geocode lat/lon → city/state/country.")
    p.add_argument("--official_dir", default=None,
                   help="Official data root (contains train/ and test/ subdirs)")
    p.add_argument("--extra_dir",    default=None,
                   help="Extra data root (contains extra_data_* subdirs)")
    p.add_argument("--cache",        default="/local/smp/geocode_cache.json",
                   help="Path to cache JSON file (created if not exists)")
    p.add_argument("--output_dir",   default=None,
                   help="Where to write enriched files (default: same as official_dir/extra_dir). "
                        "Original files are NEVER modified.")
    p.add_argument("--suffix",       default="_with_geo",
                   help="Suffix to append to enriched filenames (default: _with_geo)")
    p.add_argument("--precision",    type=int, default=2,
                   help="Decimal places for coord rounding (default: 2 ≈ 1km)")
    p.add_argument("--sleep",        type=float, default=1.1,
                   help="Sleep between requests in seconds (Nominatim requires >=1)")
    p.add_argument("--save_every",   type=int, default=200,
                   help="Save cache every N requests")
    p.add_argument("--dry_run",      action="store_true",
                   help="Count coords to query without actually calling the API")
    p.add_argument("--skip_query",   action="store_true",
                   help="Skip geocoding, just write enriched files from existing cache")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.official_dir and not args.extra_dir:
        raise SystemExit("Provide at least one of --official_dir or --extra_dir")

    official_dir = Path(args.official_dir) if args.official_dir else None
    extra_dir    = Path(args.extra_dir)    if args.extra_dir    else None
    cache_path   = Path(args.cache)

    # Output dir defaults to a sibling of the cache file
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = cache_path.parent / "enriched"

    log.info("Cache path : %s", cache_path)
    log.info("Output dir : %s", output_dir)

    # Load existing cache
    cache = load_cache(cache_path)
    if cache:
        log.info("Loaded existing cache: %d entries", len(cache))

    # Step 1: collect coords and query API
    if not args.skip_query:
        all_coords = collect_all_coords(official_dir, extra_dir)
        log.info("Total coord records collected: %d", len(all_coords))
        cache = enrich(
            all_coords=all_coords,
            cache=cache,
            cache_path=cache_path,
            precision=args.precision,
            sleep_sec=args.sleep,
            save_every=args.save_every,
            dry_run=args.dry_run,
        )

    # Step 2: write enriched files (skipped in dry_run)
    if not args.dry_run:
        log.info("Writing enriched files to %s ...", output_dir)
        write_enriched_files(
            official_dir=official_dir,
            extra_dir=extra_dir,
            output_dir=output_dir,
            cache=cache,
            precision=args.precision,
            suffix=args.suffix,
        )
        log.info("All enriched files written. Original files untouched.")


if __name__ == "__main__":
    main()


# # dry run 先確認數量
# python3 enrich_geocode.py \
#     --official_dir /local/smp/data \
#     --cache        /local/smp/geocode_cache.json \
#     --output_dir   /local/smp/data_enriched \
#     --dry_run
# python3 enrich_geocode.py --official_dir /local/smp/data --cache /local/smp/geocode_cache.json --output_dir /local/smp/data_enriched --dry_run

# # 正式跑（背景執行）
# nohup python3 enrich_geocode.py \
#     --official_dir /local/smp/data \
#     --cache        /local/smp/geocode_cache.json \
#     --output_dir   /local/smp/data_enriched \
#     > geocode.log 2>&1 &
# echo $!
# nohup python3 enrich_geocode.py --official_dir /local/smp/data --cache /local/smp/geocode_cache.json --output_dir /local/smp/data_enriched > geocode.log 2>&1 &