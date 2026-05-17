#!/usr/bin/env python3
"""
enrich_existing_data.py

Post-process already-crawled output directories:
  1. Reconcile photo-level JSONL files to a consistent PID set (intersection)
  2. Reconcile extra_user_data.jsonl (Uid-keyed) to users in the canonical photo set
  3. Enrich extra_temporalspatial_information.jsonl with city/state/country

Note on file schemas:
  - Photo-level (keyed by Pid): text, temporalspatial, additional, category,
                                 pseudo_label, crawl_manifest
  - User-level  (keyed by Uid): user_data  ← one row per unique user

Usage:
    python3 enrich_existing_data.py                           # all output/* dirs
    python3 enrich_existing_data.py --dirs path/to/dir ...    # specific dirs
    python3 enrich_existing_data.py --dry_run                 # stats only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
DEFAULT_GEO_LOOKUP  = Path(__file__).resolve().parent / "geo_lookup.json"

# Photo-level files: each row corresponds to one photo (Pid)
PID_FILES = [
    "extra_text.jsonl",
    "extra_temporalspatial_information.jsonl",
    "extra_additional_information.jsonl",
    "extra_category.jsonl",
    "extra_pseudo_label.jsonl",
    "crawl_manifest.jsonl",
]

# User-level file: one row per unique user (Uid), not per photo
UID_FILE = "extra_user_data.jsonl"


# ──────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────
def load_geo_lookup(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl_atomic(path: Path, rows: list) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    shutil.move(str(tmp), str(path))


# ──────────────────────────────────────────────
# Geo helpers
# ──────────────────────────────────────────────
def lookup_city(lat, lon, geo_lookup: dict):
    if lat in (None, 0, 0.0, "0", "") or lon in (None, 0, 0.0, "0", ""):
        return None, None, None
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        return None, None, None
    key = f"{round(lat_f, 2)},{round(lon_f, 2)}"
    entry = geo_lookup.get(key)
    if not entry:
        return None, None, None
    return entry.get("city"), entry.get("state"), entry.get("country")


# ──────────────────────────────────────────────
# Core: process one directory
# ──────────────────────────────────────────────
def process_dir(data_dir: Path, geo_lookup: dict, dry_run: bool) -> None:
    name = data_dir.name
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # ── Load photo-level files ────────────────────────────
    pid_rows: dict[str, list] = {}
    pid_sets: dict[str, set]  = {}

    for fname in PID_FILES:
        p = data_dir / fname
        if not p.exists():
            print(f"  [WARN] Missing: {fname}")
            continue
        rows = list(iter_jsonl(p))
        pid_rows[fname] = rows
        pid_sets[fname] = {str(r.get("Pid", "")) for r in rows}

    # ── Canonical PID set = intersection of all pid files ─
    all_sets = list(pid_sets.values())
    if not all_sets:
        print("  [ERROR] No photo-level files found.")
        return

    canonical_pids: set[str] = all_sets[0].copy()
    for s in all_sets[1:]:
        canonical_pids &= s
    canonical_pids.discard("")  # remove any empty-key rows

    # ── Print counts ──────────────────────────────────────
    print(f"\n  Photo-level files (before reconciliation):")
    for fname in PID_FILES:
        if fname in pid_sets:
            n    = len(pid_rows[fname])
            keep = len(pid_sets[fname] & canonical_pids)
            drop = n - keep
            flag = f"  ← drop {drop}" if drop else ""
            print(f"    {fname:<54} {n:>7,}{flag}")
        else:
            print(f"    {fname:<54}  MISSING")

    print(f"\n  Canonical PID set: {len(canonical_pids):,}")

    # ── User-level file ───────────────────────────────────
    uid_path = data_dir / UID_FILE
    uid_rows: list | None = None
    canonical_uids: set[str] = set()

    if uid_path.exists():
        # Collect Uids that appear in the canonical photo set (via text file)
        if "extra_text.jsonl" in pid_rows:
            canonical_uids = {
                str(r.get("Uid", ""))
                for r in pid_rows["extra_text.jsonl"]
                if str(r.get("Pid", "")) in canonical_pids
            }
        uid_rows = list(iter_jsonl(uid_path))
        uid_before = len(uid_rows)
        uid_filtered = [r for r in uid_rows if str(r.get("Uid", "")) in canonical_uids]
        uid_drop = uid_before - len(uid_filtered)
        flag = f"  ← drop {uid_drop}" if uid_drop else ""
        print(f"\n  User-level file:")
        print(f"    {UID_FILE:<54} {uid_before:>7,}{flag}")
        print(f"    (unique users in canonical set: {len(canonical_uids):,})")

    # ── Pseudo-label coverage ─────────────────────────────
    if "extra_pseudo_label.jsonl" in pid_rows:
        rows = pid_rows["extra_pseudo_label.jsonl"]
        total      = len(rows)
        labeled    = sum(1 for r in rows if r.get("PseudoLogViewScore") is not None)
        zero_views = sum(1 for r in rows if r.get("views") == 0)
        null_total = total - labeled
        print(f"\n  Pseudo-label coverage ({total:,} records before reconciliation):")
        print(f"    labeled        : {labeled:,}  ({labeled/total*100:.1f}%)")
        print(f"    null (0 views) : {zero_views:,}")
        if null_total - zero_views:
            print(f"    null (other)   : {null_total - zero_views:,}  (post too recent or missing views)")

    # ── Geo enrichment preview ────────────────────────────
    if "extra_temporalspatial_information.jsonl" in pid_rows:
        ts_rows = pid_rows["extra_temporalspatial_information.jsonl"]
        already = sum(1 for r in ts_rows if r.get("city") is not None)
        has_coords = sum(
            1 for r in ts_rows
            if r.get("Latitude") not in (None, 0, 0.0, "0", "")
            and r.get("Longitude") not in (None, 0, 0.0, "0", "")
        )
        will_enrich = sum(
            1 for r in ts_rows
            if r.get("city") is None
            and lookup_city(r.get("Latitude"), r.get("Longitude"), geo_lookup)[0] is not None
        )
        print(f"\n  Geo enrichment (extra_temporalspatial_information.jsonl):")
        print(f"    records with coords   : {has_coords:,}")
        print(f"    already have city     : {already:,}")
        print(f"    will be enriched now  : {will_enrich:,}")
        no_match = has_coords - already - will_enrich
        if no_match:
            print(f"    remain null (no key)  : {no_match:,}")

    if dry_run:
        print(f"\n  [DRY RUN] No files modified.")
        return

    # ── Write reconciled files ────────────────────────────
    print(f"\n  Writing reconciled files...")

    for fname in PID_FILES:
        if fname not in pid_rows:
            continue
        rows    = pid_rows[fname]
        filtered = [r for r in rows if str(r.get("Pid", "")) in canonical_pids]

        # Geo enrichment
        if fname == "extra_temporalspatial_information.jsonl":
            for r in filtered:
                if r.get("city") is None:
                    city, state, country = lookup_city(
                        r.get("Latitude"), r.get("Longitude"), geo_lookup
                    )
                    r["city"]    = city
                    r["state"]   = state
                    r["country"] = country

        write_jsonl_atomic(data_dir / fname, filtered)
        print(f"    ✓ {fname:<54} → {len(filtered):,}")

    # User-level file
    if uid_path.exists() and uid_rows is not None:
        uid_filtered = [r for r in uid_rows if str(r.get("Uid", "")) in canonical_uids]
        write_jsonl_atomic(uid_path, uid_filtered)
        print(f"    ✓ {UID_FILE:<54} → {len(uid_filtered):,}")

    print(f"\n  Done: {name}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="*", default=None)
    parser.add_argument("--geo_lookup", default=str(DEFAULT_GEO_LOOKUP))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    geo_path = Path(args.geo_lookup)
    if not geo_path.exists():
        sys.exit(f"[ERROR] geo_lookup.json not found: {geo_path}\nRun build_geo_lookup.py first.")

    print(f"Loading geo lookup: {geo_path}")
    geo_lookup = load_geo_lookup(geo_path)
    print(f"Loaded {len(geo_lookup):,} entries.")

    if args.dirs:
        target_dirs = [Path(d) for d in args.dirs]
    else:
        root = Path(args.output_root)
        target_dirs = sorted(d for d in root.iterdir() if d.is_dir())

    if not target_dirs:
        sys.exit("[ERROR] No directories found.")

    mode = "DRY RUN" if args.dry_run else "WRITING"
    print(f"\nProcessing {len(target_dirs)} director(ies) [{mode}]")

    for d in target_dirs:
        process_dir(d, geo_lookup, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print("All done." + (" (DRY RUN — no files modified)" if args.dry_run else ""))
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
