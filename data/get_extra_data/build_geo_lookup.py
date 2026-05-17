#!/usr/bin/env python3
"""
build_geo_lookup.py

Build a lat/lon → {city, state, country} lookup JSON from the official SMP
temporal-spatial data files.

Uses reverse_geocoder (offline, batch, fast) + pycountry for country names.

Usage:
    python3 build_geo_lookup.py
    python3 build_geo_lookup.py --output /path/to/geo_lookup.json

Requirements:
    pip install reverse_geocoder pycountry
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import reverse_geocoder as rg
    import pycountry
except ImportError:
    raise SystemExit(
        "Missing dependencies. Run:\n"
        "  pip install reverse_geocoder pycountry"
    )


OFFICIAL_DATA_FILES = [
    Path(__file__).resolve().parents[1]
    / "official_data/train_set/train_allmetadata_json/train_temporalspatial_information.json",
    Path(__file__).resolve().parents[1]
    / "official_data/test_set/test_allmetadata_json/test_temporalspatial_information.json",
]

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "geo_lookup.json"


def load_unique_coords(paths: list[Path]) -> list[tuple[float, float]]:
    seen: set[tuple[float, float]] = set()
    for path in paths:
        if not path.exists():
            print(f"[WARN] File not found, skipping: {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            lat_raw = item.get("Latitude")
            lon_raw = item.get("Longitude")
            if not lat_raw or not lon_raw:
                continue
            try:
                lat = float(lat_raw)
                lon = float(lon_raw)
            except (TypeError, ValueError):
                continue
            # Skip zero/missing coordinates
            if lat == 0.0 and lon == 0.0:
                continue
            # Round to 2dp to match the classmate's key format
            key = (round(lat, 2), round(lon, 2))
            seen.add(key)
        print(f"  Loaded {path.name}: cumulative {len(seen)} unique keys so far")
    return sorted(seen)


def build_lookup(coords: list[tuple[float, float]]) -> dict[str, dict]:
    print(f"\nRunning batch reverse geocoding on {len(coords)} unique coordinates...")
    results = rg.search(coords, verbose=False)

    lookup: dict[str, dict] = {}
    for (lat, lon), r in zip(coords, results):
        cc = r.get("cc", "")
        country_obj = pycountry.countries.get(alpha_2=cc)
        country_name = country_obj.name if country_obj else cc

        city = r.get("name") or None
        state = r.get("admin1") or None

        key = f"{lat},{lon}"
        lookup[key] = {
            "city": city,
            "state": state,
            "country": country_name,
        }

    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Build geo lookup JSON from official SMP data.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    output_path = Path(args.output)

    print("=== build_geo_lookup.py ===")
    print("Collecting unique lat/lon keys from official data...")
    coords = load_unique_coords(OFFICIAL_DATA_FILES)
    print(f"Total unique coordinate keys: {len(coords)}\n")

    lookup = build_lookup(coords)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(lookup)} entries → {output_path}")
    print("\nSample entries:")
    for key in list(lookup.keys())[:5]:
        print(f"  {key!r}: {lookup[key]}")

    print("\nDone. Set this path in run_smp_category_crawl.sh:")
    print(f"  export GEO_LOOKUP_PATH={output_path}")


if __name__ == "__main__":
    main()
