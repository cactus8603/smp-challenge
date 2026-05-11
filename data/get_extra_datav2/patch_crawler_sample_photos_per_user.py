#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch crawl_flickr_to_smp.py to make sample_photos_per_user() safe when
MAX_PHOTOS_PER_USER is small, e.g. 3 or 5.

Usage:
  python3 patch_crawler_sample_photos_per_user.py /path/to/crawl_flickr_to_smp.py

It creates a .bak backup before modifying.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

NEW_FUNC = r'''def sample_photos_per_user(max_photos_per_user: int) -> int:
    """
    Sample how many extra public photos to crawl from a user.

    This follows a long-tail distribution similar to SMP:
      - many users contribute only 1 photo
      - some users contribute 2-4 photos
      - a small number contribute more

    Safe for small MAX_PHOTOS_PER_USER values such as 3 or 5.
    """
    max_photos_per_user = int(max_photos_per_user or 0)

    if max_photos_per_user <= 0:
        return 0
    if max_photos_per_user == 1:
        return 1

    r = random.random()

    # Very small cap: keep it sparse.
    if max_photos_per_user <= 4:
        if r < 0.70:
            return 1
        return random.randint(2, max_photos_per_user)

    # Recommended cap: MAX_PHOTOS_PER_USER=5~10.
    if max_photos_per_user <= 10:
        if r < 0.60:
            return 1
        elif r < 0.90:
            return random.randint(2, min(4, max_photos_per_user))
        else:
            return random.randint(5, max_photos_per_user)

    # Larger cap: preserve long-tail behavior.
    if r < 0.50:
        return 1
    elif r < 0.80:
        return random.randint(2, 4)
    elif r < 0.95:
        return random.randint(5, min(15, max_photos_per_user))
    else:
        low = min(20, max_photos_per_user)
        return random.randint(low, max_photos_per_user)
'''


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python3 patch_crawler_sample_photos_per_user.py /path/to/crawl_flickr_to_smp.py")
        return 1

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return 1

    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"def sample_photos_per_user\s*\([^)]*\)\s*(?:->\s*[^:]+)?\:\n"
        r"(?:    .*\n|\s*\n)*?"
        r"(?=\ndef\s+|\nclass\s+|\n# ─|\Z)",
        flags=re.MULTILINE,
    )

    matches = list(pattern.finditer(text))
    if not matches:
        print("ERROR: sample_photos_per_user() not found. No changes made.")
        return 1
    if len(matches) > 1:
        print(f"ERROR: found {len(matches)} sample_photos_per_user() definitions. No changes made.")
        return 1

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)

    new_text = text[:matches[0].start()] + NEW_FUNC + text[matches[0].end():]
    path.write_text(new_text, encoding="utf-8")

    print(f"Patched: {path}")
    print(f"Backup : {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
