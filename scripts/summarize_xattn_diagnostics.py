#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


EXPERIMENTS = [
    "xattn_diag_pairwise_baseline",
    "xattn_diag_caption",
    "xattn_diag_userdesc",
    "xattn_diag_ftmeta",
    "xattn_diag_residual_xattn",
    "xattn_diag_tag_off",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--experiments", nargs="*", default=EXPERIMENTS)
    return parser.parse_args()


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_epoch_record(history: Iterable[Dict[str, Any]], epoch: int) -> Optional[Dict[str, Any]]:
    for record in history:
        if int(record.get("epoch", -1)) == int(epoch):
            return record
    return None


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def get_drop(record: Optional[Dict[str, Any]], name: str) -> Optional[float]:
    if not record:
        return None
    ablation = record.get("modality_ablation") or {}
    drops = ablation.get("drops") or ablation.get("spearman_drop") or {}
    if name in drops:
        return drops[name]

    results = ablation.get("results") or ablation
    full = results.get("full", {}).get("spearman")
    masked = results.get(f"mask_{name}", {}).get("spearman")
    if full is None or masked is None:
        return None
    return float(full) - float(masked)


def get_gbdt(record: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not record:
        return None
    monitor = record.get("gbdt_monitor") or {}
    return monitor.get(key)


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    fold_name = f"fold_{args.fold}"

    rows = []
    for exp in args.experiments:
        exp_dir = root / exp / fold_name
        summary = load_json(exp_dir / "summary.json")
        history_doc = load_json(exp_dir / "history.json") or {}
        history = history_doc.get("history", [])

        if not summary:
            rows.append([exp, "missing", "-", "-", "-", "-", "-", "-", "-", "-"])
            continue

        best_epoch = int(summary.get("best_epoch", -1))
        best_record = find_epoch_record(history, best_epoch)
        rows.append(
            [
                exp,
                fmt(summary.get("best_val_spearman")),
                str(best_epoch),
                fmt(get_drop(best_record, "text")),
                fmt(get_drop(best_record, "meta")),
                fmt(get_drop(best_record, "image")),
                fmt(get_drop(best_record, "user_desc")),
                fmt(get_gbdt(best_record, "lightgbm_spearman")),
                fmt(get_gbdt(best_record, "catboost_spearman")),
                str(exp_dir),
            ]
        )

    headers = [
        "experiment",
        "best",
        "epoch",
        "drop_text",
        "drop_meta",
        "drop_image",
        "drop_user_desc",
        "lgbm",
        "cat",
        "path",
    ]
    widths = [
        max(len(str(row[i])) for row in [headers, *rows])
        for i in range(len(headers))
    ]

    print(" | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


if __name__ == "__main__":
    main()
