#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Iterable


EPOCH_RE = re.compile(
    r"train_loss=([0-9.]+) \| "
    r"val_loss=([0-9.]+) \| "
    r"val_mae=([0-9.]+) \| "
    r"val_spearman=([0-9.]+) \| "
    r"epoch_time=([0-9.]+)s"
)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def extract_exp_name(text: str, fallback: str) -> str:
    match = re.search(r"\[CONFIG\].*?exp_name[=:]\s*['\"]?([^,'\"}\s]+)", text)
    if match:
        return match.group(1)
    best = re.search(r"outputs/([^/\s]+)/fold_", text)
    if best:
        return best.group(1)
    return fallback


def extract_error(text: str) -> str:
    patterns = [
        r"(CUDA out of memory[^\n]*)",
        r"(RuntimeError:[^\n]*)",
        r"(ValueError:[^\n]*)",
        r"(ImportError:[^\n]*)",
        r"(ModuleNotFoundError:[^\n]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            msg = match.group(1).strip()
            return msg[:96]
    if "Traceback" in text:
        return "Traceback"
    return "-"


def parse_log(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    run_name = extract_exp_name(text, path.stem)
    rows = [tuple(map(float, match.groups())) for match in EPOCH_RE.finditer(text)]
    if not rows:
        status = "running" if "Traceback" not in text and "ERROR" not in text else "failed"
        return [run_name, status, "-", "-", "-", "-", "-", "-", extract_error(text), str(path)]

    best_epoch, best = max(enumerate(rows, 1), key=lambda item: item[1][3])
    last = rows[-1]
    stopped = "yes" if "Early stopping at epoch" in text else "no"
    status = "done" if "Training finished." in text else "running"
    if "Traceback" in text or "RuntimeError" in text:
        status = "failed"

    return [
        run_name,
        status,
        str(len(rows)),
        str(best_epoch),
        fmt(best[3]),
        fmt(best[2]),
        fmt(last[3]),
        fmt(sum(row[4] for row in rows) / len(rows), 1),
        stopped,
        extract_error(text),
        str(path),
    ]


def print_table(headers: list[str], rows: Iterable[list[str]]) -> None:
    rows = list(rows)
    widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
    print(" | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs/xattn_experiments")
    parser.add_argument("--sort", choices=["name", "best"], default="best")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    logs = sorted(log_dir.glob("*.log"))
    if not logs:
        raise SystemExit(f"No .log files found in {log_dir}")

    headers = [
        "run",
        "status",
        "epochs",
        "best_ep",
        "best_sp",
        "best_mae",
        "last_sp",
        "avg_sec",
        "early_stop",
        "error",
        "log",
    ]
    rows = [parse_log(path) for path in logs]
    if args.sort == "best":
        rows.sort(key=lambda row: float(row[4]) if row[4] != "-" else -1.0, reverse=True)
    print_table(headers, rows)


if __name__ == "__main__":
    main()
