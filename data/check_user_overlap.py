import json
from pathlib import Path
import pandas as pd


def load_json_or_jsonl(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def check_overlap(train_path, test_path, uid_col="Uid", pid_col="Pid"):
    train_df = load_json_or_jsonl(train_path)
    test_df = load_json_or_jsonl(test_path)

    print("Train shape:", train_df.shape)
    print("Test shape :", test_df.shape)

    train_uids = set(train_df[uid_col].dropna().astype(str).unique()) if uid_col in train_df.columns else set()
    test_uids = set(test_df[uid_col].dropna().astype(str).unique()) if uid_col in test_df.columns else set()

    train_pids = set(train_df[pid_col].dropna().astype(str).unique()) if pid_col in train_df.columns else set()
    test_pids = set(test_df[pid_col].dropna().astype(str).unique()) if pid_col in test_df.columns else set()

    uid_overlap = train_uids & test_uids
    pid_overlap = train_pids & test_pids

    print("\n=== UID overlap ===")
    print("Train unique UIDs:", len(train_uids))
    print("Test unique UIDs :", len(test_uids))
    print("Overlap UIDs     :", len(uid_overlap))
    if len(train_uids) > 0:
        print("Overlap ratio vs train UIDs: {:.4f}%".format(100 * len(uid_overlap) / len(train_uids)))
    if len(test_uids) > 0:
        print("Overlap ratio vs test UIDs : {:.4f}%".format(100 * len(uid_overlap) / len(test_uids)))

    print("\n=== PID overlap ===")
    print("Train unique PIDs:", len(train_pids))
    print("Test unique PIDs :", len(test_pids))
    print("Overlap PIDs     :", len(pid_overlap))

    overlap_uid_df = pd.DataFrame({uid_col: sorted(uid_overlap)})
    overlap_pid_df = pd.DataFrame({pid_col: sorted(pid_overlap)})

    overlap_uid_df.to_csv("uid_overlap.csv", index=False, encoding="utf-8-sig")
    overlap_pid_df.to_csv("pid_overlap.csv", index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print("- uid_overlap.csv")
    print("- pid_overlap.csv")


if __name__ == "__main__":
    train_path = "./train_allmetadata_json/train_category.json"
    test_path = "./test_allmetadata_json/test_category.json"
    check_overlap(train_path, test_path)