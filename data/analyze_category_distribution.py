import json
import pandas as pd
from pathlib import Path


def load_json_or_jsonl(path):
    path = Path(path)

    data = []
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Only support .json or .jsonl")

    return pd.DataFrame(data)


def analyze_category(df):
    # 清理
    df["Category"] = df["Category"].fillna("unknown")

    # 統計
    counts = df["Category"].value_counts()
    ratios = df["Category"].value_counts(normalize=True) * 100

    result = pd.DataFrame({
        "count": counts,
        "ratio (%)": ratios.round(2)
    })

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="category_distribution.csv")

    args = parser.parse_args()

    df = load_json_or_jsonl(args.input)

    result = analyze_category(df)

    print("\n=== Category Distribution ===")
    print(result)

    result.to_csv(args.output)
    print(f"\nSaved to {args.output}")


#    python analyze_category_distribution.py --input ./raw/train/train_category.json
#    python analyze_category_distribution.py --input ./raw/test/test_category.json

""" train_category.json 的分析結果：
=== Category Distribution ===
                      count  ratio (%)
Category
Travel&Active&Sports  66469      21.75
Holiday&Celebrations  54751      17.92
Fashion               49282      16.13
Entertainment         29590       9.68
Social&People         23351       7.64
Whether&Season        20292       6.64
Animal                19992       6.54
Food                  16727       5.47
Urban                 15272       5.00
Electronics            5613       1.84
Family                 4274       1.40
"""

""" test_category.json 的分析結果：
=== Category Distribution ===
                      count  ratio (%)
Category
Travel&Active&Sports  45462      25.18
Holiday&Celebrations  19485      10.79
Animal                18482      10.23
Entertainment         17969       9.95
Fashion               17961       9.95
Whether&Season        14926       8.27
Social&People         14447       8.00
Urban                 12000       6.65
Food                  11833       6.55
Electronics            5987       3.32
Family                 2029       1.12
"""