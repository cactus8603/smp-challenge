import os
import json

def format_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"✔ formatted: {file_path}")

    except Exception as e:
        print(f"✘ error in {file_path}: {e}")


def format_json_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                format_json_file(file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="./train_allmetadata_json", help="Path to folder containing JSON files")
    args = parser.parse_args()

    format_json_folder(args.folder)