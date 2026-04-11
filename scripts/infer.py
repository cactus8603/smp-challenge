import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.models.fusion_model import FusionModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Path to input file or folder")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FusionModel(cfg)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    print(f"Inference placeholder for input: {args.input}")
    print("You can later connect this to your dataset / preprocessing pipeline.")


if __name__ == "__main__":
    main()