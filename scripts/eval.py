import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.datasets.dataset import SMPDataset
from src.models.fusion_model import FusionModel
from src.engine.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SMPDataset(cfg, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    model = FusionModel(cfg)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    evaluator = Evaluator(model=model, device=device, cfg=cfg)
    metrics = evaluator.evaluate(loader)

    print("Evaluation results:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()