from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.ham10000_dataset import HAM10000Dataset, build_transforms
from src.models.resnet50 import build_resnet50
from src.training.common import get_device, load_checkpoint
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_json
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels using a teacher checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--label-map", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    label_map = load_json(args.label_map)
    idx_to_label = {v: k for k, v in label_map.items()}

    threshold = (
        args.confidence_threshold
        if args.confidence_threshold is not None
        else float(config["training"].get("confidence_threshold", 0.0))
    )

    dataset = HAM10000Dataset(
        csv_path=args.input_csv,
        label_map=label_map,
        transform=build_transforms(config["dataset"]["image_size"], train=False),
        unlabeled=True,
        image_root=args.image_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )

    device = get_device()
    model = build_resnet50(
        num_classes=config["dataset"]["num_classes"],
        pretrained=False,
        freeze_backbone=False,
    )
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    rows = []
    for batch in tqdm(loader, desc="Pseudo-label", leave=False):
        images = batch["image"].to(device)
        image_ids = batch["image_id"]
        image_paths = batch["image_path"]

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        confidence, pseudo_idx = torch.max(probs, dim=1)

        for i in range(images.size(0)):
            conf = float(confidence[i].item())
            if conf < threshold:
                continue

            idx = int(pseudo_idx[i].item())
            rows.append(
                {
                    "image_id": image_ids[i],
                    "image_path": image_paths[i],
                    "pseudo_label_idx": idx,
                    "pseudo_label_name": idx_to_label[idx],
                    "confidence": conf,
                }
            )

    out_df = pd.DataFrame(rows)
    output_path = Path(args.output_csv)
    ensure_dir(output_path.parent)
    out_df.to_csv(output_path, index=False)
    print(f"Saved {len(out_df)} pseudo-labeled rows to {output_path}")


if __name__ == "__main__":
    main()
