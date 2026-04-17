from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.ham10000_dataset import HAM10000Dataset, build_transforms
from src.evaluation.metrics import compute_classification_metrics, compute_confusion_matrix
from src.models.resnet50 import build_resnet50
from src.training.common import get_device, load_checkpoint
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_json, save_json
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a CSV split.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--label-map", type=str, required=True)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--split-name", type=str, required=True)
    parser.add_argument("--plot-confusion", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    device = get_device()
    label_map = load_json(args.label_map)
    idx_to_label = {v: k for k, v in label_map.items()}

    dataset = HAM10000Dataset(
        csv_path=args.test_csv,
        label_map=label_map,
        transform=build_transforms(config["dataset"]["image_size"], train=False),
        unlabeled=False,
        image_root=args.image_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )

    model = build_resnet50(
        num_classes=config["dataset"]["num_classes"],
        pretrained=False,
        freeze_backbone=False,
    )
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_targets = []
    running_loss = 0.0

    for batch in tqdm(loader, desc="Test", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    metrics = compute_classification_metrics(all_targets, all_preds)
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)

    table_dir = ensure_dir(config["paths"]["table_dir"])
    result_path = table_dir / f"{args.split_name}_metrics.json"
    save_json(metrics, result_path)
    print(f"Saved metrics to {result_path}")
    print(metrics)

    if args.plot_confusion:
        cm = compute_confusion_matrix(all_targets, all_preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm)
        ax.set_xticks(range(len(idx_to_label)))
        ax.set_yticks(range(len(idx_to_label)))
        ax.set_xticklabels([idx_to_label[i] for i in range(len(idx_to_label))], rotation=45, ha="right")
        ax.set_yticklabels([idx_to_label[i] for i in range(len(idx_to_label))])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Confusion Matrix: {args.split_name}")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        plot_dir = ensure_dir(config["paths"]["plot_dir"])
        fig_path = plot_dir / f"{args.split_name}_confusion_matrix.png"
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved confusion matrix to {fig_path}")


if __name__ == "__main__":
    main()
