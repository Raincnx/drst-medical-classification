from __future__ import annotations

import argparse

import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.ham10000_dataset import HAM10000Dataset, build_transforms
from src.models.resnet50 import build_resnet50
from src.training.common import create_optimizer, fit_supervised, get_device
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_json
from src.utils.logger import CSVLogger
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train supervised-only baseline.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--label-map", type=str, required=True)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    label_map = load_json(args.label_map)
    train_dataset = HAM10000Dataset(
        csv_path=args.train_csv,
        label_map=label_map,
        transform=build_transforms(config["dataset"]["image_size"], train=True),
        unlabeled=False,
        image_root=args.image_root,
    )
    val_dataset = HAM10000Dataset(
        csv_path=args.val_csv,
        label_map=label_map,
        transform=build_transforms(config["dataset"]["image_size"], train=False),
        unlabeled=False,
        image_root=args.image_root,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )

    model = build_resnet50(
        num_classes=config["dataset"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        freeze_backbone=config["model"]["freeze_backbone"],
    )
    device = get_device()
    model.to(device)

    optimizer = create_optimizer(
        model,
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = ensure_dir(config["paths"]["checkpoint_dir"])
    log_dir = ensure_dir(config["paths"]["log_dir"])
    checkpoint_path = checkpoint_dir / f"{args.run_name}_best.pt"
    logger = CSVLogger(
        path=log_dir / f"{args.run_name}.csv",
        fieldnames=[
            "epoch",
            "train_loss",
            "train_accuracy",
            "train_macro_f1",
            "train_balanced_accuracy",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
            "val_balanced_accuracy",
        ],
    )

    fit_supervised(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=int(config["training"]["epochs"]),
        checkpoint_path=checkpoint_path,
        logger=logger,
    )
    print(f"Saved best supervised checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
