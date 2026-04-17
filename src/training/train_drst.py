from __future__ import annotations

import argparse
from itertools import cycle
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.ham10000_dataset import HAM10000Dataset, build_transforms
from src.models.resnet50 import build_resnet50
from src.training.common import (
    create_optimizer,
    evaluate_epoch,
    get_device,
    load_checkpoint,
    save_checkpoint,
)
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_json
from src.utils.logger import CSVLogger
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train DRST with a frozen teacher.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--teacher-checkpoint", type=str, required=True)
    parser.add_argument("--labeled-csv", type=str, required=True)
    parser.add_argument("--unlabeled-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, required=True)
    parser.add_argument("--label-map", type=str, required=True)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    return parser.parse_args()


def hard_pseudo_labels(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=1)
    confidence, pseudo = torch.max(probs, dim=1)
    return pseudo, confidence


def cross_entropy_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, labels, reduction="none")


def train_drst_epoch(
    student: nn.Module,
    teacher: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_idx: int,
    total_epochs: int,
    confidence_threshold: float = 0.0,
) -> Dict[str, float]:
    student.train()
    teacher.eval()

    alpha = epoch_idx / total_epochs
    unlabeled_iter = cycle(unlabeled_loader)
    num_steps = max(len(labeled_loader), len(unlabeled_loader), 1)

    running_loss = 0.0
    running_examples = 0
    all_preds = []
    all_targets = []

    for _, batch_l in zip(
        range(num_steps),
        tqdm(cycle(labeled_loader), total=num_steps, desc=f"DRST Train {epoch_idx}/{total_epochs}", leave=False),
    ):
        batch_u = next(unlabeled_iter)

        x_l = batch_l["image"].to(device)
        y_l = batch_l["label"].to(device)
        x_u = batch_u["image"].to(device)

        with torch.no_grad():
            teacher_logits_l = teacher(x_l)
            teacher_logits_u = teacher(x_u)
            pseudo_l, _ = hard_pseudo_labels(teacher_logits_l)
            pseudo_u, conf_u = hard_pseudo_labels(teacher_logits_u)

        valid_u = conf_u >= confidence_threshold
        x_u_valid = x_u[valid_u]
        pseudo_u_valid = pseudo_u[valid_u]

        optimizer.zero_grad()

        student_logits_l = student(x_l)
        loss_pseudo_labeled_vec = cross_entropy_per_sample(student_logits_l, pseudo_l)
        loss_true_labeled_vec = cross_entropy_per_sample(student_logits_l, y_l)

        if x_u_valid.size(0) > 0:
            student_logits_u = student(x_u_valid)
            loss_pseudo_unlabeled_vec = cross_entropy_per_sample(student_logits_u, pseudo_u_valid)
            pseudo_all_sum = loss_pseudo_labeled_vec.sum() + loss_pseudo_unlabeled_vec.sum()
            total_count = x_l.size(0) + x_u_valid.size(0)
        else:
            pseudo_all_sum = loss_pseudo_labeled_vec.sum()
            total_count = x_l.size(0)

        loss_pseudo_all = pseudo_all_sum / max(total_count, 1)
        loss_pseudo_labeled = loss_pseudo_labeled_vec.mean()
        loss_true_labeled = loss_true_labeled_vec.mean()
        loss = loss_pseudo_all - alpha * (loss_pseudo_labeled - loss_true_labeled)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_l.size(0)
        running_examples += x_l.size(0)

        preds_l = torch.argmax(student_logits_l, dim=1)
        all_preds.extend(preds_l.detach().cpu().tolist())
        all_targets.extend(y_l.detach().cpu().tolist())


    from src.evaluation.metrics import compute_classification_metrics

    metrics = compute_classification_metrics(all_targets, all_preds)
    metrics["loss"] = running_loss / max(running_examples, 1)
    metrics["alpha"] = float(alpha)
    return metrics


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    threshold = (
        args.confidence_threshold
        if args.confidence_threshold is not None
        else float(config["training"].get("confidence_threshold", 0.0))
    )

    label_map = load_json(args.label_map)
    device = get_device()

    labeled_dataset = HAM10000Dataset(
        csv_path=args.labeled_csv,
        label_map=label_map,
        transform=build_transforms(config["dataset"]["image_size"], train=True),
        unlabeled=False,
        image_root=args.image_root,
    )
    unlabeled_dataset = HAM10000Dataset(
        csv_path=args.unlabeled_csv,
        label_map=label_map,
        transform=build_transforms(config["dataset"]["image_size"], train=True),
        unlabeled=True,
        image_root=args.image_root,
    )
    val_dataset = HAM10000Dataset(
        csv_path=args.val_csv,
        label_map=label_map,
        transform=build_transforms(config["dataset"]["image_size"], train=False),
        unlabeled=False,
        image_root=args.image_root,
    )

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
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

    teacher = build_resnet50(
        num_classes=config["dataset"]["num_classes"],
        pretrained=False,
        freeze_backbone=False,
    )
    teacher_ckpt = load_checkpoint(args.teacher_checkpoint, map_location=device)
    teacher.load_state_dict(teacher_ckpt["model_state_dict"])
    teacher.to(device)
    teacher.eval()

    student = build_resnet50(
        num_classes=config["dataset"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        freeze_backbone=config["model"]["freeze_backbone"],
    )
    student.to(device)

    optimizer = create_optimizer(
        student,
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
            "alpha",
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

    best_score = -1.0
    best_state = None

    total_epochs = int(config["training"]["epochs"])
    for epoch in range(1, total_epochs + 1):
        train_metrics = train_drst_epoch(
            student=student,
            teacher=teacher,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            total_epochs=total_epochs,
            confidence_threshold=threshold,
        )
        val_metrics = evaluate_epoch(
            student,
            val_loader,
            criterion=criterion,
            device=device,
            desc=f"DRST Val {epoch}/{total_epochs}",
        )

        logger.log(
            {
                "epoch": epoch,
                "alpha": train_metrics["alpha"],
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            }
        )

        score = val_metrics["macro_f1"]
        if score > best_score:
            best_score = score
            best_state = {
                "model_state_dict": student.state_dict(),
                "val_metrics": val_metrics,
                "epoch": epoch,
            }
            save_checkpoint(best_state, checkpoint_path)

        print(
            f"[Epoch {epoch}/{total_epochs}] "
            f"alpha={train_metrics['alpha']:.3f} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("DRST training finished but no checkpoint was saved.")

    print(f"Saved best DRST checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
