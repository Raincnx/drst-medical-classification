from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_classification_metrics
from src.utils.io import ensure_dir


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_optimizer(model: nn.Module, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def save_checkpoint(state: Dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict:
    return torch.load(path, map_location=map_location)


def train_supervised_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Train",
) -> Dict[str, float]:
    model.train()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    metrics = compute_classification_metrics(all_targets, all_preds)
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)
    return metrics


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Eval",
) -> Dict[str, float]:
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in tqdm(loader, desc=desc, leave=False):
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
    return metrics


def fit_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    checkpoint_path: str | Path,
    logger=None,
) -> Tuple[nn.Module, Dict[str, float]]:
    best_val_metric = -1.0
    best_state = None
    best_metrics: Dict[str, float] = {}

    for epoch in range(1, epochs + 1):
        train_metrics = train_supervised_epoch(
            model, train_loader, optimizer, criterion, device, desc=f"Train {epoch}/{epochs}"
        )
        val_metrics = evaluate_epoch(
            model, val_loader, criterion, device, desc=f"Val {epoch}/{epochs}"
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        }
        if logger is not None:
            logger.log(row)

        score = val_metrics["macro_f1"]
        if score > best_val_metric:
            best_val_metric = score
            best_metrics = copy.deepcopy(val_metrics)
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "val_metrics": best_metrics,
                "epoch": epoch,
            }
            save_checkpoint(best_state, checkpoint_path)

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training finished but no checkpoint was saved.")

    model.load_state_dict(best_state["model_state_dict"])
    return model, best_metrics
