from __future__ import annotations

from typing import Dict, List

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def compute_confusion_matrix(y_true: List[int], y_pred: List[int]):
    return confusion_matrix(y_true, y_pred)
