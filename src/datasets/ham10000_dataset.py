from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_transforms(image_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class HAM10000Dataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        label_map: Dict[str, int],
        transform: Optional[Callable] = None,
        unlabeled: bool = False,
        use_pseudo_labels: bool = False,
        image_root: str | Path | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self.label_map = label_map
        self.transform = transform
        self.unlabeled = unlabeled
        self.use_pseudo_labels = use_pseudo_labels
        self.image_root = Path(image_root) if image_root else None

        required_columns = {"image_id"}
        if not required_columns.issubset(self.df.columns):
            raise ValueError(
                f"CSV must contain columns {required_columns}. Got: {self.df.columns.tolist()}"
            )

        if "image_path" not in self.df.columns and self.image_root is None:
            raise ValueError(
                "CSV must contain image_path or you must provide --image-root."
            )

        if not self.unlabeled and not self.use_pseudo_labels and "label" not in self.df.columns:
            raise ValueError("Labeled dataset CSV must contain 'label' column.")

        if self.use_pseudo_labels and "pseudo_label_idx" not in self.df.columns:
            raise ValueError("Pseudo-labeled dataset CSV must contain 'pseudo_label_idx' column.")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, row: pd.Series) -> Path:
        if "image_path" in row and isinstance(row["image_path"], str) and row["image_path"]:
            path = Path(row["image_path"])
            if path.exists():
                return path
            if self.image_root is not None:
                alt_path = self.image_root / path.name
                if alt_path.exists():
                    return alt_path
            return path

        if self.image_root is None:
            raise ValueError("Cannot resolve image path without image_root.")
        return self.image_root / f"{row['image_id']}.jpg"

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = self._resolve_image_path(row)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        item = {
            "image": image,
            "image_id": row["image_id"],
            "image_path": str(image_path),
        }

        if self.unlabeled:
            return item

        if self.use_pseudo_labels:
            item["label"] = int(row["pseudo_label_idx"])
            item["confidence"] = float(row.get("confidence", 1.0))
            return item

        label_name = str(row["label"])
        if label_name not in self.label_map:
            raise KeyError(f"Label '{label_name}' not found in label_map.")
        item["label"] = self.label_map[label_name]
        return item
