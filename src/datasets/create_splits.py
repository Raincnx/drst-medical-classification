from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Create HAM10000 train/val/test and low-label splits.")
    parser.add_argument("--metadata-csv", type=str, required=True, help="Path to HAM10000_metadata.csv")
    parser.add_argument("--image-root", type=str, required=True, help="Path to image directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Where split CSV files will be saved")
    parser.add_argument("--label-map-out", type=str, required=True, help="Where label_map.json will be saved")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--labeled-ratios", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def add_paths(df: pd.DataFrame, image_root: str | Path) -> pd.DataFrame:
    image_root = Path(image_root)
    df = df.copy()
    df["image_path"] = df["image_id"].apply(lambda x: str(image_root / f"{x}.jpg"))
    return df


def save_split(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    metadata = pd.read_csv(args.metadata_csv)
    if "image_id" not in metadata.columns or "dx" not in metadata.columns:
        raise ValueError("Expected HAM10000 metadata to contain 'image_id' and 'dx' columns.")

    df = metadata[["image_id", "dx"]].rename(columns={"dx": "label"})
    df = add_paths(df, args.image_root)

    labels_sorted = sorted(df["label"].unique().tolist())
    label_map = {label: idx for idx, label in enumerate(labels_sorted)}
    save_json(label_map, args.label_map_out)

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - args.train_ratio),
        stratify=df["label"],
        random_state=args.seed,
    )

    relative_test_ratio = args.test_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["label"],
        random_state=args.seed,
    )

    output_dir = Path(args.output_dir)
    save_split(train_df, output_dir / "train.csv")
    save_split(val_df, output_dir / "val.csv")
    save_split(test_df, output_dir / "test.csv")

    for labeled_ratio in args.labeled_ratios:
        labeled_df, unlabeled_df = train_test_split(
            train_df,
            train_size=labeled_ratio,
            stratify=train_df["label"],
            random_state=args.seed,
        )
        labeled_pct = int(round(labeled_ratio * 100))
        unlabeled_pct = 100 - labeled_pct
        save_split(labeled_df, output_dir / f"train_labeled_{labeled_pct}.csv")
        save_split(unlabeled_df, output_dir / f"train_unlabeled_{unlabeled_pct}.csv")

    print("Saved splits to:", output_dir)
    print("Saved label map to:", args.label_map_out)


if __name__ == "__main__":
    main()
