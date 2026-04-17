#!/usr/bin/env bash
set -euo pipefail

python -m src.training.train_self_training \
  --config configs/ham10000_resnet50.yaml \
  --labeled-csv data/splits/train_labeled_10.csv \
  --pseudo-csv data/processed/pseudo_labels_10.csv \
  --val-csv data/splits/val.csv \
  --label-map data/processed/label_map.json \
  --run-name self_training_10
