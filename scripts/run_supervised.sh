#!/usr/bin/env bash
set -euo pipefail

python -m src.training.train_supervised \
  --config configs/ham10000_resnet50.yaml \
  --train-csv data/splits/train_labeled_10.csv \
  --val-csv data/splits/val.csv \
  --label-map data/processed/label_map.json \
  --run-name supervised_10
