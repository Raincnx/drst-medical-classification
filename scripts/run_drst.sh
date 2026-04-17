#!/usr/bin/env bash
set -euo pipefail

python -m src.training.train_drst \
  --config configs/ham10000_resnet50.yaml \
  --teacher-checkpoint outputs/checkpoints/supervised_10_best.pt \
  --labeled-csv data/splits/train_labeled_10.csv \
  --unlabeled-csv data/splits/train_unlabeled_90.csv \
  --val-csv data/splits/val.csv \
  --label-map data/processed/label_map.json \
  --run-name drst_10
