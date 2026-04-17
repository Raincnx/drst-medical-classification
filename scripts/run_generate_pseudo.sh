#!/usr/bin/env bash
set -euo pipefail

python -m src.pseudo_label.generate_pseudo_labels \
  --config configs/ham10000_resnet50.yaml \
  --checkpoint outputs/checkpoints/supervised_10_best.pt \
  --input-csv data/splits/train_unlabeled_90.csv \
  --label-map data/processed/label_map.json \
  --output-csv data/processed/pseudo_labels_10.csv
