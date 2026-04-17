#!/usr/bin/env bash
set -euo pipefail

python -m src.datasets.create_splits \
  --metadata-csv data/raw/HAM10000_metadata.csv \
  --image-root data/raw/images \
  --output-dir data/splits \
  --label-map-out data/processed/label_map.json \
  --labeled-ratios 0.1 0.2 \
  --seed 42
