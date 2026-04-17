# DRST Medical Classification Starter

This repository is a starter framework for a 3-person class project on **Doubly Robust Self-Training (DRST)** for low-label medical image classification.

## Project goal

Compare three settings on **HAM10000** with the same **ResNet50** backbone:

1. **Supervised-only**
2. **Standard self-training**
3. **Doubly Robust Self-Training (DRST)**

Main question:

> Is DRST more robust than standard self-training when labeled data is scarce and pseudo-labels are noisy?

## Recommended dataset

- **HAM10000** Skin Lesion Dataset

Expected file layout:

```text
data/
  raw/
    HAM10000_metadata.csv
    images/
      ISIC_0000000.jpg
      ...
```

If your image folder has a different name, pass `--image-root` explicitly.

## Repository structure

```text
drst-medical-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── ham10000_resnet50.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── datasets/
│   │   ├── create_splits.py
│   │   └── ham10000_dataset.py
│   ├── models/
│   │   └── resnet50.py
│   ├── training/
│   │   ├── common.py
│   │   ├── train_supervised.py
│   │   ├── train_self_training.py
│   │   └── train_drst.py
│   ├── pseudo_label/
│   │   └── generate_pseudo_labels.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluate.py
│   └── utils/
│       ├── config.py
│       ├── io.py
│       ├── logger.py
│       └── seed.py
├── scripts/
│   ├── run_create_splits.sh
│   ├── run_supervised.sh
│   ├── run_generate_pseudo.sh
│   ├── run_self_training.sh
│   └── run_drst.sh
└── report/
    └── outline.md
```

## Quick start

### 1) Create environment

```bash
conda create -n drst python=3.10 -y
conda activate drst
pip install -r requirements.txt
```

### 2) Prepare splits

```bash
python -m src.datasets.create_splits \
  --metadata-csv data/raw/HAM10000_metadata.csv \
  --image-root data/raw/images \
  --output-dir data/splits \
  --label-map-out data/processed/label_map.json \
  --labeled-ratios 0.1 0.2 \
  --seed 42
```

### 3) Train supervised teacher

```bash
python -m src.training.train_supervised \
  --config configs/ham10000_resnet50.yaml \
  --train-csv data/splits/train_labeled_10.csv \
  --val-csv data/splits/val.csv \
  --label-map data/processed/label_map.json \
  --run-name supervised_10
```

### 4) Generate pseudo-labels

```bash
python -m src.pseudo_label.generate_pseudo_labels \
  --config configs/ham10000_resnet50.yaml \
  --checkpoint outputs/checkpoints/supervised_10_best.pt \
  --input-csv data/splits/train_unlabeled_90.csv \
  --label-map data/processed/label_map.json \
  --output-csv data/processed/pseudo_labels_10.csv
```

### 5) Run standard self-training

```bash
python -m src.training.train_self_training \
  --config configs/ham10000_resnet50.yaml \
  --labeled-csv data/splits/train_labeled_10.csv \
  --pseudo-csv data/processed/pseudo_labels_10.csv \
  --val-csv data/splits/val.csv \
  --label-map data/processed/label_map.json \
  --run-name self_training_10
```

### 6) Run DRST

```bash
python -m src.training.train_drst \
  --config configs/ham10000_resnet50.yaml \
  --teacher-checkpoint outputs/checkpoints/supervised_10_best.pt \
  --labeled-csv data/splits/train_labeled_10.csv \
  --unlabeled-csv data/splits/train_unlabeled_90.csv \
  --val-csv data/splits/val.csv \
  --label-map data/processed/label_map.json \
  --run-name drst_10
```

### 7) Evaluate on test set

```bash
python -m src.evaluation.evaluate \
  --config configs/ham10000_resnet50.yaml \
  --checkpoint outputs/checkpoints/drst_10_best.pt \
  --test-csv data/splits/test.csv \
  --label-map data/processed/label_map.json \
  --split-name drst_10_test \
  --plot-confusion
```

## Suggested team split

- **Person 1**: dataset prep + splits + supervised baseline
- **Person 2**: teacher + pseudo-label generation + standard self-training
- **Person 3**: DRST + evaluation + report tables/figures

## Notes

- This starter uses **ResNet50** from `torchvision`.
- The DRST implementation uses a **curriculum coefficient** `alpha = epoch / num_epochs`.
- The code is designed to be easy to modify, not overly optimized.
