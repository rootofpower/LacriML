# LacriML — Disease Screening from Tear Microscopy Images

Binary image classifiers that screen **microscopy images of tear samples** for disease. Built at **Hack Kosice 2026** for the challenge set by Pavol Jozef Šafárik University (UPJŠ).

The goal is **screening**: a missed disease (false negative) costs far more than a false alarm. So every model is evaluated by sweeping the decision threshold and choosing operating points with recall close to 1.0, rather than by accuracy alone.

## Results

Held-out test set, operating point *"recall ≥ 0.95 with the best precision"*:

| Task (disease vs healthy) | Test images<br/>disease / healthy | Model | Recall | Precision | Missed (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|
| Any disease | 112 / 112 | B0 ensemble + TTA | **0.982** | **1.000** | 2 | 0 |
| Diabetes (+ dry eye) | 80 / 80 | B0 weighted ensemble + TTA | **0.975** | 0.886 | 2 | 10 |
| PGOV (+ dry eye) | 112 / 112 | B0 ensemble + TTA | **1.000** | **1.000** | 0 | 0 |
| Multiple sclerosis | 112 / 112 | B2 two-seed ensemble + TTA | **0.964** | **1.000** | 4 | 0 |
| Dry eye | 22 / 22 | B0 ensemble + TTA | **1.000** | **1.000** | 0 | 0 |

Full training and evaluation logs for every run are in [`results/`](results/).

> **Caveat.** These are hackathon results on a single train/test split with small test sets (only 44 images for dry eye). They show the approach is promising, not that it is clinically validated. This is not a medical device.

## Approach

The project went through three iterations, each kept in the repo:

**1. Hand-crafted texture features → Random Forest** (`train_pipeline.py`, `test_pipeline.py`)
Baseline using LBP, GLCM, HOG, shape descriptors, Gabor filter banks, box-counting fractal dimension and topological features. These describe the crystalline structure of dried tear samples.

**2. Transfer learning with EfficientNet-B0** (`efficientnet_train.py` → `efficientnet_finetune.py` → `efficientnet_focal.py` → `fn_ensemble.py`)
- train the classifier head on a frozen ImageNet backbone, then fine-tune the deeper layers;
- retrain with **Focal Loss** to focus on hard, misclassified examples;
- average the fine-tuned and focal models (simple / weighted / max) with **30-fold test-time augmentation**;
- sweep thresholds and report three operating points: max recall, recall ≥ 0.95 with best precision, and max F1.

**3. Improved v2 training** (`efficientnet_v2.py`, `fn_ensemble_v2.py`, `run_all.sh`)
- two-phase schedule: warm-up of the frozen backbone, then unfreeze blocks 4–8;
- Focal Loss with per-class α, **Mixup**, label smoothing, cosine annealing with warm restarts;
- **EfficientNet-B2** for the harder pairs and B0 for the easier ones, ensembling models trained with different seeds.

Plus a **data-leakage check** (`check_duplicates.py`) that finds duplicate and near-duplicate images across train and test splits using both file hashes and perceptual pixel hashes.

## Usage

The dataset and trained weights are **not** included: the data belongs to the challenge organizers.

```bash
pip install -r requirements.txt
```

Arrange the data as described in [`config.py`](config.py) (`dataset/<pair>/{learning,test}/...`). Then:

```bash
python check_duplicates.py                 # 0. verify there is no train/test leakage

# v1 pipeline for one pair (chory | diabetes | pgov | skleroza | suche_oko)
python efficientnet_train.py    --pair pgov
python efficientnet_finetune.py --pair pgov
python efficientnet_focal.py    --pair pgov
python fn_ensemble.py           --pair pgov

# v2 pipeline for all pairs
PYTHON=python3 ./run_all.sh
```

### Inference

```bash
python predict.py --pair pgov --image path/to/image_or_folder --mode balanced --output predictions.json
```

| Mode | Threshold | Use case |
|---|---|---|
| `screening` | 0.20 | Catch every possible case, accept more false alarms |
| `balanced` | 0.42 | Default |
| `precision` | 0.49 | Fewer false alarms |

`predict.py` can also be imported (`predict()`, `predict_batch()`). It expects `results/<pair>/efficientnet_{finetuned,focal}_<pair>.pth`, produced by the v1 pipeline.

## Project structure

```
config.py                 # Dataset paths for each disease/healthy pair
check_duplicates.py       # Train/test leakage detection
train_pipeline.py         # Phase 0: hand-crafted features + Random Forest
test_pipeline.py          # Feature pipeline on arbitrary class pairs
efficientnet_train.py     # v1: frozen backbone, head only
efficientnet_finetune.py  # v1: fine-tuning
efficientnet_focal.py     # v1: Focal Loss retraining
fn_ensemble.py            # v1: ensemble + TTA + threshold sweep
fn_reduction.py           # Early experiments on reducing false negatives
efficientnet_v2.py        # v2: Mixup, Focal Loss, B0/B2, seeds
fn_ensemble_v2.py         # v2: ensemble evaluation
run_all.sh                # v2 training for all pairs
predict.py                # Inference CLI / module
check_image.py            # Minimal single-image check
results/                  # Training and evaluation logs
```

## Tech stack

Python · PyTorch · torchvision (EfficientNet B0/B2) · scikit-learn · scikit-image · OpenCV · NumPy · Matplotlib / Seaborn
