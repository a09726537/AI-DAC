# AI-DAC (RDBMS-XAI) — Experiments & Reproducibility

This repository contains configurations, scripts, notebooks, and artifacts for the thesis experiments on **Triple Loop Learning (TLL)** and explainability for database/security anomaly detection.

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)

> **Author:** William K., University of Vienna

---

## Contents

```
repo/
|-- configs/                  # YAML configs (detector, policy, explainer, preprocess)
|   |-- detector.yaml         # who uses it (train.py), GPU hint, imbalance tip, and alternatives
|   |-- policy.yaml           # threshold auto-tuning behavior, lock semantics, and doc-only knobs.
|   |-- explainer.yaml        # TreeSHAP/RAG settings, budgets, caching, privacy masks.
|   \-- preprocess.yaml       # hashing salt env var, safety notes, and pipeline steps.
|-- scripts/                  # Experiment orchestration & utilities
|   |-- run_prequential.sh    # slide windows; train + eval per window/seed
|   |-- run_trust_study.sh    # build/analyze No-XAI vs XAI analyst study
|   |-- prepare_data.py       # preprocessing + anonymization pipeline
|   |-- train.py              # train + (optional) Platt calib + val thresholding
|   |-- eval.py               # evaluate model; metrics + predictions
|   |-- measure_latency.py    # p50/p95 wall-clock latency, batch=1
|   \-- replay_artifacts.py   # integrity/replay checks for decisions/artifacts
|-- notebooks/                # Jupyter (analysis, plots, paper figs)
|-- data/                     # DS2/DS3 (public) placeholders; DS1 is access-controlled
|-- tests/                    # Unit tests (pytest)
|-- .github/workflows/        # CI workflows (lint/tests)
|-- LICENSE
\-- README.md
```

---

## Quickstart

```bash
conda create -n rdbms-xai python=3.11 -y
conda activate rdbms-xai
pip install -r requirements.txt -r requirements-dev.txt
```

### Prepare data

```bash
python scripts/prepare_data.py \
  --input data/raw/postgres_logs.csv \
  --output data/DS1_processed/all.csv \
  --config configs/preprocess.yaml
```

### Run the prequential pipeline

```bash
bash scripts/run_prequential.sh \
  --dataset DS2 \
  --source-csv data/DS2_processed/all.csv \
  --time-col ts --label-col label --id-col event_id \
  --windows "W1:W12" \
  --seeds "1,7,13,21,42" \
  --detector-config configs/detector.yaml \
  --policy-config configs/policy.yaml \
  --outdir artifacts/prequential/DS2
```

### Evaluate a trained run

```bash
python scripts/eval.py \
  --test artifacts/prequential/DS2/windows/W5/test.csv \
  --model-dir artifacts/prequential/DS2/runs/W5/seed_1 \
  --policy-config configs/policy.yaml \
  --id-col event_id --label-col label \
  --out-json artifacts/prequential/DS2/runs/W5/seed_1/metrics.json \
  --out-csv  artifacts/prequential/DS2/runs/W5/seed_1/predictions.csv
```

### Measure online latency (batch=1)

```bash
python scripts/measure_latency.py \
  --model-dir artifacts/prequential/DS2/runs/W5/seed_1 \
  --sample-csv artifacts/prequential/DS2/windows/W5/test.csv \
  --id-col event_id --label-col label \
  --n-warmup 100 --n-runs 10000 --device cpu
```

---

## Configuration Overview

- `configs/detector.yaml` (example)
  ```yaml
  detector:
    type: xgboost
    params:
      max_depth: 8
      n_estimators: 600
      learning_rate: 0.05
      subsample: 0.8
      colsample_bytree: 0.8
    calibration: platt
  ```
- `configs/policy.yaml`
  ```yaml
  threshold: 0.5
  lock_threshold: false
  ```
- `configs/preprocess.yaml`
  ```yaml
  time_bucket_minutes: 1
  clip_quantile: 0.995
  standardize: true
  map_ip_to_cidr: "/24"
  hash_salt_env: "ANON_SALT"
  ```

---

## Reproducibility

- Fixed seeds; version-pinned dependencies; artifact checksums.
- Artifacts per run: `model.pkl`, `policy.resolved.yaml`, `train_metrics.json`, `model_meta.json`, `feature_cols.json`, `metrics.json`, `predictions.csv`.
- Replay: `scripts/replay_artifacts.py` re-scores immutable decisions for audit.

---

## Data

- **DS1** (enterprise PostgreSQL logs): access-controlled; anonymized.
- **DS2** (TPC+inj): public, staged drift.
- **DS3** (adv/synth): public MAD-GAN sequences.

---

## Code & Availability

- **Repository:** https://github.com/a09726537/AI-DAC (MIT License)
