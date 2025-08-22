# AI-DAC (RDBMS-XAI) — Experiments & Reproducibility

This repository contains configurations, scripts, notebooks, and artifacts for the thesis experiments on **Triple Loop Learning (TLL)** and explainability for database/security anomaly detection.

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)

> **Author:** William K., University of Vienna

---

## Contents



```
epo/
|-- configs/ # YAML configs (detector, policy, explainer, preprocess)
| |-- detector.yaml
| |-- policy.yaml
| |-- explainer.yaml
| -- preprocess.yaml
|-- scripts/ # Experiment orchestration & utilities
| |-- run_prequential.sh # slide windows; train + eval per window/seed
| |-- run_trust_study.sh # build/analyze No-XAI vs XAI analyst study
| |-- prepare_data.py # preprocessing + anonymization pipeline
| |-- train.py # train + (optional) Platt calib + val thresholding
| |-- eval.py # evaluate model; metrics + predictions
| |-- measure_latency.py # p50/p95 wall-clock latency, batch=1
| -- replay_artifacts.py # integrity/replay checks for decisions/artifacts
|-- notebooks/ # Jupyter (analysis, plots, paper figs)
|-- data/ # DS2/DS3 (public) placeholders; DS1 is access-controlled
|-- tests/ # Unit tests (pytest)
|-- .github/workflows/ # CI workflows (lint/tests)
|-- LICENSE
-- README.md
```


---

## Quickstart

```bash
# Environment
conda create -n rdbms-xai python=3.11 -y
conda activate rdbms-xai
pip install -r requirements.txt -r requirements-dev.txt

```

## Common Tasks

- Run prequential pipeline:

  ```bash
  python scripts/prepare_data.py \
  --input data/raw/postgres_logs.csv \
  --output data/DS1_processed/all.csv \
  --config configs/preprocess.yaml

## Run the prequential pipeline
bash scripts/run_prequential.sh \
  --dataset DS2 \
  --source-csv data/DS2_processed/all.csv \
  --time-col ts --label-col label --id-col event_id \
  --windows "W1:W12" \
  --seeds "1,7,13,21,42" \
  --detector-config configs/detector.yaml \
  --policy-config configs/policy.yaml \
  --outdir artifacts/prequential/DS2
## Outputs per window/seed go to:
artifacts/prequential/DS2/
  windows/Wk/{train,val,test}.csv
  runs/Wk/seed_S/{model.pkl, policy.resolved.yaml, train_metrics.json, metrics.json, predictions.csv, ...}
  aggregate_metrics.csv
## Evaluate a trained run
python scripts/eval.py \
  --test artifacts/prequential/DS2/windows/W5/test.csv \
  --model-dir artifacts/prequential/DS2/runs/W5/seed_1 \
  --policy-config configs/policy.yaml \
  --id-col event_id --label-col label \
  --out-json artifacts/prequential/DS2/runs/W5/seed_1/metrics.json \
  --out-csv  artifacts/prequential/DS2/runs/W5/seed_1/predictions.csv
##  Measure online latency (batch=1)
python scripts/measure_latency.py \
  --model-dir artifacts/prequential/DS2/runs/W5/seed_1 \
  --n-warmup 100 --n-runs 10000 --device cpu
## Run the trust/XAI user study
# Generate study packs (score-only vs. score+top-k SHAP), counterbalanced
bash scripts/run_trust_study.sh generate \
  --dataset DS1 \
  --predictions artifacts/prequential/DS1/runs/W5/seed_1/predictions.csv \
  --id-col event_id --label-col label --score-col score \
  --n 60 --analysts "a01,a02,a03,a04" \
  --policy-config configs/policy.yaml \
  --shap-csv artifacts/prequential/DS1/runs/W5/seed_1/predictions_shap.csv \
  --topk 5 \
  --outdir artifacts/trust_study/DS1_W5_seed1

# Analyze collected responses
bash scripts/run_trust_study.sh analyze \
  --responses-dir artifacts/trust_study/DS1_W5_seed1/responses \
  --outdir        artifacts/trust_study/DS1_W5_seed1

  ```

- Measure latency:

  ```bash
  python scripts/measure_latency.py
  ```

- Run tests and lint:

  ```bash
  pytest -q
  flake8
  black --check .
  isort --check-only .
  ```

## CI Badge

Replace `OWNER/REPO` in the badge URL with your GitHub org and repo name.

## License

MIT License (see LICENSE).
