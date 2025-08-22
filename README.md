# RDBMS-XAI Experiments

This repository contains configurations, scripts, notebooks, and reproducibility artifacts for the thesis experiments on **Triple Loop Learning (TLL)** and explainability in database security.

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)

## Structure

```
repo/
|-- configs/          # YAML config files for detector, policy, explainer
|-- scripts/          # Shell and Python scripts for experiments
|-- notebooks/        # Jupyter notebooks (analysis, plotting)
|-- tests/            # Unit tests (pytest)
|-- data/             # Public datasets (DS2/DS3); DS1 is access-controlled
|-- .github/workflows # GitHub Actions CI
```

## Quickstart

```bash
conda create -n rdbms-xai python=3.11 -y
conda activate rdbms-xai
pip install -r requirements.txt -r requirements-dev.txt
```

## Common Tasks

- Run prequential pipeline:

  ```bash
  bash scripts/run_prequential.sh --dataset DS2 --seeds "1,7,13,21,42"
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
