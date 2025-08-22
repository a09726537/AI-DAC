#!/usr/bin/env python3
"""
replay_artifacts.py

Author: William K. (University of Vienna)
Project: AI-DAC — Artificial Intelligence–Driven Anomaly Detection and Control

Purpose
-------
Deterministically replay historical decisions for auditability/compliance.
Given a list of decision IDs and archived artifacts (features, model, policy,
and feature spec), the script:
  1) Filters the feature store to the requested decision IDs
  2) Recomputes scores and actions using the archived model and policy config
  3) Compares replay outputs against logged decisions
  4) Verifies that (model_ver, policy_ver, feat_ver) and config hashes match
  5) Produces JSON and CSV reports (mismatch details, summary stats)

Inputs
------
- Decision IDs file (newline-separated)
- Feature store (CSV/Parquet) containing at least: [event_id, label?, score?, action?]
- Model artifact (pickle | torch state_dict | torchscript | or callable)
- Policy config (YAML) that defines threshold(s) and optional feature masks
- Feature-spec / preprocess state JSON (columns, scaling parameters, etc.)
- Logged decisions CSV for ground truth comparison (optional but recommended)

Outputs
-------
- JSON summary (--out-json)
- CSV with row-by-row replay vs logged decisions (--out-csv)
