#!/usr/bin/env python3
"""
train.py

Author: William K. (University of Vienna)
Project: AI-DAC — Artificial Intelligence–Driven Anomaly Detection and Control

Train a tabular detector on train.csv and validate on val.csv, then:
- (optional) calibrate probabilities on the validation set (Platt / sigmoid)
- auto-select a decision threshold on validation (F1 by default)
- persist artifacts into --outdir:
    - model.pkl (pickle: base or calibrated estimator)
    - policy.resolved.yaml (includes chosen threshold)
    - train_metrics.json (val metrics + threshold choice)
    - model_meta.json (config hashes, feature columns, seeds, etc.)
    - feature_cols.json (list for eval/replay)

Usage (as called by run_prequential.sh):
  python scripts/train.py \
    --train path/to/train.csv \
    --val path/to/val.csv \
    --detector-config configs/detector.yaml \
    --policy-config configs/policy.yaml \
    --seed 42 \
    --device cpu \
    --outdir artifacts/prequential/DS2/runs/W5/seed_42

Optional flags:
  --id-col event_id --label-col label --time-col ts
  --feature-cols "f1,f2,..."
  --drop-cols "ts,raw_text"
  --auto-threshold f1            # or: mcc, youden, tpr@0.90
  --no-calibration               # disable Platt/“sigmoid” calibration
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# sklearn is standard; xgboost is optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state
import joblib

try:
    import xgboost as xgb  # type: ignore
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# -------------------------------
# Utility
# -------------------------------
def sha256_file(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Optional[str]) -> Dict:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML configs.")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def to_float_list(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [c.strip() for c in s.split(",") if c.strip()]


def select_columns(
    df: pd.DataFrame,
    id_col: str,
    label_col: str,
    feature_cols: Optional[List[str]],
    drop_cols: Optional[List[str]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    work = df.copy()
    if drop_cols:
        work = work[[c for c in work.columns if c not in set(drop_cols)]]
    if feature_cols:
        missing = [c for c in feature_cols if c not in work.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        feats = feature_cols
    else:
        feats = [c for c in work.columns if c not in {id_col, label_col}]
    X = work[feats].to_numpy(dtype=np.float32, copy=False)
    y = work[label_col].to_numpy().astype(int)
    return X, y, feats


def ece_binary(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (binary) with equal-width bins."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = p[mask].mean()
        acc = y_true[mask].mean()
        ece += (np.sum(mask) / len(p)) * abs(acc - conf)
    return float(ece)


def find_threshold(
    y_true: np.ndarray,
    p: np.ndarray,
    strategy: str = "f1",
) -> float:
    """
    Pick a threshold on validation scores `p` using:
      - 'f1' (default)
      - 'mcc'
      - 'youden' (max TPR - FPR)
      - 'tpr@X' (e.g., 'tpr@0.90')
    """
    s = strategy.lower().strip()
    if s.startswith("tpr@"):
        try:
            target = float(s.split("@", 1)[1])
        except Exception:
            target = 0.90
        # sweep thresholds; choose highest TPR >= target with smallest FPR
        thresh_grid = np.unique(np.clip(p, 1e-7, 1 - 1e-7))
        best_t, best_fpr = 0.5, 1.0
        for t in thresh_grid:
            yhat = (p >= t).astype(int)
            tp = np.sum((yhat == 1) & (y_true == 1))
            fp = np.sum((yhat == 1) & (y_true == 0))
            fn = np.sum((yhat == 0) & (y_true == 1))
            tn = np.sum((yhat == 0) & (y_true == 0))
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            if tpr >= target and fpr <= best_fpr:
                best_t, best_fpr = float(t), float(fpr)
        return float(best_t)

    # grid sweep
    thresh_grid = np.unique(np.clip(p, 1e-7, 1 - 1e-7))
    best_t, best_v = 0.5, -1e9
    for t in thresh_grid:
        yhat = (p >= t).astype(int)
        if s == "mcc":
            v = matthews_corrcoef(y_true, yhat)
        elif s == "youden":
            # youden J = TPR - FPR
            tp = np.sum((yhat == 1) & (y_true == 1))
            fp = np.sum((yhat == 1) & (y_true == 0))
            fn = np.sum((yhat == 0) & (y_true == 1))
            tn = np.sum((yhat == 0) & (y_true == 0))
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            v = tpr - fpr
        else:
            v = f1_score(y_true, yhat, zero_division=0)
        if v > best_v:
            best_v, best_t = float(v), float(t)
    return float(best_t)


# -------------------------------
# Model builders
# -------------------------------
def build_model(det_cfg: Dict, seed: int):
    det = (det_cfg or {}).get("detector", {})
    det_type = str(det.get("type", "xgboost")).lower()
    params = det.get("params", {}) or {}
    rng = check_random_state(seed)

    if det_type in ("xgboost", "xgb"):
        if _HAVE_XGB:
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                **params,
            )
            return model, "xgboost"
        # fallback
        model = GradientBoostingClassifier(random_state=seed)
        return model, "sklearn_gbdt"

    if det_type in ("gbdt", "gradient_boosting"):
        model = GradientBoostingClassifier(random_state=seed, **params)
        return model, "sklearn_gbdt"

    if det_type in ("logreg", "logistic", "logistic_regression"):
        model = LogisticRegression(
            random_state=seed,
            max_iter=params.pop("max_iter", 1000),
            **params,
        )
        return model, "sklearn_logreg"

    # default fallback
    model = GradientBoostingClassifier(random_state=seed)
    return model, "sklearn_gbdt"


# -------------------------------
# Training flow
# -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train detector with val-based threshold selection.")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--detector-config", required=True)
    ap.add_argument("--policy-config", required=False, default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")  # reserved for future torch usage

    ap.add_argument("--id-col", type=str, default="event_id")
    ap.add_argument("--label-col", type=str, default="label")
    ap.add_argument("--time-col", type=str, default="ts")

    ap.add_argument("--feature-cols", type=str, default=None,
                    help="Comma-separated feature column list; default: all except id/label.")
    ap.add_argument("--drop-cols", type=str, default=None,
                    help="Comma-separated columns to drop before training.")

    ap.add_argument("--auto-threshold", type=str, default="f1",
                    help="Threshold strategy: f1 (default), mcc, youden, tpr@0.90")
    ap.add_argument("--no-calibration", action="store_true",
                    help="Disable Platt/sigmoid calibration on the validation set")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load configs
    det_cfg = load_yaml(args.detector_config)
    pol_cfg = load_yaml(args.policy_config) if args.policy_config else {}

    # Read data
    df_tr = pd.read_csv(args.train)
    df_va = pd.read_csv(args.val)

    feature_cols = to_float_list(args.feature_cols)
    drop_cols = to_float_list(args.drop_cols)

    X_tr, y_tr, feats = select_columns(df_tr, args.id_col, args.label_col, feature_cols, drop_cols)
    X_va, y_va, _ = select_columns(df_va, args.id_col, args.label_col, feature_cols, drop_cols)

    # Build and fit model
    model, framework = build_model(det_cfg, seed=args.seed)
    model.fit(X_tr, y_tr)

    # Raw validation probabilities
    def _predict_proba(m, X):
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X)
            return p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else np.squeeze(p)
        if hasattr(m, "decision_function"):
            z = m.decision_function(X).astype(np.float32)
            return 1.0 / (1.0 + np.exp(-z))
        # last resort
        out = m.predict(X)
        if out.ndim == 1:
            return out.astype(np.float32)
        return out[:, 1].astype(np.float32)

    p_va_raw = _predict_proba(model, X_va).astype(np.float32)
    # Calibration (Platt) on validation set (prefit=True)
    calibrated = None
    if not args.no_calibration:
        # Use sigmoid (Platt). To avoid reusing val for both calibration & selection,
        # we keep it deterministic and clearly documented.
        try:
            calibrated = CalibratedClassifierCV(model, cv="prefit", method="sigmoid")
            calibrated.fit(X_va, y_va)
            p_va = _predict_proba(calibrated, X_va).astype(np.float32)
            cal_method = "platt_sigmoid"
        except Exception as e:
            print(f"[WARN] Calibration failed ({e}); using raw probabilities.")
            calibrated = None
            p_va = p_va_raw
            cal_method = "none"
    else:
        p_va = p_va_raw
        cal_method = "none"

    # Threshold selection on validation
    threshold = find_threshold(y_va, p_va, strategy=args.auto_threshold)

    # Validation metrics at chosen threshold
    yhat = (p_va >= threshold).astype(int)
    metrics = {
        "auc_roc": float(roc_auc_score(y_va, p_va)) if len(np.unique(y_va)) > 1 else float("nan"),
        "auc_pr": float(average_precision_score(y_va, p_va)) if len(np.unique(y_va)) > 1 else float("nan"),
        "f1": float(f1_score(y_va, yhat, zero_division=0)),
        "precision": float(precision_score(y_va, yhat, zero_division=0)),
        "recall": float(recall_score(y_va, yhat, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_va, yhat)) if len(np.unique(y_va)) > 1 else float("nan"),
        "ece": float(ece_binary(y_va, p_va)),
        "threshold_strategy": args.auto_threshold,
        "threshold": float(threshold),
        "calibration": cal_method,
        "framework": framework,
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
    }

    # Persist model (calibrated if available)
    model_to_save = calibrated if calibrated is not None else model
    model_path = os.path.join(args.outdir, "model.pkl")
    joblib.dump(model_to_save, model_path)

    # Persist policy.resolved.yaml (use threshold from val unless policy explicitly locks it)
    resolved_policy = dict(pol_cfg) if pol_cfg else {}
    # Respect explicit lock if present; else write chosen threshold
    locked = str(resolved_policy.get("lock_threshold", "false")).lower() in {"1", "true", "yes"}
    if not locked:
        resolved_policy["threshold"] = float(threshold)
    resolved_policy_path = os.path.join(args.outdir, "policy.resolved.yaml")
    if yaml is not None:
        with open(resolved_policy_path, "w") as f:
            yaml.safe_dump(resolved_policy, f, sort_keys=False)

    # Persist feature columns
    with open(os.path.join(args.outdir, "feature_cols.json"), "w") as f:
        json.dump({"feature_cols": feats}, f, indent=2)

    # Persist metrics + meta
    with open(os.path.join(args.outdir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    meta = {
        "seed": int(args.seed),
        "id_col": args.id_col,
        "label_col": args.label_col,
        "time_col": args.time_col,
        "feature_cols_file": "feature_cols.json",
        "model_file": "model.pkl",
        "policy_resolved": os.path.basename(resolved_policy_path),
        "detector_config": os.path.abspath(args.detector_config),
        "policy_config": os.path.abspath(args.policy_config) if args.policy_config else None,
        "detector_config_sha256": sha256_file(args.detector_config),
        "policy_config_sha256": sha256_file(args.policy_config) if args.policy_config else None,
        "git_commit": os.environ.get("GIT_COMMIT", None),
        "framework": framework,
        "calibration": cal_method,
        "threshold_strategy": args.auto_threshold,
        "threshold": float(threshold),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "xgboost": getattr(__import__("sys"), "modules", {}).get("xgboost", type("X", (), {"__version__": "N/A"})).__version__
                if _HAVE_XGB else "N/A",
        },
    }
    with open(os.path.join(args.outdir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("[train] saved:")
    print(f"  - {model_path}")
    if yaml is not None:
        print(f"  - {resolved_policy_path}")
    print(f"  - {os.path.join(args.outdir, 'train_metrics.json')}")
    print(f"  - {os.path.join(args.outdir, 'model_meta.json')}")
    print(f"  - {os.path.join(args.outdir, 'feature_cols.json')}")


if __name__ == "__main__":
    main()

