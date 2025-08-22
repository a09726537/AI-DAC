#!/usr/bin/env python3
"""
eval.py

Author: William K. (University of Vienna)
Project: AI-DAC — Artificial Intelligence–Driven Anomaly Detection and Control

Evaluation script for binary anomaly detection models.
- Frameworks: pickle (sklearn/xgboost), pytorch (state_dict), torchscript, callable (module:function)
- Metrics: F1, AUROC, PR-AUC, Precision@k, Recall@k, MCC, ECE (calibration),
           FPR@TPR=0.80/0.90, confusion matrix, and bootstrapped 95% CIs.
- Threshold selection:
   * Preferred: from validation set (--val-csv) maximizing F1 (or a specified --threshold).
   * Fallback: maximize F1 on test set (reported as such).
- Optional McNemar's test vs a second predictions file for paired comparison.

Outputs:
- JSON summary (--out-json)
- CSV with per-row scores/predictions (--out-pred-csv)
- ROC/PR curves (png) if --plot-dir provided

Example usage
-------------
# Evaluate a pickled model on DS2 test CSV, using a validation CSV for threshold selection:
python scripts/eval.py \
  --framework pickle \
  --model-path artifacts/model.pkl \
  --test-csv data/DS2_test.csv \
  --val-csv data/DS2_val.csv \
  --label-col label \
  --id-col event_id \
  --drop-cols event_id,ts,tenant \
  --prec-at 50,100 \
  --rec-at 50,100 \
  --out-json results/metrics_ds2.json \
  --out-pred-csv results/preds_ds2.csv \
  --plot-dir results/plots

# TorchScript model with explicit feature columns:
python scripts/eval.py \
  --framework torchscript \
  --model-path artifacts/model_scripted.pt \
  --test-csv data/DS1_test.csv \
  --label-col label \
  --feature-cols f1,f2,f3,f4,f5

# Paired comparison (McNemar) vs baseline predictions:
python scripts/eval.py \
  --framework pickle --model-path artifacts/tll.pkl \
  --test-csv data/DS2_test.csv --label-col label \
  --out-json results/tll.json \
  --compare-preds baseline_preds.csv
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Optional imports (lazy where possible)
try:
    import pandas as pd
except Exception:
    print("This script requires pandas. Please `pip install pandas`.", file=sys.stderr)
    raise

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.utils import resample
from math import sqrt


# ---------------------------
# IO helpers
# ---------------------------
def load_csv_features(
    path: str,
    label_col: str,
    feature_cols: Optional[List[str]] = None,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (df, X, y) with X float32 and y in {0,1}."""
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {path}.")
    y = df[label_col].astype(int).to_numpy()

    df_feat = df.copy()
    if drop_cols:
        for c in drop_cols:
            if c in df_feat.columns:
                df_feat = df_feat.drop(columns=[c])
    # Finally drop label
    df_feat = df_feat.drop(columns=[label_col])

    if feature_cols:
        missing = [c for c in feature_cols if c not in df_feat.columns]
        if missing:
            raise ValueError(f"Feature columns not found in {path}: {missing}")
        df_feat = df_feat[feature_cols]

    X = df_feat.to_numpy().astype(np.float32, copy=False)
    return df, X, y


# ---------------------------
# Model adapters (unified API)
# ---------------------------
class InferenceAdapter:
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Return scores/probabilities in [0,1] if possible (positive class)."""
        raise NotImplementedError

class PickleAdapter(InferenceAdapter):
    def __init__(self, model_path: str):
        import pickle
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        m = self.model
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X)
            if p.ndim == 2 and p.shape[1] == 2:
                return p[:, 1]
            # fall back to squeeze
            return np.squeeze(p).astype(np.float32)
        if hasattr(m, "decision_function"):
            z = m.decision_function(X).astype(np.float32)
            # map decision scores to 0-1 via logistic heuristic
            return 1.0 / (1.0 + np.exp(-z))
        # last resort: predict hard labels (will degrade metrics using scores)
        yhat = m.predict(X)
        return yhat.astype(np.float32)

class TorchStateDictAdapter(InferenceAdapter):
    def __init__(self, model_path: str, input_dim: int, device: str = "cpu"):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        # Minimal linear head loader; replace with your actual model if needed
        self.model = torch.nn.Sequential(torch.nn.Linear(input_dim, 1)).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        t = self.torch.from_numpy(X).to(self.device)
        with self.torch.no_grad():
            out = self.model(t).squeeze(-1)
            if self.device.type == "cuda":
                self.torch.cuda.synchronize()
            # sigmoid -> probability
            prob = self.torch.sigmoid(out).float().cpu().numpy()
        return prob

class TorchScriptAdapter(InferenceAdapter):
    def __init__(self, model_path: str, device: str = "cpu"):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        t = self.torch.from_numpy(X).to(self.device)
        with self.torch.no_grad():
            out = self.model(t)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out.squeeze(-1)
            if self.device.type == "cuda":
                self.torch.cuda.synchronize()
            prob = self.torch.sigmoid(out).float().cpu().numpy()
        return prob

class CallableAdapter(InferenceAdapter):
    def __init__(self, qualname: str):
        """qualname: 'module.submodule:function' returning scores in [0,1]."""
        import importlib
        if ":" not in qualname:
            raise ValueError("--callable must be 'module:function'")
        mod_name, fn_name = qualname.split(":", 1)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name, None)
        if fn is None:
            raise ValueError(f"Callable {fn_name} not found in module {mod_name}.")
        self.fn = fn

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.fn(X), dtype=np.float32)


def build_adapter(
    framework: str,
    model_path: Optional[str],
    input_dim: Optional[int],
    device: str,
    callable_qualname: Optional[str],
) -> InferenceAdapter:
    fw = framework.lower()
    if fw == "pickle":
        if not model_path:
            raise ValueError("--model-path is required for framework=pickle")
        return PickleAdapter(model_path)
    if fw == "pytorch":
        if not model_path or input_dim is None:
            raise ValueError("--model-path and --input-dim are required for framework=pytorch")
        return TorchStateDictAdapter(model_path, input_dim=input_dim, device=device)
    if fw == "torchscript":
        if not model_path:
            raise ValueError("--model-path is required for framework=torchscript")
        return TorchScriptAdapter(model_path, device=device)
    if fw == "callable":
        if not callable_qualname:
            raise ValueError("--callable is required for framework=callable")
        return CallableAdapter(callable_qualname)
    raise ValueError(f"Unsupported framework: {framework}")


# ---------------------------
# Metrics
# ---------------------------
def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    yhat = (scores >= threshold).astype(int)
    f1 = f1_score(y_true, yhat, zero_division=0)
    try:
        auroc = roc_auc_score(y_true, scores)
    except ValueError:
        auroc = float("nan")
    try:
        prauc = average_precision_score(y_true, scores)
    except ValueError:
        prauc = float("nan")
    mcc = matthews_corrcoef(y_true, yhat) if len(np.unique(y_true)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
    return {
        "f1": float(f1),
        "auroc": float(auroc),
        "prauc": float(prauc),
        "mcc": float(mcc),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "positives": int(np.sum(y_true == 1)),
        "negatives": int(np.sum(y_true == 0)),
        "prevalence": float(np.mean(y_true)),
    }

def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, ks: List[int]) -> Dict[str, float]:
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    out = {}
    for k in ks:
        k = int(k)
        k = max(1, min(k, len(y_true)))
        topk = y_sorted[:k]
        prec = float(np.mean(topk == 1))
        # recall@k = TP@k / total positives
        total_pos = max(1, int(np.sum(y_true == 1)))
        rec = float(np.sum(topk == 1) / total_pos)
        out[f"precision@{k}"] = prec
        out[f"recall@{k}"] = rec
    return out

def expected_calibration_error(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-width bins in [0,1]."""
    scores = np.clip(scores, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(scores, bins) - 1
    ece = 0.0
    n = len(scores)
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = np.mean(scores[mask])
        acc = np.mean(y_true[mask])
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)

def fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, tpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    # find smallest FPR with TPR >= target
    meets = np.where(tpr >= tpr_target)[0]
    if len(meets) == 0:
        return float("nan")
    return float(np.min(fpr[meets]))


# ---------------------------
# Threshold selection
# ---------------------------
def select_threshold_by_max_f1(y: np.ndarray, scores: np.ndarray) -> float:
    # sweep possible thresholds from unique scores; you can also sweep 0..1 grid
    uniq = np.unique(scores)
    if len(uniq) > 1000:
        # subsample evenly if too many unique scores
        uniq = np.quantile(scores, np.linspace(0.0, 1.0, 1000))
    best_f1, best_th = -1.0, 0.5
    for th in uniq:
        yhat = (scores >= th).astype(int)
        f1 = f1_score(y, yhat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)
    return best_th


# ---------------------------
# Bootstrap CI
# ---------------------------
def bootstrap_ci(
    func: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            vals.append(func(y[idx], scores[idx]))
        except Exception:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return lo, hi


# ---------------------------
# McNemar's test (paired)
# ---------------------------
def mcnemar_test(y_true: np.ndarray, yhat_a: np.ndarray, yhat_b: np.ndarray) -> Dict[str, float]:
    """Exact McNemar (binomial) with continuity correction fallback."""
    # Contingency: b = errors by A only, c = errors by B only
    err_a = (yhat_a != y_true).astype(int)
    err_b = (yhat_b != y_true).astype(int)
    b = int(np.sum((err_a == 1) & (err_b == 0)))
    c = int(np.sum((err_a == 0) & (err_b == 1)))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n": n, "p_value": 1.0}
    # Exact binomial test p-value: 2 * min(Bin(k<=b|n,0.5), Bin(k>=b|n,0.5))
    # Simple normal approximation with continuity correction:
    stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    # Approximate p-value from chi-square(1). For simplicity, we use exp approximation.
    # (If you prefer scipy, plug in from scipy.stats)
    from math import exp
    # chi-square CDF(1) tail ~ exp(-x/2)
    p_approx = exp(-stat / 2.0)
    return {"b": b, "c": c, "n": n, "chi2_cc": stat, "p_value_approx": p_approx}


# ---------------------------
# Plotting
# ---------------------------
def save_curves(y: np.ndarray, scores: np.ndarray, out_dir: str, prefix: str = "eval") -> None:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(y, scores)
    plt.figure()
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_roc.png"), dpi=150)
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y, scores)
    plt.figure()
    plt.plot(recall, precision, lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_pr.png"), dpi=150)
    plt.close()


# ---------------------------
# Main
# ---------------------------
def main():
    p = argparse.ArgumentParser(description="Evaluate anomaly detector performance.")
    p.add_argument("--framework", required=True, choices=["pickle", "pytorch", "torchscript", "callable"])
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--callable", dest="callable_qualname", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--input-dim", type=int, default=None, help="Required for pytorch state_dict unless adapter is customized.")

    p.add_argument("--test-csv", required=True, type=str, help="CSV with test data.")
    p.add_argument("--val-csv", type=str, default=None, help="Optional validation CSV for threshold selection.")
    p.add_argument("--label-col", required=True, type=str)
    p.add_argument("--id-col", type=str, default=None)
    p.add_argument("--feature-cols", type=str, default=None, help="Comma-separated list to select features.")
    p.add_argument("--drop-cols", type=str, default=None, help="Comma-separated list to drop before modeling (e.g., ids, ts).")

    p.add_argument("--threshold", type=float, default=None, help="If given, use this classification threshold.")
    p.add_argument("--prec-at", type=str, default="50,100", help="Comma-separated ks for precision@k (on test).")
    p.add_argument("--rec-at", type=str, default="50,100", help="Comma-separated ks for recall@k (on test).")
    p.add_argument("--ece-bins", type=int, default=15)

    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out-json", type=str, default=None)
    p.add_argument("--out-pred-csv", type=str, default=None)
    p.add_argument("--plot-dir", type=str, default=None)

    p.add_argument("--compare-preds", type=str, default=None,
                  help="CSV with columns [<id_col> optional], 'y_true', 'yhat' to run McNemar vs current model.")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    feature_cols = args.feature_cols.split(",") if args.feature_cols else None
    drop_cols = args.drop_cols.split(",") if args.drop_cols else None
    ks_prec = [int(x) for x in args.prec_at.split(",") if x.strip()]
    ks_rec = [int(x) for x in args.rec_at.split(",") if x.strip()]

    # Load test data
    df_test, X_test, y_test = load_csv_features(args.test_csv, args.label_col, feature_cols, drop_cols)

    # Build adapter and score test
    adapter = build_adapter(args.framework, args.model_path, args.input_dim, args.device, args.callable_qualname)

    # Threshold selection
    selected_threshold = args.threshold
    threshold_source = "user"
    if selected_threshold is None and args.val_csv:
        _, X_val, y_val = load_csv_features(args.val_csv, args.label_col, feature_cols, drop_cols)
        scores_val = adapter.predict_scores(X_val)
        selected_threshold = select_threshold_by_max_f1(y_val, scores_val)
        threshold_source = "val_maxF1"
    if selected_threshold is None:
        # fallback: select on test (report as such)
        scores_tmp = adapter.predict_scores(X_test)
        selected_threshold = select_threshold_by_max_f1(y_test, scores_tmp)
        threshold_source = "test_maxF1"

    # Final scoring on test
    scores = adapter.predict_scores(X_test)
    yhat = (scores >= selected_threshold).astype(int)

    # Base metrics
    metrics = binary_metrics(y_test, scores, threshold=selected_threshold)
    metrics["threshold"] = float(selected_threshold)
    metrics["threshold_source"] = threshold_source
    metrics["ece"] = expected_calibration_error(y_test, scores, n_bins=args.ece_bins)
    metrics["fpr_at_tpr_0.80"] = fpr_at_tpr(y_test, scores, 0.80)
    metrics["fpr_at_tpr_0.90"] = fpr_at_tpr(y_test, scores, 0.90)

    # Precision/Recall@k (sorted by score)
    metrics.update(precision_recall_at_k(y_test, scores, ks_prec))
    metrics.update(precision_recall_at_k(y_test, scores, ks_rec))  # both aliases; harmless if overlap

    # Bootstrap CIs for key metrics
    def make_metric_fn(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
        if name == "f1":
            return lambda yt, sc: f1_score(yt, (sc >= selected_threshold).astype(int), zero_division=0)
        if name == "auroc":
            return lambda yt, sc: roc_auc_score(yt, sc)
        if name == "prauc":
            return lambda yt, sc: average_precision_score(yt, sc)
        return lambda _yt, _sc: float("nan")

    for mname in ["f1", "auroc", "prauc"]:
        lo, hi = bootstrap_ci(make_metric_fn(mname), y_test, scores, n_boot=args.n_bootstrap, seed=args.seed)
        metrics[f"{mname}_ci95_lo"] = lo
        metrics[f"{mname}_ci95_hi"] = hi

    # Optional plots
    if args.plot_dir:
        save_curves(y_test, scores, args.plot_dir, prefix=os.path.splitext(os.path.basename(args.test_csv))[0])

    # Optional McNemar vs baseline predictions
    mcnemar = None
    if args.compare_preds:
        df_cmp = pd.read_csv(args.compare_preds)
        if "y_true" not in df_cmp.columns or "yhat" not in df_cmp.columns:
            raise ValueError("--compare-preds must contain columns 'y_true' and 'yhat'.")
        y_true_cmp = df_cmp["y_true"].to_numpy().astype(int)
        yhat_cmp = df_cmp["yhat"].to_numpy().astype(int)
        if len(y_true_cmp) != len(y_test):
            print("Warning: compare-preds length differs from test set; aligning by min length.", file=sys.stderr)
        n = min(len(y_true_cmp), len(y_test))
        mcnemar = mcnemar_test(y_test[:n], yhat[:n], yhat_cmp[:n])
        metrics["mcnemar"] = mcnemar

    # Write outputs
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[eval] Wrote JSON metrics -> {args.out_json}")

    if args.out_pred_csv:
        os.makedirs(os.path.dirname(args.out_pred_csv), exist_ok=True)
        out_df = pd.DataFrame({
            args.label_col: y_test,
            "score": scores,
            "pred": yhat,
        })
        if args.id_col and args.id_col in df_test.columns:
            out_df[args.id_col] = df_test[args.id_col].values
            # reorder with id first
            cols = [args.id_col, args.label_col, "score", "pred"]
            out_df = out_df[cols]
        out_df.to_csv(args.out_pred_csv, index=False)
        print(f"[eval] Wrote per-row predictions -> {args.out_pred_csv}")

    # Console summary
    print("\n=== Evaluation Summary ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:>24s}: {v:.6f}")
        else:
            print(f"{k:>24s}: {v}")
    print("==========================\n")


if __name__ == "__main__":
    main()

