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
- Exit code 0 if no mismatches; non-zero if mismatches or verification failure

Examples
--------
python scripts/replay_artifacts.py \
  --decision-ids file:decision_ids.txt \
  --feature-store data/DS1_processed/test.csv \
  --framework pickle --model-path artifacts/model.pkl \
  --policy-yaml configs/policy.yaml \
  --feat-spec data/DS1_processed/preprocess_state.json \
  --logged-decisions logs/decisions_ds1.csv \
  --id-col event_id --label-col label \
  --out-json results/replay_summary.json \
  --out-csv results/replay_diffs.csv

# TorchScript + callable policy (module:function returning action given score)
python scripts/replay_artifacts.py \
  --decision-ids "abc,def,ghi" \
  --feature-store data/DS2_processed/test.parquet \
  --framework torchscript --model-path artifacts/model_scripted.pt \
  --callable-policy mypkg.policy:decide \
  --feat-spec data/DS2_processed/preprocess_state.json \
  --id-col event_id \
  --out-json results/replay.json --out-csv results/replay.csv
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:
    print("This script requires pandas. Please `pip install pandas`.", file=sys.stderr)
    raise

# YAML is optional (policy config); only needed if --policy-yaml is provided.
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# ---------------------------
# Helpers
# ---------------------------
def read_decision_ids(spec: str) -> List[str]:
    """
    spec can be:
      - 'file:path/to/ids.txt' (one ID per line)
      - 'id1,id2,id3'
    """
    if spec.startswith("file:"):
        path = spec.split("file:", 1)[1]
        with open(path, "r") as f:
            return [ln.strip() for ln in f if ln.strip()]
    return [x.strip() for x in spec.split(",") if x.strip()]

def load_feat_spec(path: Optional[str]) -> Dict:
    if not path:
        return {}
    with open(path, "r") as f:
        return json.load(f)

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def yaml_load_file(path: str) -> Dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read --policy-yaml.")
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ---------------------------
# Model Adapters
# ---------------------------
class InferenceAdapter:
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
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
            return p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else np.squeeze(p).astype(np.float32)
        if hasattr(m, "decision_function"):
            z = m.decision_function(X).astype(np.float32)
            return 1.0 / (1.0 + np.exp(-z))
        return m.predict(X).astype(np.float32)

class TorchStateDictAdapter(InferenceAdapter):
    def __init__(self, model_path: str, input_dim: int, device: str = "cpu"):
        import torch
        self.torch = torch
        self.device = torch.device(device)
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


def build_adapter(framework: str,
                  model_path: Optional[str],
                  input_dim: Optional[int],
                  device: str,
                  callable_qualname: Optional[str]) -> InferenceAdapter:
    fw = framework.lower()
    if fw == "pickle":
        if not model_path:
            raise ValueError("--model-path required for framework=pickle")
        return PickleAdapter(model_path)
    if fw == "pytorch":
        if not model_path or input_dim is None:
            raise ValueError("--model-path and --input-dim required for framework=pytorch")
        return TorchStateDictAdapter(model_path, input_dim=input_dim, device=device)
    if fw == "torchscript":
        if not model_path:
            raise ValueError("--model-path required for framework=torchscript")
        return TorchScriptAdapter(model_path, device=device)
    if fw == "callable":
        if not callable_qualname:
            raise ValueError("--callable required for framework=callable")
        return CallableAdapter(callable_qualname)
    raise ValueError(f"Unsupported framework: {framework}")

# ---------------------------
# Policy
# ---------------------------
def load_policy(policy_yaml: Optional[str], callable_qualname: Optional[str]):
    """
    Return a function action = policy_fn(score, row) that produces a discrete action
    (e.g., 'allow'/'alert' or 0/1). Policy precedence:
      1) callable module:function if provided
      2) YAML thresholds: either single 'threshold' or dict of 'thresholds'
         Optionally supports 'feature_masks': {mask_name: [col1, col2, ...]}
    """
    if callable_qualname:
        import importlib
        mod_name, fn_name = callable_qualname.split(":", 1)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name, None)
        if fn is None:
            raise ValueError(f"Policy callable {fn_name} not found in {mod_name}.")
        return fn, {"type": "callable"}

    if policy_yaml:
        cfg = yaml_load_file(policy_yaml)
        thr = cfg.get("threshold", None)
        thrs = cfg.get("thresholds", None)
        if thr is None and not thrs:
            # Default policy: 0.5 cutoff
            thr = 0.5

        def policy_fn(score: float, row: Optional[pd.Series] = None):
            # If multiple thresholds keyed by tenant or class, choose based on row fields
            if isinstance(thrs, dict) and row is not None:
                # Heuristic: try tenant- or schema-specific thresholds
                key = str(row.get("tenant", "") or row.get("schema", "") or "")
                t = thrs.get(key, thr if thr is not None else 0.5)
                return int(score >= float(t))
            # Single threshold
            return int(score >= float(thr))

        return policy_fn, {"type": "yaml", "threshold": thr, "thresholds": thrs}

    # Default
    return (lambda score, row=None: int(score >= 0.5)), {"type": "default", "threshold": 0.5}

# ---------------------------
# Replay
# ---------------------------
def select_columns_for_model(df: pd.DataFrame,
                             id_col: str,
                             label_col: Optional[str],
                             feature_cols: Optional[List[str]],
                             drop_cols: Optional[List[str]]) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
    """Prepare X (float32) and optional y from df according to provided columns."""
    work = df.copy()
    if drop_cols:
        keep = [c for c in work.columns if c not in set(drop_cols)]
        work = work[keep]

    if feature_cols:
        missing = [c for c in feature_cols if c not in work.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        X = work[feature_cols].to_numpy(dtype=np.float32, copy=False)
    else:
        # use all non-id/label columns as features
        cols = [c for c in work.columns if c not in {id_col} | ({label_col} if label_col else set())]
        X = work[cols].to_numpy(dtype=np.float32, copy=False)

    y = None
    if label_col and label_col in work.columns:
        y = work[label_col].to_numpy().astype(int)
    return work, X, y

def replay(df_feats: pd.DataFrame,
           df_logged: Optional[pd.DataFrame],
           adapter: InferenceAdapter,
           policy_fn,
           id_col: str,
           label_col: Optional[str],
           feature_cols: Optional[List[str]],
           drop_cols: Optional[List[str]],
           model_ver: Optional[str],
           policy_ver: Optional[str],
           feat_ver: Optional[str],
           expected_model_ver: Optional[str],
           expected_policy_ver: Optional[str],
           expected_feat_ver: Optional[str],
           policy_hash: Optional[str],
           expected_policy_hash: Optional[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Returns a DataFrame with:
      [event_id, score_replay, action_replay, score_logged?, action_logged?, match?]
    And a summary dict with mismatch counts and version/hash checks.
    """
    rows, X, _ = select_columns_for_model(df_feats, id_col, label_col, feature_cols, drop_cols)
    scores = adapter.predict_scores(X)
    actions = []
    for s, (_, row) in zip(scores, rows.iterrows()):
        actions.append(int(policy_fn(float(s), row)))
    rows = rows.copy()
    rows["score_replay"] = scores
    rows["action_replay"] = np.asarray(actions, dtype=int)

    summary = {
        "n_replayed": int(len(rows)),
        "n_logged_present": 0,
        "n_score_mismatch": 0,
        "n_action_mismatch": 0,
        "version_check": {
            "model_ver_expected": expected_model_ver,
            "model_ver_logged": model_ver,
            "policy_ver_expected": expected_policy_ver,
            "policy_ver_logged": policy_ver,
            "feat_ver_expected": expected_feat_ver,
            "feat_ver_logged": feat_ver,
            "policy_yaml_hash_expected": expected_policy_hash,
            "policy_yaml_hash_logged": policy_hash,
            "ok": True,
        },
    }

    if df_logged is not None:
        # Expect columns: event_id, score (logged), action (logged), model_ver, policy_ver, feat_ver, policy_yaml_hash?
        df_logged_idx = df_logged.set_index(id_col, drop=False)
        sc_mismatch, ac_mismatch, logged_present = 0, 0, 0
        sc_eps = 1e-6  # tolerance for floating diffs due to numeric libs

        logged_scores = []
        logged_actions = []
        for _, r in rows.iterrows():
            eid = r[id_col]
            if eid in df_logged_idx.index:
                logged_present += 1
                rlog = df_logged_idx.loc[eid]
                # Handle multi-index duplication if any by taking first
                if isinstance(rlog, pd.DataFrame):
                    rlog = rlog.iloc[0]
                # Attach logged for export
                logged_scores.append(float(rlog.get("score", np.nan)))
                logged_actions.append(int(rlog.get("action", -1)))
                # Compare
                s_log = float(rlog.get("score", np.nan))
                a_log = int(rlog.get("action", -1))
                if not (np.isfinite(s_log) and abs(r["score_replay"] - s_log) <= sc_eps):
                    sc_mismatch += 1
                if int(r["action_replay"]) != a_log:
                    ac_mismatch += 1
            else:
                logged_scores.append(np.nan)
                logged_actions.append(-1)

        rows["score_logged"] = logged_scores
        rows["action_logged"] = logged_actions
        rows["score_match"] = np.isclose(rows["score_replay"], rows["score_logged"], atol=sc_eps, rtol=0.0)
        rows["action_match"] = (rows["action_replay"] == rows["action_logged"])

        summary["n_logged_present"] = int(logged_present)
        summary["n_score_mismatch"] = int(sc_mismatch)
        summary["n_action_mismatch"] = int(ac_mismatch)

        # Version/hash checks pulled from the first logged row (assuming uniform)
        if {"model_ver", "policy_ver", "feat_ver"}.issubset(df_logged.columns):
            logged_model_ver = str(df_logged["model_ver"].iloc[0])
            logged_policy_ver = str(df_logged["policy_ver"].iloc[0])
            logged_feat_ver = str(df_logged["feat_ver"].iloc[0])
        else:
            logged_model_ver = model_ver
            logged_policy_ver = policy_ver
            logged_feat_ver = feat_ver

        ver_ok = True
        if expected_model_ver and expected_model_ver != logged_model_ver:
            ver_ok = False
        if expected_policy_ver and expected_policy_ver != logged_policy_ver:
            ver_ok = False
        if expected_feat_ver and expected_feat_ver != logged_feat_ver:
            ver_ok = False

        pol_ok = True
        if expected_policy_hash and policy_hash and expected_policy_hash != policy_hash:
            pol_ok = False

        summary["version_check"].update({
            "model_ver_logged": logged_model_ver,
            "policy_ver_logged": logged_policy_ver,
            "feat_ver_logged": logged_feat_ver,
            "ok": bool(ver_ok and pol_ok),
        })

    return rows, summary

# ---------------------------
# CLI
# ---------------------------
def main():
    p = argparse.ArgumentParser(description="Deterministic decision replay for audit.")
    # Inputs
    p.add_argument("--decision-ids", required=True,
                   help="Either 'file:path.txt' or comma-separated IDs.")
    p.add_argument("--feature-store", required=True,
                   help="CSV or Parquet with features (must include --id-col).")
    p.add_argument("--framework", required=True,
                   choices=["pickle", "pytorch", "torchscript", "callable"])
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--input-dim", type=int, default=None,
                   help="For pytorch state_dict adapter.")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--callable", dest="callable_qualname", type=str, default=None,
                   help="For framework=callable: 'module:function' returning scores.")
    # Policy
    p.add_argument("--policy-yaml", type=str, default=None,
                   help="YAML with 'threshold' or 'thresholds' mapping.")
    p.add_argument("--callable-policy", type=str, default=None,
                   help="Policy callable 'module:function' mapping (score,row)->action.")
    # Columns
    p.add_argument("--id-col", required=True)
    p.add_argument("--label-col", type=str, default=None)
    p.add_argument("--feature-cols", type=str, default=None,
                   help="Comma-separated list; default is all except id/label.")
    p.add_argument("--drop-cols", type=str, default=None,
                   help="Comma-separated columns to drop before scoring.")
    # Optional logged decisions for comparison
    p.add_argument("--logged-decisions", type=str, default=None,
                   help="CSV with columns: id, score, action, model_ver, policy_ver, feat_ver.")
    # Feat spec / versions
    p.add_argument("--feat-spec", type=str, default=None,
                   help="preprocess_state.json (records feat_ver, scaling, etc.)")
    p.add_argument("--expected-model-ver", type=str, default=None)
    p.add_argument("--expected-policy-ver", type=str, default=None)
    p.add_argument("--expected-feat-ver", type=str, default=None)
    p.add_argument("--expected-policy-yaml-hash", type=str, default=None)
    # Outputs
    p.add_argument("--out-json", type=str, default=None)
    p.add_argument("--out-csv", type=str, default=None)

    args = p.parse_args()

    # Load decision IDs
    ids = read_decision_ids(args.decision_ids)
    if not ids:
        print("[replay] No decision IDs provided.", file=sys.stderr)
        sys.exit(2)

    # Feature store
    if args.feature_store.lower().endswith(".parquet"):
        df_feats = pd.read_parquet(args.feature_store)
    else:
        df_feats = pd.read_csv(args.feature_store)

    # Filter to requested IDs
    if args.id_col not in df_feats.columns:
        raise ValueError(f"--id-col '{args.id_col}' not found in feature store.")
    df_feats = df_feats[df_feats[args.id_col].isin(ids)].copy()
    if len(df_feats) == 0:
        print("[replay] No matching rows for provided decision IDs.", file=sys.stderr)
        sys.exit(3)

    # Prepare model adapter
    feature_cols = args.feature_cols.split(",") if args.feature_cols else None
    drop_cols = args.drop_cols.split(",") if args.drop_cols else None
    adapter = build_adapter(args.framework, args.model_path, args.input_dim, args.device, args.callable_qualname)

    # Policy
    policy_fn, policy_meta = load_policy(args.policy_yaml, args.callable_policy)

    # Optional logged decisions
    df_logged = None
    if args.logged_decisions:
        df_logged = pd.read_csv(args.logged_decisions)

    # Version info
    feat_spec = load_feat_spec(args.feat_spec)
    feat_ver_logged = str(feat_spec.get("version", "")) if feat_spec else None
    # Hash policy YAML if present
    policy_yaml_hash = sha256_file(args.policy_yaml) if args.policy_yaml and os.path.exists(args.policy_yaml) else None

    # Replay
    rows, summary = replay(
        df_feats=df_feats,
        df_logged=df_logged,
        adapter=adapter,
        policy_fn=policy_fn,
        id_col=args.id_col,
        label_col=args.label_col,
        feature_cols=feature_cols,
        drop_cols=drop_cols,
        model_ver=None,
        policy_ver=None,
        feat_ver=feat_ver_logged,
        expected_model_ver=args.expected_model_ver,
        expected_policy_ver=args.expected_policy_ver,
        expected_feat_ver=args.expected_feat_ver or feat_ver_logged,
        policy_hash=policy_yaml_hash,
        expected_policy_hash=args.expected_policy_yaml_hash
    )

    # Summarize results
    any_mismatch = False
    if "n_score_mismatch" in summary and summary["n_score_mismatch"] > 0:
        any_mismatch = True
    if "n_action_mismatch" in summary and summary["n_action_mismatch"] > 0:
        any_mismatch = True
    if "version_check" in summary and not summary["version_check"]["ok"]:
        any_mismatch = True

    # Write outputs
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        # Keep a focused set of columns for auditors
        keep_cols = [args.id_col, "score_replay", "action_replay"]
        for c in ("score_logged", "action_logged", "score_match", "action_match"):
            if c in rows.columns:
                keep_cols.append(c)
        rows[keep_cols].to_csv(args.out_csv, index=False)
        print(f"[replay] Wrote row-level diff CSV -> {args.out_csv}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "summary": summary,
                "policy_meta": policy_meta,
                "policy_yaml_hash": policy_yaml_hash,
                "inputs": {
                    "decision_ids_count": len(ids),
                    "feature_store": os.path.abspath(args.feature_store),
                    "framework": args.framework,
                    "model_path": args.model_path,
                    "policy_yaml": args.policy_yaml,
                    "feat_spec": args.feat_spec,
                    "logged_decisions": args.logged_decisions,
                    "feature_cols": feature_cols,
                    "drop_cols": drop_cols,
                }
            }, f, indent=2)
        print(f"[replay] Wrote summary JSON -> {args.out_json}")

    # Console report
    print("\n=== Replay Summary ===")
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"  - {kk}: {vv}")
        else:
            print(f"{k}: {v}")
    print("======================\n")

    # Exit non-zero if mismatch
    sys.exit(1 if any_mismatch else 0)


if __name__ == "__main__":
    main()
