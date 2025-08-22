#!/usr/bin/env python3
"""
prepare_data.py

Author: William K. (University of Vienna)
Project: AI-DAC — Artificial Intelligence–Driven Anomaly Detection and Control

Standardized preprocessing & splitting pipeline:
- Categorical one-hot encoding
- 1-minute timestamp bucketing
- Continuous clipping at 99.5th percentile (train-only)
- Z-score standardization with train-only stats
- /24 CIDR mapping for client IPs
- Per-field SHA-256 hashing with secret salts (irreversible)
- Time-ordered 70/15/15 split (temporal), with optional stratified fallback
- Deterministic seeds, versioned artifacts, and full state logging

Usage
-----
python scripts/prepare_data.py \
  --config configs/preprocess.yaml \
  --input data/raw.csv \
  --outdir data/processed \
  --temporal true

Config (YAML) example (configs/preprocess.yaml)
-----------------------------------------------
seed: 42
time_col: ts               # ISO 8601 or epoch seconds
label_col: label           # {0,1}
id_col: event_id
tenant_col: tenant         # optional
ip_cols: [client_cidr]     # columns containing client IPs (v4)
categorical_cols: [user_role, db, schema, object, op_type]
numeric_cols: [rows_affected, duration_ms]
drop_cols: []              # cols to drop before modeling (post-anon)
hash_anonymize_cols: [user_role, db, schema, object]  # per-field salted
hash_salt_env_prefix: "AIDAC_SALT_"     # env var prefix for salts (one per col)
timestamp_bucket: "1min"  # pandas offset alias
clip_quantile: 0.995
temporal: true            # if false, uses stratified (if label available)
stratify: true
val_ratio: 0.15
test_ratio: 0.15
export_intermediate: false
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import hashlib
import ipaddress
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import yaml
except Exception as e:
    print("This script requires PyYAML. Please `pip install pyyaml`.", file=sys.stderr)
    raise

# -----------------------
# Utils & helpers
# -----------------------
def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def sha256_with_salt(value: str, salt: str) -> str:
    h = hashlib.sha256()
    h.update((salt + value).encode("utf-8"))
    return h.hexdigest()

def salt_for_column(col: str, env_prefix: str) -> str:
    """
    Obtain salt from environment: e.g., for col='user_role' and prefix='AIDAC_SALT_',
    read env var AIDAC_SALT_USER_ROLE. If not set, raise (to avoid weak anonymization).
    """
    env_name = f"{env_prefix}{col.upper()}"
    salt = os.getenv(env_name)
    if not salt:
        raise RuntimeError(
            f"Missing salt for column '{col}'. Set environment variable '{env_name}'."
        )
    return salt

def to_cidr24(ip: str) -> str:
    """
    Map IPv4 to /24 CIDR (e.g., 192.168.1.42 -> 192.168.1.0/24).
    Returns empty string if invalid or missing.
    """
    try:
        ip_obj = ipaddress.ip_address(ip)
        if ip_obj.version != 4:
            return ""
        net = ipaddress.ip_network(f"{ip}/24", strict=False)
        return f"{net.network_address}/24"
    except Exception:
        return ""

def ensure_datetime(series: pd.Series) -> pd.Series:
    # Accept ISO 8601 strings or numeric epoch seconds
    if np.issubdtype(series.dtype, np.number):
        return pd.to_datetime(series, unit="s", utc=True)
    return pd.to_datetime(series, utc=True, errors="coerce")

def bucket_timestamp(ts: pd.Series, freq: str = "1min") -> pd.Series:
    return ts.dt.floor(freq)

def compute_quantile_clip(values: pd.Series, q: float) -> float:
    if values.empty:
        return np.nan
    return float(np.nanquantile(values.astype(float), q))

def zscore_fit(x: pd.Series) -> Tuple[float, float]:
    mu = float(np.nanmean(x.astype(float)))
    sigma = float(np.nanstd(x.astype(float), ddof=0))
    # avoid divide-by-zero
    if sigma == 0.0 or np.isnan(sigma):
        sigma = 1.0
    return mu, sigma

def zscore_apply(x: pd.Series, mu: float, sigma: float) -> pd.Series:
    return (x.astype(float) - mu) / sigma

def deterministic_split_temporal(
    df: pd.DataFrame,
    time_col: str,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_val - n_test
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError("Invalid split sizes; adjust val_ratio/test_ratio.")
    train = df_sorted.iloc[:n_train].copy()
    val   = df_sorted.iloc[n_train:n_train + n_val].copy()
    test  = df_sorted.iloc[n_train + n_val:].copy()
    return train, val, test

def deterministic_split_stratified(
    df: pd.DataFrame,
    label_col: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simple stratified split without sklearn; uses per-class shuffling via numpy.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for lbl in sorted(df[label_col].unique()):
        sub = df[df[label_col] == lbl].sample(frac=1.0, random_state=seed)  # shuffle
        n = len(sub)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        n_train = n - n_val - n_test
        if n_train < 0:
            raise ValueError("Stratified split failure; reduce val/test ratios.")
        parts.append((
            sub.iloc[:n_train], sub.iloc[n_train:n_train + n_val], sub.iloc[n_train + n_val:]
        ))
    # concat partitions per split
    train = pd.concat([p[0] for p in parts]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val   = pd.concat([p[1] for p in parts]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test  = pd.concat([p[2] for p in parts]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, val, test

# -----------------------
# Core preprocessing
# -----------------------
def anonymize_columns(df: pd.DataFrame, cols: List[str], salt_prefix: str) -> pd.DataFrame:
    """
    Irreversibly hash sensitive columns with per-field salts (GDPR-compliant).
    Salts are obtained from environment variables (not persisted).
    """
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            warnings.warn(f"[anon] Column '{col}' not present; skipping.")
            continue
        salt = salt_for_column(col, salt_prefix)
        out[col] = out[col].fillna("").astype(str).map(lambda v: sha256_with_salt(v, salt))
    return out

def map_ip_cols_to_cidr(df: pd.DataFrame, ip_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in ip_cols:
        if col not in out.columns:
            warnings.warn(f"[cidr] IP column '{col}' not found; skipping.")
            continue
        out[col] = out[col].fillna("").astype(str).map(to_cidr24)
    return out

def one_hot_encode(df: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Fit-less one-hot using pandas.get_dummies, but we will capture categories
    present in TRAIN and later align VAL/TEST to the same columns.
    """
    ohe = pd.get_dummies(df[categorical_cols].astype("category"), dummy_na=False)
    out = pd.concat([df.drop(columns=categorical_cols), ohe], axis=1)
    # record one-hot columns by source prefix
    mapping: Dict[str, List[str]] = {}
    for c in categorical_cols:
        # columns start with 'c_' pattern 'colname_value'
        prefix = f"{c}_"
        cols = [col for col in out.columns if col.startswith(prefix)]
        mapping[c] = cols
    return out, mapping

def align_ohe_columns(df: pd.DataFrame, ohe_map: Dict[str, List[str]]) -> pd.DataFrame:
    out = df.copy()
    for src_col, cols in ohe_map.items():
        for c in cols:
            if c not in out.columns:
                out[c] = 0
    # drop any stray dummies not seen in train
    allowed = set(df.columns) | {c for cols in ohe_map.values() for c in cols}
    # Reorder columns deterministically
    out = out[[c for c in sorted(out.columns) if c in allowed]]
    return out

def clip_and_standardize(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    numeric_cols: List[str],
    clip_q: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Compute train-only clip and z-score stats; apply to all splits.
    Returns transformed splits + state dict.
    """
    state: Dict[str, Dict[str, float]] = {}
    t_train, t_val, t_test = train.copy(), val.copy(), test.copy()

    for col in numeric_cols:
        if col not in t_train.columns:
            warnings.warn(f"[scale] Numeric column '{col}' not in TRAIN; skipping.")
            continue
        clip_val = compute_quantile_clip(t_train[col], clip_q)
        # clip
        for frame in (t_train, t_val, t_test):
            if col in frame.columns:
                frame[col] = frame[col].astype(float).clip(upper=clip_val)

        mu, sigma = zscore_fit(t_train[col])
        for frame in (t_train, t_val, t_test):
            if col in frame.columns:
                frame[col] = zscore_apply(frame[col], mu, sigma)

        state[col] = {"clip_q": clip_q, "clip_val": clip_val, "mu": mu, "sigma": sigma}
    return t_train, t_val, t_test, state

# -----------------------
# Pipeline
# -----------------------
def run_pipeline(cfg: dict, input_csv: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    time_col   = cfg["time_col"]
    label_col  = cfg.get("label_col", None)
    id_col     = cfg.get("id_col", None)
    tenant_col = cfg.get("tenant_col", None)

    ip_cols = cfg.get("ip_cols", []) or []
    categorical_cols = cfg.get("categorical_cols", []) or []
    numeric_cols = cfg.get("numeric_cols", []) or []
    drop_cols = cfg.get("drop_cols", []) or []

    anon_cols = cfg.get("hash_anonymize_cols", []) or []
    salt_prefix = cfg.get("hash_salt_env_prefix", "AIDAC_SALT_")
    ts_bucket = cfg.get("timestamp_bucket", "1min")
    clip_q = float(cfg.get("clip_quantile", 0.995))

    temporal = bool(cfg.get("temporal", True))
    stratify = bool(cfg.get("stratify", True))
    val_ratio = float(cfg.get("val_ratio", 0.15))
    test_ratio = float(cfg.get("test_ratio", 0.15))
    export_intermediate = bool(cfg.get("export_intermediate", False))

    # Load raw
    df = pd.read_csv(input_csv)
    n0 = len(df)

    # Basic sanitation
    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not present in input.")
    if label_col and label_col not in df.columns:
        warnings.warn(f"label_col '{label_col}' not found; proceeding unlabeled.")
        label_col = None

    # Timestamp to UTC and bucket
    df[time_col] = ensure_datetime(df[time_col])
    df[time_col] = bucket_timestamp(df[time_col], ts_bucket)

    # IP -> CIDR /24
    if ip_cols:
        df = map_ip_cols_to_cidr(df, ip_cols)

    # Anonymize sensitive columns (irreversible hashing with per-field salts)
    if anon_cols:
        df = anonymize_columns(df, anon_cols, salt_prefix)

    # Prepare split
    if temporal:
        train_df, val_df, test_df = deterministic_split_temporal(df, time_col, val_ratio, test_ratio)
    else:
        if label_col is None or not stratify:
            warnings.warn("Non-temporal split without labels or stratify=false; using random split.")
            # random split
            df_shuf = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            n = len(df_shuf)
            n_test = int(round(n * test_ratio))
            n_val = int(round(n * val_ratio))
            n_train = n - n_val - n_test
            train_df, val_df, test_df = df_shuf.iloc[:n_train], df_shuf.iloc[n_train:n_train+n_val], df_shuf.iloc[n_train+n_val:]
        else:
            train_df, val_df, test_df = deterministic_split_stratified(df, label_col, val_ratio, test_ratio, seed)

    # Optional intermediate export
    if export_intermediate:
        train_df.to_csv(os.path.join(outdir, "train_raw.csv"), index=False)
        val_df.to_csv(os.path.join(outdir, "val_raw.csv"), index=False)
        test_df.to_csv(os.path.join(outdir, "test_raw.csv"), index=False)

    # One-hot encode categoricals on TRAIN; align VAL/TEST
    ohe_map: Dict[str, List[str]] = {}
    if categorical_cols:
        train_ohe, ohe_map = one_hot_encode(train_df, categorical_cols)
        # we need to apply same OHE mapping to VAL/TEST
        # re-run get_dummies, then align columns
        val_tmp = pd.get_dummies(val_df[categorical_cols].astype("category"), dummy_na=False)
        test_tmp = pd.get_dummies(test_df[categorical_cols].astype("category"), dummy_na=False)

        val_ohe = pd.concat([val_df.drop(columns=categorical_cols), val_tmp], axis=1)
        test_ohe = pd.concat([test_df.drop(columns=categorical_cols), test_tmp], axis=1)

        # Align to train's OHE columns
        train_df = train_ohe
        val_df = align_ohe_columns(val_ohe, ohe_map)
        test_df = align_ohe_columns(test_ohe, ohe_map)

    # Clip & standardize numeric features with train-only stats
    train_df, val_df, test_df, scale_state = clip_and_standardize(
        train_df, val_df, test_df, numeric_cols, clip_q
    )

    # Drop columns after anonymization/prep if requested
    drop_final = [c for c in drop_cols if c in train_df.columns]
    if drop_final:
        train_df = train_df.drop(columns=drop_final)
        val_df = val_df.drop(columns=drop_final)
        test_df = test_df.drop(columns=drop_final)

    # Final ordering: (optional) id, time, tenant, label, others
    col_order = []
    for c in [id_col, time_col, tenant_col, label_col]:
        if c and c in train_df.columns:
            col_order.append(c)
    # add rest sorted for determinism
    rest = [c for c in sorted(train_df.columns) if c not in col_order]
    train_df = train_df[col_order + rest]
    val_df   = val_df[[c for c in train_df.columns if c in val_df.columns]]  # align
    # For test, add any missing columns with zeros (possible if some OHE cols absent)
    for c in train_df.columns:
        if c not in val_df.columns:
            val_df[c] = 0
    test_missing = [c for c in train_df.columns if c not in test_df.columns]
    for c in test_missing:
        test_df[c] = 0
    test_df = test_df[[c for c in train_df.columns]]

    # Export processed splits
    train_path = os.path.join(outdir, "train.csv")
    val_path   = os.path.join(outdir, "val.csv")
    test_path  = os.path.join(outdir, "test.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Save preprocessing state (for exact replay)
    state = {
        "seed": seed,
        "input_csv": os.path.abspath(input_csv),
        "n_rows_raw": n0,
        "time_col": time_col,
        "label_col": label_col,
        "id_col": id_col,
        "tenant_col": tenant_col,
        "ip_cols": ip_cols,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "drop_cols": drop_cols,
        "timestamp_bucket": ts_bucket,
        "clip_quantile": clip_q,
        "temporal": temporal,
        "stratify": stratify,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "ohe_map": ohe_map,
        "scaling_state": scale_state,
        "export_intermediate": export_intermediate,
        "outputs": {
            "train_csv": os.path.abspath(train_path),
            "val_csv": os.path.abspath(val_path),
            "test_csv": os.path.abspath(test_path),
        },
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "version": "1.0.0",
    }
    with open(os.path.join(outdir, "preprocess_state.json"), "w") as f:
        json.dump(state, f, indent=2)

    print("[prepare_data] Completed.")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")
    print(f"  State: {os.path.join(outdir, 'preprocess_state.json')}")


# -----------------------
# CLI
# -----------------------
def main():
    p = argparse.ArgumentParser(description="Standardized preprocessing & splitting.")
    p.add_argument("--config", required=True, help="YAML config path (see docstring).")
    p.add_argument("--input", required=True, help="Input CSV.")
    p.add_argument("--outdir", required=True, help="Output directory for processed splits.")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    run_pipeline(cfg, args.input, args.outdir)


if __name__ == "__main__":
    main()
