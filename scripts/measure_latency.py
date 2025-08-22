#!/usr/bin/env python3
"""
measure_latency.py

Author: Dr. William Kandolo
Affiliation: University of Vienna

Wall-clock inference latency benchmarking for batch=1.
- Warm-up: 100 inferences (excluded from stats)
- Measured runs: 10,000 (default; configurable)
- Reports: mean, median, p95, p99, min, max (ms) + total throughput
- Frameworks: sklearn/xgboost (pickle), pytorch (state_dict), torchscript,
  or a custom Python callable via --callable module:function

Usage examples
--------------
# 1) Sklearn/XGBoost pickled model with CSV inputs:
python scripts/measure_latency.py \
  --framework pickle \
  --model-path artifacts/model.pkl \
  --input-csv data/DS2_sample.csv \
  --feature-cols event_id,ts,tenant,label --drop-cols event_id,ts,tenant,label \
  --runs 10000 --warmup 100

# 2) PyTorch state_dict with input dim (synthetic random inputs):
python scripts/measure_latency.py \
  --framework pytorch \
  --model-path artifacts/bilstm.pt \
  --input-dim 128 --runs 10000 --warmup 100

# 3) TorchScript model and CSV inputs (select feature columns by name):
python scripts/measure_latency.py \
  --framework torchscript \
  --model-path artifacts/model_scripted.pt \
  --input-csv data/DS2_sample.csv \
  --feature-cols f1,f2,f3,f4

# 4) Custom callable (expects module:function that accepts np.ndarray shape (1, D)):
python scripts/measure_latency.py \
  --framework callable \
  --callable mypkg.infer:predict \
  --input-dim 64
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
import time
import statistics
from typing import Callable, Optional, Tuple, List

# Lazy imports to avoid hard dependencies
try:
    import numpy as np  # type: ignore
except Exception:
    print("This script requires numpy. Please `pip install numpy`.", file=sys.stderr)
    raise

def _maybe_import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception as e:
        raise RuntimeError("Pandas is required when using --input-csv. Install via `pip install pandas`.") from e

def _maybe_import_pickle():
    try:
        import pickle  # type: ignore
        return pickle
    except Exception as e:
        raise RuntimeError("Pickle is unavailable in this environment.") from e

def _maybe_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception as e:
        raise RuntimeError("PyTorch not available; install torch or use --framework pickle/callable.") from e


# --------------------------
# Helpers
# --------------------------
def set_single_thread_env():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def percentile(values: List[float], q: float) -> float:
    """q in [0,100]; returns percentile using nearest-rank on sorted data."""
    if not values:
        return float("nan")
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    s = sorted(values)
    k = (len(s) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return d0 + d1


def load_numpy_from_csv(path: str,
                        feature_cols: Optional[List[str]] = None,
                        drop_cols: Optional[List[str]] = None) -> np.ndarray:
    pd = _maybe_import_pandas()
    df = pd.read_csv(path)
    if drop_cols:
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found in CSV: {missing}")
        df = df[feature_cols]
    # Coerce to float32 when possible
    x = df.to_numpy()
    x = x.astype(np.float32, copy=False)
    return x


def make_synthetic_data(n: int, d: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d), dtype=np.float32)


# --------------------------
# Model adapters
# --------------------------
class InferenceAdapter:
    """
    Unifies the interface to call model inference with a single example (batch=1).
    Subclasses must implement .predict_one(x: np.ndarray) -> Any
    """
    def predict_one(self, x: np.ndarray):
        raise NotImplementedError()

class PickleAdapter(InferenceAdapter):
    def __init__(self, model_path: str):
        pickle = _maybe_import_pickle()
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict_one(self, x: np.ndarray):
        # Expect x shape (1, D)
        return self.model.predict_proba(x) if hasattr(self.model, "predict_proba") else self.model.predict(x)


class TorchStateDictAdapter(InferenceAdapter):
    def __init__(self, model_path: str, device: str = "cpu", single_thread: bool = True, input_dim: Optional[int] = None):
        torch = _maybe_import_torch()
        if single_thread:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        self.torch = torch
        self.device = torch.device(device)
        # Expect a Python module in sys.path that defines `build_model(input_dim)` or a saved scripted model.
        # Here we assume user saved a state_dict for a known architecture they can rebuild.
        # For generality, try to infer a simple linear model if input_dim provided.
        if input_dim is None:
            raise ValueError("--input-dim is required for pytorch state_dict models unless you customize this adapter.")
        # Minimal reference model (Linear) to allow loading weights with matching shape.
        self.model = torch.nn.Sequential(torch.nn.Linear(input_dim, 1)).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict_one(self, x: np.ndarray):
        with self.torch.no_grad():
            t = self.torch.from_numpy(x).to(self.device)
            out = self.model(t)
            # ensure some CPU sync if CUDA
            if self.device.type == "cuda":
                self.torch.cuda.synchronize()
            return out


class TorchScriptAdapter(InferenceAdapter):
    def __init__(self, model_path: str, device: str = "cpu", single_thread: bool = True):
        torch = _maybe_import_torch()
        if single_thread:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        self.torch = torch
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def predict_one(self, x: np.ndarray):
        with self.torch.no_grad():
            t = self.torch.from_numpy(x).to(self.device)
            out = self.model(t)
            if self.device.type == "cuda":
                self.torch.cuda.synchronize()
            return out


class CallableAdapter(InferenceAdapter):
    def __init__(self, qualname: str):
        """
        qualname format: "module.submodule:function"
        Function must accept a numpy array of shape (1, D) and return predictions.
        """
        if ":" not in qualname:
            raise ValueError("--callable must be 'module:function'")
        mod_name, fn_name = qualname.split(":", 1)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name, None)
        if fn is None or not callable(fn):
            raise ValueError(f"Function '{fn_name}' not found or not callable in module '{mod_name}'.")
        self.fn: Callable[[np.ndarray], object] = fn

    def predict_one(self, x: np.ndarray):
        return self.fn(x)


def build_adapter(framework: str,
                  model_path: Optional[str],
                  device: str,
                  single_thread: bool,
                  callable_qualname: Optional[str],
                  input_dim: Optional[int]) -> InferenceAdapter:
    fw = framework.lower()
    if fw == "pickle":
        if not model_path:
            raise ValueError("--model-path is required for framework=pickle")
        return PickleAdapter(model_path)
    elif fw == "pytorch":
        if not model_path:
            raise ValueError("--model-path is required for framework=pytorch")
        return TorchStateDictAdapter(model_path, device=device, single_thread=single_thread, input_dim=input_dim)
    elif fw == "torchscript":
        if not model_path:
            raise ValueError("--model-path is required for framework=torchscript")
        return TorchScriptAdapter(model_path, device=device, single_thread=single_thread)
    elif fw == "callable":
        if not callable_qualname:
            raise ValueError("--callable is required for framework=callable (format: module:function)")
        return CallableAdapter(callable_qualname)
    else:
        raise ValueError(f"Unsupported framework: {framework}")


# --------------------------
# Benchmark
# --------------------------
def run_benchmark(adapter: InferenceAdapter,
                  X: np.ndarray,
                  runs: int,
                  warmup: int,
                  seed: int = 42) -> Tuple[List[float], float]:
    """
    Returns (latencies_ms, total_wall_ms)
    - latencies_ms: per-inference wall-clock times (ms) for measured runs only
    - total_wall_ms: elapsed wall time for measured runs
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[0] == 0:
        raise ValueError("Input data is empty.")
    n = X.shape[0]
    # Warm-up
    for i in range(warmup):
        x = X[i % n : i % n + 1]  # batch=1
        adapter.predict_one(x)

    times_ms: List[float] = []
    t0_total = time.perf_counter_ns()
    for i in range(runs):
        x = X[i % n : i % n + 1]
        t0 = time.perf_counter_ns()
        adapter.predict_one(x)
        t1 = time.perf_counter_ns()
        times_ms.append((t1 - t0) / 1e6)
    t1_total = time.perf_counter_ns()
    total_ms = (t1_total - t0_total) / 1e6
    return times_ms, total_ms


def summarize(latencies_ms: List[float], total_ms: float) -> dict:
    mean_ms = statistics.fmean(latencies_ms) if latencies_ms else float("nan")
    med_ms = percentile(latencies_ms, 50)
    p95_ms = percentile(latencies_ms, 95)
    p99_ms = percentile(latencies_ms, 99)
    min_ms = min(latencies_ms) if latencies_ms else float("nan")
    max_ms = max(latencies_ms) if latencies_ms else float("nan")
    runs = len(latencies_ms)
    throughput_eps = (runs / (total_ms / 1000.0)) if total_ms > 0 else float("nan")
    qps_from_median = 1000.0 / med_ms if med_ms > 0 else float("nan")
    return {
        "runs": runs,
        "total_ms": total_ms,
        "mean_ms": mean_ms,
        "median_ms": med_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "throughput_events_per_sec": throughput_eps,
        "qps_est_from_median": qps_from_median,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure batch=1 inference latency.")
    parser.add_argument("--framework", required=True,
                        choices=["pickle", "pytorch", "torchscript", "callable"],
                        help="Model framework/loader to use.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model artifact (pickle file, .pt, or TorchScript).")
    parser.add_argument("--callable", dest="callable_qualname", type=str, default=None,
                        help="Custom callable in format 'module.submodule:function'.")
    parser.add_argument("--input-csv", type=str, default=None, help="CSV file for inputs.")
    parser.add_argument("--feature-cols", type=str, default=None,
                        help="Comma-separated list of feature columns to select from CSV.")
    parser.add_argument("--drop-cols", type=str, default=None,
                        help="Comma-separated list of columns to drop from CSV.")
    parser.add_argument("--input-dim", type=int, default=None,
                        help="If no CSV given, generate synthetic inputs with this dimensionality.")
    parser.add_argument("--runs", type=int, default=10000, help="Measured runs (default: 10000).")
    parser.add_argument("--warmup", type=int, default=100, help="Warm-up runs (default: 100).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device for PyTorch/TorchScript models.")
    parser.add_argument("--single-thread", action="store_true", default=True,
                        help="Force single-thread execution via env + torch threads.")
    parser.add_argument("--no-single-thread", dest="single_thread", action="store_false",
                        help="Disable single-thread pinning.")
    parser.add_argument("--export-json", type=str, default=None, help="Path to write JSON summary.")
    parser.add_argument("--export-csv", type=str, default=None, help="Path to write per-run latencies CSV.")
    args = parser.parse_args()

    if args.single_thread:
        set_single_thread_env()

    # Prepare inputs
    if args.input_csv:
        feature_cols = args.feature_cols.split(",") if args.feature_cols else None
        drop_cols = args.drop_cols.split(",") if args.drop_cols else None
        X = load_numpy_from_csv(args.input_csv, feature_cols=feature_cols, drop_cols=drop_cols)
    else:
        if args.input_dim is None:
            raise ValueError("Either --input-csv or --input-dim must be provided.")
        # Create a modest pool (e.g., 1024 rows) and cycle through for runs.
        X = make_synthetic_data(n=1024, d=args.input_dim, seed=42)

    # Build adapter
    adapter = build_adapter(
        framework=args.framework,
        model_path=args.model_path,
        device=args.device,
        single_thread=args.single_thread,
        callable_qualname=args.callable_qualname,
        input_dim=args.input_dim,
    )

    # Benchmark
    latencies_ms, total_ms = run_benchmark(adapter, X, args.runs, args.warmup)

    # Summary
    summary = summarize(latencies_ms, total_ms)

    # Pretty print
    print("\n=== Inference Latency Summary (batch=1) ===")
    for k in ["runs", "mean_ms", "median_ms", "p95_ms", "p99_ms", "min_ms", "max_ms",
              "throughput_events_per_sec", "qps_est_from_median", "total_ms"]:
        print(f"{k:>28s}: {summary[k]:.3f}" if isinstance(summary[k], float) else f"{k:>28s}: {summary[k]}")
    print("===========================================\n")

    # Optional exports
    if args.export_json:
        with open(args.export_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote JSON summary to: {args.export_json}")

    if args.export_csv:
        try:
            pd = _maybe_import_pandas()
            pd.DataFrame({"latency_ms": latencies_ms}).to_csv(args.export_csv, index=False)
            print(f"Wrote per-run latencies to: {args.export_csv}")
        except Exception as e:
            print(f"Failed to export CSV: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
