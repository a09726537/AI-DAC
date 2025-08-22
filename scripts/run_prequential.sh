#!/usr/bin/env bash
# run_prequential.sh
# Orchestrates prequential (sliding-window) experiments:
#   For k in [1..N-2]: train on W1..Wk, validate on W(k+1), test on W(k+2)
#
# Author: William K. (University of Vienna)
# Project: AI-DAC — Artificial Intelligence–Driven Anomaly Detection and Control
#
# Example:
#   bash scripts/run_prequential.sh \
#     --dataset DS2 \
#     --source-csv data/DS2_processed/all.csv \
#     --time-col ts --label-col label --id-col event_id \
#     --windows "W1:W12" \
#     --seeds "1,7,13,21,42" \
#     --detector-config configs/detector.yaml \
#     --policy-config configs/policy.yaml \
#     --outdir artifacts/prequential/DS2
#
# Notes:
# - Requires: python (with pandas, pyyaml if your train/eval scripts need it)
# - Expects train.py / eval.py at scripts/train.py and scripts/eval.py
# - Creates windowed CSVs under: $OUTDIR/windows/Wk/{train,val,test}.csv
# - Writes per-run logs/metrics under: $OUTDIR/runs/Wk/seed_<S>/

set -euo pipefail

# ------------- defaults -------------
DATASET=""
SOURCE_CSV=""
TIME_COL="ts"
LABEL_COL="label"
ID_COL="event_id"
WINDOWS_SPEC="W1:W12"
SEEDS="1,7,13,21,42"
DET_CFG=""
POL_CFG=""
OUTDIR="artifacts/prequential"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"  # for eval/train consistency if they accept it
EXTRA_TRAIN_ARGS=""
EXTRA_EVAL_ARGS=""
FEATURE_COLS=""          # optional comma-separated list to pass into eval.py

# ------------- helpers -------------
usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Required:
  --dataset NAME                 Logical dataset name (e.g., DS1, DS2, DS3)
  --source-csv PATH              Single CSV to be windowized (time-sorted by ${TIME_COL})
  --detector-config PATH         Detector YAML config (e.g., configs/detector.yaml)
  --policy-config PATH           Policy YAML config (e.g., configs/policy.yaml)

Optional:
  --time-col NAME                Time column (default: ${TIME_COL})
  --label-col NAME               Label column (default: ${LABEL_COL})
  --id-col NAME                  Event ID column (default: ${ID_COL})
  --windows "W1:W12"             Window range (default: ${WINDOWS_SPEC})
  --seeds "1,7,13,21,42"         Comma-separated seeds (default: ${SEEDS})
  --outdir PATH                  Base output dir (default: ${OUTDIR})
  --python-bin PATH              Python interpreter (default: ${PYTHON_BIN})
  --feature-cols "f1,f2,..."     Optional explicit feature columns for eval.py
  --device cpu|cuda              Device hint forwarded to eval/train (default: ${DEVICE})
  --extra-train-args "...args"   Extra args forwarded verbatim to train.py
  --extra-eval-args  "...args"   Extra args forwarded verbatim to eval.py

Examples:
  $0 --dataset DS2 --source-csv data/DS2_processed/all.csv \\
     --detector-config configs/detector.yaml --policy-config configs/policy.yaml

EOF
  exit 1
}

die() { echo "[ERR] $*" >&2; exit 1; }

req() {
  local vname="$1"; local val="${!vname:-}"
  [[ -n "$val" ]] || die "Missing required option: --${vname//_/-}"
}

# ------------- parse args -------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --source-csv) SOURCE_CSV="$2"; shift 2;;
    --time-col) TIME_COL="$2"; shift 2;;
    --label-col) LABEL_COL="$2"; shift 2;;
    --id-col) ID_COL="$2"; shift 2;;
    --windows) WINDOWS_SPEC="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --detector-config) DET_CFG="$2"; shift 2;;
    --policy-config) POL_CFG="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    --python-bin) PYTHON_BIN="$2"; shift 2;;
    --feature-cols) FEATURE_COLS="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --extra-train-args) EXTRA_TRAIN_ARGS="$2"; shift 2;;
    --extra-eval-args) EXTRA_EVAL_ARGS="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "[WARN] Unknown arg: $1"; shift 1;;
  esac
done

req DATASET
req SOURCE_CSV
req DET_CFG
req POL_CFG

[[ -f "$SOURCE_CSV" ]] || die "source CSV not found: $SOURCE_CSV"
[[ -f "$DET_CFG" ]] || die "detector config not found: $DET_CFG"
[[ -f "$POL_CFG" ]] || die "policy config not found: $POL_CFG"
command -v "$PYTHON_BIN" >/dev/null || die "python not found at: ${PYTHON_BIN}"

# Ensure scripts exist
[[ -f "scripts/train.py" ]] || echo "[WARN] scripts/train.py not found (continuing; ensure your path is correct)."
[[ -f "scripts/eval.py"  ]] || echo "[WARN] scripts/eval.py  not found (continuing; ensure your path is correct)."

# ------------- derive paths -------------
OUTDIR="${OUTDIR%/}/${DATASET}"
WIN_DIR="${OUTDIR}/windows"     # holds sliced CSVs per window
RUNS_DIR="${OUTDIR}/runs"       # holds per-window, per-seed outputs
LOGS_DIR="${OUTDIR}/logs"
mkdir -p "$WIN_DIR" "$RUNS_DIR" "$LOGS_DIR"

echo "[INFO] Dataset:        $DATASET"
echo "[INFO] Source CSV:     $SOURCE_CSV"
echo "[INFO] Windows:        $WINDOWS_SPEC"
echo "[INFO] Seeds:          $SEEDS"
echo "[INFO] Outdir:         $OUTDIR"
echo "[INFO] Time/Label/ID:  $TIME_COL / $LABEL_COL / $ID_COL"
echo

# ------------- windowize (Python inline) -------------
# WINDOWS_SPEC like "W1:W12" or "1:12" -> N_WINDOWS = 12
echo "[INFO] Building prequential windows under ${WIN_DIR} ..."
"$PYTHON_BIN" - <<PYWIN
import os, sys, re, json
import pandas as pd
from datetime import datetime

source = "${SOURCE_CSV}"
outdir = "${WIN_DIR}"
time_col = "${TIME_COL}"
label_col = "${LABEL_COL}"
id_col = "${ID_COL}"
spec = "${WINDOWS_SPEC}"

if not os.path.exists(source):
    sys.stderr.write(f"[windowize] missing source: {source}\\n"); sys.exit(2)

# Parse "W1:W12" or "1:12"
m = re.match(r"^W?(\\d+):W?(\\d+)$", spec.strip(), flags=re.IGNORECASE)
if not m:
    sys.stderr.write(f"[windowize] invalid --windows spec: {spec}\\n"); sys.exit(2)
start, end = int(m.group(1)), int(m.group(2))
if end <= start:
    sys.stderr.write("[windowize] end must be > start\\n"); sys.exit(2)
N = end  # use 1..N as W1..WN
os.makedirs(outdir, exist_ok=True)

df = pd.read_csv(source)
if time_col not in df.columns:
    sys.stderr.write(f"[windowize] time_col '{time_col}' not in CSV.\\n"); sys.exit(2)

# Parse time & sort; robustly handle epoch or ISO8601
def to_dt(s):
    try:
        return pd.to_datetime(s, utc=True)
    except Exception:
        try:
            return pd.to_datetime(s, unit="s", utc=True)
        except Exception:
            return pd.NaT

df[time_col] = df[time_col].apply(to_dt)
df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

# Equal-size windows by row count (deterministic)
parts = []
n = len(df)
if N < 3:
    sys.stderr.write("[windowize] require at least 3 windows for prequential (train/val/test).\\n"); sys.exit(2)
split_idxs = [round(i * n / N) for i in range(N+1)]
for k in range(N):
    a, b = split_idxs[k], split_idxs[k+1]
    win = df.iloc[a:b].copy()
    win["_window"] = f"W{k+1}"
    parts.append(win)

# Persist windows and an index
index_rows = []
for k in range(1, N-1):  # only k=1..N-2 used
    w_train = pd.concat(parts[:k], ignore_index=True) if k > 0 else pd.DataFrame(columns=df.columns)
    w_val   = parts[k].copy()
    w_test  = parts[k+1].copy()

    wdir = os.path.join(outdir, f"W{k}")
    os.makedirs(wdir, exist_ok=True)
    w_train.to_csv(os.path.join(wdir, "train.csv"), index=False)
    w_val.to_csv(os.path.join(wdir, "val.csv"), index=False)
    w_test.to_csv(os.path.join(wdir, "test.csv"), index=False)

    index_rows.append({
        "window": f"W{k}",
        "train_rows": int(len(w_train)),
        "val_rows": int(len(w_val)),
        "test_rows": int(len(w_test)),
        "train_span": f"{w_train[time_col].min()} -> {w_train[time_col].max()}" if len(w_train) else "",
        "val_span":   f"{w_val[time_col].min()} -> {w_val[time_col].max()}",
        "test_span":  f"{w_test[time_col].min()} -> {w_test[time_col].max()}",
    })

pd.DataFrame(index_rows).to_csv(os.path.join(outdir, "index.csv"), index=False)
print(f"[windowize] wrote {len(index_rows)} window triplets to: {outdir}")
PYWIN

echo

# ------------- iterate windows & seeds -------------
# Extract numeric start/end from WINDOWS_SPEC
WIN_START="${WINDOWS_SPEC#W}"
WIN_START="${WIN_START%%:*}"
WIN_END="${WINDOWS_SPEC##*:}"
WIN_END="${WIN_END#W}"

# windows used for train/val/test triplets: W1..W(N-2)
LAST_TRIPLET=$((WIN_END-2))

IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
FEATURE_FLAG=""
if [[ -n "${FEATURE_COLS}" ]]; then
  FEATURE_FLAG="--feature-cols ${FEATURE_COLS}"
fi

for (( k=WIN_START; k<=LAST_TRIPLET; k++ )); do
  WDIR="${WIN_DIR}/W${k}"
  TRAIN_CSV="${WDIR}/train.csv"
  VAL_CSV="${WDIR}/val.csv"
  TEST_CSV="${WDIR}/test.csv"
  [[ -f "$TRAIN_CSV" && -f "$VAL_CSV" && -f "$TEST_CSV" ]] || die "Missing window CSVs in $WDIR"

  for SEED in "${SEED_ARR[@]}"; do
    RUN_DIR="${RUNS_DIR}/W${k}/seed_${SEED}"
    mkdir -p "$RUN_DIR"
    echo "[RUN] W${k} seed ${SEED}"

    # ---------- TRAIN ----------
    # You can adapt flags to your train.py signature.
    # Suggested train.py arguments (align with your implementation):
    #   --train, --val, --detector-config, --policy-config, --seed, --outdir, --device
    {
      echo "=== TRAIN W${k} seed ${SEED} $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
      echo "Train: ${TRAIN_CSV}"
      echo "Val:   ${VAL_CSV}"
    } > "${RUN_DIR}/train.log"

    set +e
    ${PYTHON_BIN} scripts/train.py \
      --train "${TRAIN_CSV}" \
      --val "${VAL_CSV}" \
      --detector-config "${DET_CFG}" \
      --policy-config "${POL_CFG}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --outdir "${RUN_DIR}" \
      ${EXTRA_TRAIN_ARGS} >> "${RUN_DIR}/train.log" 2>&1
    TRN_RC=$?
    set -e
    [[ $TRN_RC -eq 0 ]] || die "Training failed for W${k} seed ${SEED} (see ${RUN_DIR}/train.log)"

    # ---------- EVAL ----------
    # Suggested eval.py arguments (align with your implementation):
    #   --test, --model-dir, --policy-config, --id-col, --label-col, --out-json, --out-csv, --device
    {
      echo "=== EVAL W${k} seed ${SEED} $(date -u +"%Y-%m-%dT%H:%M:%SZ") ==="
      echo "Test:  ${TEST_CSV}"
    } > "${RUN_DIR}/eval.log"

    OUT_JSON="${RUN_DIR}/metrics.json"
    OUT_CSV="${RUN_DIR}/predictions.csv"

    set +e
    ${PYTHON_BIN} scripts/eval.py \
      --test "${TEST_CSV}" \
      --model-dir "${RUN_DIR}" \
      --policy-config "${POL_CFG}" \
      --id-col "${ID_COL}" \
      --label-col "${LABEL_COL}" \
      --device "${DEVICE}" \
      ${FEATURE_FLAG} \
      --out-json "${OUT_JSON}" \
      --out-csv "${OUT_CSV}" \
      ${EXTRA_EVAL_ARGS} >> "${RUN_DIR}/eval.log" 2>&1
    EV_RC=$?
    set -e
    [[ $EV_RC -eq 0 ]] || die "Evaluation failed for W${k} seed ${SEED} (see ${RUN_DIR}/eval.log)"

    echo "[OK] W${k} seed ${SEED} complete → ${RUN_DIR}"
  done
done

# ------------- aggregate metrics -------------
echo "[INFO] Aggregating metrics → ${OUTDIR}/aggregate_metrics.csv"
${PYTHON_BIN} - <<'PYAGG'
import os, json, glob, pandas as pd, sys
outdir = os.environ.get("OUTDIR")
runs = glob.glob(os.path.join(outdir, "runs", "W*", "seed_*", "metrics.json"))
rows = []
for path in runs:
    parts = path.split(os.sep)
    try:
        w = [p for p in parts if p.startswith("W")][0]
        s = [p for p in parts if p.startswith("seed_")][0].split("_",1)[1]
    except Exception:
        w, s = "", ""
    try:
        with open(path, "r") as f:
            js = json.load(f)
        flat = {"window": w, "seed": int(s)}
        # Expect common keys; tolerate missing
        for k in ("auc_roc","auc_pr","f1","precision","recall","ece","mcc","accuracy","loss","latency_p95_ms"):
            if k in js:
                flat[k] = js[k]
        rows.append(flat)
    except Exception as e:
        print(f"[agg] skip {path}: {e}", file=sys.stderr)
if rows:
    df = pd.DataFrame(rows).sort_values(["window","seed"])
    df.to_csv(os.path.join(outdir, "aggregate_metrics.csv"), index=False)
    print(f"[agg] wrote {os.path.join(outdir, 'aggregate_metrics.csv')}")
else:
    print("[agg] no metrics found", file=sys.stderr)
PYAGG
echo "[DONE] Prequential runs finished."

