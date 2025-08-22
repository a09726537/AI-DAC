#!/usr/bin/env bash
# run_trust_study.sh
# Build and (optionally) analyze a human trust/XAI study.
#
# Author: William K. (University of Vienna)
# Project: AI-DAC — Artificial Intelligence–Driven Anomaly Detection and Control
#
# USAGE (generate study materials):
#   bash scripts/run_trust_study.sh generate \
#     --dataset DS1 \
#     --predictions artifacts/prequential/DS1/runs/W5/seed_1/predictions.csv \
#     --id-col event_id --label-col label --score-col score \
#     --n 60 --analysts "a01,a02,a03,a04" \
#     --policy-config configs/policy.yaml \
#     --shap-csv artifacts/prequential/DS1/runs/W5/seed_1/predictions_shap.csv \
#     --topk 5 \
#     --outdir artifacts/trust_study/DS1_W5_seed1
#
# USAGE (analyze collected responses):
#   bash scripts/run_trust_study.sh analyze \
#     --responses-dir artifacts/trust_study/DS1_W5_seed1/responses \
#     --outdir artifacts/trust_study/DS1_W5_seed1
#
# Notes:
# - Requires Python with pandas and (optionally) PyYAML if you use --policy-config.
# - SHAP CSV is optional. If provided, it should have at least:
#     event_id, feature, phi (one row per feature). The script will aggregate top-k by |phi|.
# - Outputs per-analyst case files + response templates and a manifest.json.
# - “Analyze” computes summary stats: trust (Likert 1–7), time, escalation, overrides.

set -euo pipefail

MODE="${1:-}"
shift || true

# ------- defaults -------
DATASET=""
PREDICTIONS=""
ID_COL="event_id"
LABEL_COL="label"
SCORE_COL="score"
N_CASES=60
ANALYSTS=""
ROSTER=""
OUTDIR="artifacts/trust_study/run"
POLICY_CFG=""
THRESHOLD=""
SHAP_CSV=""
TOPK=5
RANDOM_SEED=42

RESPONSES_DIR=""

PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<EOF
Usage:
  $0 generate [OPTIONS]    # create study materials
  $0 analyze  [OPTIONS]    # analyze collected responses

Generate options:
  --dataset NAME
  --predictions PATH                 CSV with predictions (must include id/label/score)
  --id-col NAME                      default: ${ID_COL}
  --label-col NAME                   default: ${LABEL_COL}
  --score-col NAME                   default: ${SCORE_COL}
  --n INT                            total cases per analyst (default: ${N_CASES})
  --analysts "a1,a2,..."             inline list of analyst IDs
  --roster PATH                      CSV with column 'analyst_id' (alternative to --analysts)
  --outdir PATH                      output directory (default: ${OUTDIR})
  --policy-config PATH               YAML with threshold{,s} (optional)
  --threshold FLOAT                  override decision threshold (else from policy or 0.5)
  --shap-csv PATH                    optional SHAP rows: event_id,feature,phi
  --topk INT                         top-k features for XAI packs (default: ${TOPK})
  --seed INT                         random seed (default: ${RANDOM_SEED})

Analyze options:
  --responses-dir PATH               directory containing analyst CSV responses
  --outdir PATH                      output directory for summary (default: ${OUTDIR})

Examples:
  $0 generate --dataset DS2 --predictions runs/W5/seed_1/preds.csv --analysts "a01,a02" --policy-config configs/policy.yaml
  $0 analyze --responses-dir artifacts/trust_study/DS2_W5_seed1/responses --outdir artifacts/trust_study/DS2_W5_seed1
EOF
  exit 1
}

die(){ echo "[ERR] $*" >&2; exit 2; }

parse_generate() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset) DATASET="$2"; shift 2;;
      --predictions) PREDICTIONS="$2"; shift 2;;
      --id-col) ID_COL="$2"; shift 2;;
      --label-col) LABEL_COL="$2"; shift 2;;
      --score-col) SCORE_COL="$2"; shift 2;;
      --n) N_CASES="$2"; shift 2;;
      --analysts) ANALYSTS="$2"; shift 2;;
      --roster) ROSTER="$2"; shift 2;;
      --outdir) OUTDIR="$2"; shift 2;;
      --policy-config) POLICY_CFG="$2"; shift 2;;
      --threshold) THRESHOLD="$2"; shift 2;;
      --shap-csv) SHAP_CSV="$2"; shift 2;;
      --topk) TOPK="$2"; shift 2;;
      --seed) RANDOM_SEED="$2"; shift 2;;
      -h|--help) usage;;
      *) echo "[WARN] Unknown arg: $1"; shift 1;;
    esac
  done
  [[ -n "$PREDICTIONS" ]] || die "Missing --predictions"
  [[ -f "$PREDICTIONS" ]] || die "Predictions CSV not found: $PREDICTIONS"
  [[ -n "$ANALYSTS" || -n "$ROSTER" ]] || die "Provide --analysts or --roster"
  mkdir -p "$OUTDIR"/{packs,responses,logs}
}

parse_analyze() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --responses-dir) RESPONSES_DIR="$2"; shift 2;;
      --outdir) OUTDIR="$2"; shift 2;;
      -h|--help) usage;;
      *) echo "[WARN] Unknown arg: $1"; shift 1;;
    esac
  done
  [[ -n "$RESPONSES_DIR" ]] || die "Missing --responses-dir"
  [[ -d "$RESPONSES_DIR" ]] || die "Responses dir not found: $RESPONSES_DIR"
  mkdir -p "$OUTDIR"
}

if [[ "$MODE" != "generate" && "$MODE" != "analyze" ]]; then
  usage
fi

if [[ "$MODE" == "generate" ]]; then
  parse_generate "$@"

  # Inline Python does the heavy lifting (sampling, packs, manifest)
  "${PYTHON_BIN}" - <<PYGEN
import os, sys, json, math, random
import pandas as pd
from datetime import datetime

pred_path   = "${PREDICTIONS}"
id_col      = "${ID_COL}"
label_col   = "${LABEL_COL}"
score_col   = "${SCORE_COL}"
n_cases     = int(${N_CASES})
analysts_in = "${ANALYSTS}"
roster_csv  = "${ROSTER}"
outdir      = "${OUTDIR}"
policy_yaml = "${POLICY_CFG}"
threshold_s = "${THRESHOLD}"
shap_csv    = "${SHAP_CSV}"
topk        = int(${TOPK})
seed        = int(${RANDOM_SEED})

random.seed(seed)

# --- load predictions ---
df = pd.read_csv(pred_path)
for col in (id_col, label_col, score_col):
    if col not in df.columns:
        sys.stderr.write(f"[trust] Required column '{col}' missing in predictions.\\n")
        sys.exit(2)

# --- optional SHAP aggregation ---
xai_map = {}
if shap_csv and os.path.exists(shap_csv):
    try:
        dfsh = pd.read_csv(shap_csv)
        req = {"event_id","feature","phi"}
        if not req.issubset(set(dfsh.columns)):
            sys.stderr.write(f"[trust] SHAP CSV must include columns: {sorted(req)}.\\n")
        else:
            g = (dfsh
                 .assign(absphi=lambda x: x["phi"].abs())
                 .sort_values(["event_id","absphi"], ascending=[True, False])
                 .groupby("event_id")
                 .head(topk))
            for eid, grp in g.groupby("event_id"):
                items = [f"{r['feature']}({r['phi']:.3f})" for _, r in grp.iterrows()]
                xai_map[eid] = ", ".join(items)
            print(f"[trust] XAI: aggregated top-{topk} SHAP features for {len(xai_map)} events.")
    except Exception as e:
        sys.stderr.write(f"[trust] Failed SHAP join: {e}\\n")

# --- threshold ---
thr = None
if threshold_s.strip():
    try:
        thr = float(threshold_s)
    except Exception:
        pass

if thr is None and policy_yaml.strip():
    try:
        import yaml
        with open(policy_yaml, "r") as f:
            pol = yaml.safe_load(f)
        if isinstance(pol.get("threshold"), (int,float)):
            thr = float(pol["threshold"])
        elif isinstance(pol.get("thresholds"), dict):
            # Use global default if present, otherwise median
            thr = float(pol.get("threshold", 0.5))
        else:
            thr = 0.5
        print(f"[trust] Threshold from policy: {thr}")
    except Exception as e:
        print(f"[trust] Policy parse warning: {e}; using 0.5")
        thr = 0.5

if thr is None:
    thr = 0.5
    print(f"[trust] Threshold defaulted to {thr}")

# --- difficulty bins ---
#   easy: confident correct (|score - thr| >= 0.25 AND correct)
#   medium: borderline (|score - thr| < 0.1)
#   hard: incorrect predictions (regardless of margin)
df = df.copy()
df["pred"] = (df[score_col] >= thr).astype(int)
df["correct"] = (df["pred"] == df[label_col]).astype(int)
df["margin"] = (df[score_col] - thr).abs()

def bin_row(r):
    if r["pred"] != r[label_col]:
        return "hard"
    if r["margin"] < 0.10:
        return "medium"
    return "easy"  # confident & correct

df["difficulty"] = df.apply(bin_row, axis=1)

# Balance sampling across label x difficulty
# If counts are tight, sample with floor/ceil allocation.
labels = [0,1]
diffs = ["easy","medium","hard"]
plan = {(y,d): 0 for y in labels for d in diffs}

# Target approximately equal splits; adjust down if insufficient pool
target = max(1, n_cases // (len(labels)*len(diffs)))
for y in labels:
    for d in diffs:
        pool = df[(df[label_col]==y) & (df["difficulty"]==d)]
        plan[(y,d)] = min(target, len(pool))

# If under-allocated due to scarcity, fill remaining from global pool not yet selected
alloc_total = sum(plan.values())
remaining = max(0, n_cases - alloc_total)

selected_idx = []
for (y,d), k in plan.items():
    pool = df[(df[label_col]==y) & (df["difficulty"]==d)]
    if k>0:
        idx = pool.sample(n=k, random_state=seed, replace=False).index.tolist()
        selected_idx.extend(idx)

if remaining > 0:
    pool = df.drop(index=selected_idx)
    extra = pool.sample(n=min(remaining, len(pool)), random_state=seed, replace=False).index.tolist()
    selected_idx.extend(extra)

sel = df.loc[selected_idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)

# Prepare packs: No-XAI (score only), XAI (score + top-k SHAP string if available)
def mk_pack(dfpack, include_xai: bool):
    out = pd.DataFrame({
        id_col: dfpack[id_col],
        "score": dfpack[score_col].round(6),
        "label_hidden": dfpack[label_col]  # keep hidden; analysts shouldn't see it
    })
    if include_xai:
        xai_col = []
        for eid in dfpack[id_col].tolist():
            xai_col.append(xai_map.get(eid, ""))  # empty if absent
        out["xai_topk"] = xai_col
    return out

pack_no_xai = mk_pack(sel, include_xai=False)
pack_xai    = mk_pack(sel, include_xai=True)

# Load analyst roster
if roster_csv.strip():
    r = pd.read_csv(roster_csv)
    if "analyst_id" not in r.columns:
        sys.stderr.write("[trust] Roster must include column 'analyst_id'\\n")
        sys.exit(2)
    analysts = r["analyst_id"].astype(str).tolist()
else:
    analysts = [s for s in analysts_in.split(",") if s.strip()]
if not analysts:
    sys.stderr.write("[trust] No analysts provided.\\n"); sys.exit(2)

# Counterbalance order: half AB (No-XAI then XAI), half BA
AB, BA = [], []
for i, a in enumerate(analysts):
    (AB if i % 2 == 0 else BA).append(a)

ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
manifest = {
    "created_utc": ts,
    "dataset": "${DATASET}",
    "predictions_csv": os.path.abspath(pred_path),
    "threshold": thr,
    "n_cases_per_analyst": int(n_cases),
    "topk": int(topk),
    "analysts": [{"id": a, "order": ("AB" if a in AB else "BA")} for a in analysts],
}

packs_dir = os.path.join(outdir, "packs")
resp_dir  = os.path.join(outdir, "responses")
os.makedirs(packs_dir, exist_ok=True)
os.makedirs(resp_dir, exist_ok=True)

# Emit per-analyst files + response templates
for a in analysts:
    base = f"analyst_{a}"
    condA_path = os.path.join(packs_dir, f"{base}_A_NoXAI.csv")
    condB_path = os.path.join(packs_dir, f"{base}_B_XAI.csv")

    pack_no_xai.to_csv(condA_path, index=False)
    pack_xai.to_csv(condB_path, index=False)

    # Response templates
    tmpl_cols = [id_col, "decision(allow|alert)", "trust_1_7", "time_seconds", "override(0|1)", "notes"]
    pd.DataFrame(columns=tmpl_cols).to_csv(os.path.join(resp_dir, f"{base}_A_responses.csv"), index=False)
    pd.DataFrame(columns=tmpl_cols).to_csv(os.path.join(resp_dir, f"{base}_B_responses.csv"), index=False)

# Write manifest and a README
with open(os.path.join(outdir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

readme = f"""# Trust Study Pack

Created: {ts}
Dataset: ${DATASET}

Folders:
- packs/: per-analyst case lists
  - *_A_NoXAI.csv : score-only
  - *_B_XAI.csv   : score + top-k SHAP (if available)
- responses/: empty templates to collect analyst responses

Instructions (AB order):
1) Open *_A_NoXAI.csv, review each case, and fill *_A_responses.csv
2) Then open *_B_XAI.csv, and fill *_B_responses.csv

For BA order, swap A/B sequence.

Response columns:
- decision(allow|alert)
- trust_1_7 (Likert 1–7)
- time_seconds (integer)
- override(0|1)  (1 if analyst disagrees with suggested action)
- notes (free text)

After collection, run analysis:
  bash scripts/run_trust_study.sh analyze \\
    --responses-dir ${outdir}/responses \\
    --outdir ${outdir}
"""
with open(os.path.join(outdir, "README.txt"), "w") as f:
    f.write(readme)

print(f"[trust] Study packs written to: {outdir}")
PYGEN

  echo "[OK] Trust study generation complete → ${OUTDIR}"
  exit 0
fi

if [[ "$MODE" == "analyze" ]]; then
  parse_analyze "$@"

  "${PYTHON_BIN}" - <<PYANA
import os, sys, json, glob
import pandas as pd
import numpy as np
from pathlib import Path

resp_dir = Path("${RESPONSES_DIR}")
outdir   = Path("${OUTDIR}")
outdir.mkdir(parents=True, exist_ok=True)

files = sorted(glob.glob(str(resp_dir / "analyst_*_*_responses.csv")))
if not files:
    print("[trust] No response CSVs found.", file=sys.stderr)
    sys.exit(2)

rows = []
for fp in files:
    base = os.path.basename(fp)
    # analyst_<id>_<A|B>_responses.csv
    try:
        parts = base.split("_")
        analyst = parts[1]
        cond = parts[2]  # A or B
    except Exception:
        analyst, cond = "unknown", "A"
    df = pd.read_csv(fp)
    df["analyst_id"] = analyst
    df["condition"] = cond
    # Normalize columns (tolerate missing)
    for c in ("decision(allow|alert)","trust_1_7","time_seconds","override(0|1)"):
        if c not in df.columns:
            df[c] = np.nan
    rows.append(df)

allr = pd.concat(rows, ignore_index=True)
# Clean up col names for ease
rename = {
    "decision(allow|alert)": "decision",
    "trust_1_7": "trust",
    "time_seconds": "time_s",
    "override(0|1)": "override",
}
allr = allr.rename(columns=rename)
# Basic normalization
allr["decision"] = allr["decision"].astype(str).str.lower().str.strip()
allr["trust"] = pd.to_numeric(allr["trust"], errors="coerce")
allr["time_s"] = pd.to_numeric(allr["time_s"], errors="coerce")
allr["override"] = pd.to_numeric(allr["override"], errors="coerce").fillna(0).astype(int)

# Map condition A/B to NoXAI/XAI by filename convention
allr["condition_name"] = np.where(allr["condition"].eq("A"), "No-XAI", "XAI")

# Metrics per condition
def safe_mean(x): 
    x = pd.to_numeric(x, errors="coerce"); 
    return float(np.nanmean(x)) if np.isfinite(np.nanmean(x)) else float("nan")

def safe_median(x):
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmedian(x)) if len(x)>0 else float("nan")

summary = (allr
           .groupby(["condition_name"], dropna=False)
           .agg(
               n=("decision","size"),
               trust_mean=("trust", safe_mean),
               trust_median=("trust", safe_median),
               time_mean_s=("time_s", safe_mean),
               time_median_s=("time_s", safe_median),
               escalation_rate=("decision", lambda s: float(np.mean(s.astype(str).str.contains("alert", na=False)))*100.0),
               override_rate=("override", lambda s: float(np.mean(pd.to_numeric(s, errors="coerce").fillna(0)))*100.0),
           ).reset_index()
)

# Also per-analyst
by_analyst = (allr
              .groupby(["analyst_id","condition_name"], dropna=False)
              .agg(
                  n=("decision","size"),
                  trust_mean=("trust", safe_mean),
                  time_median_s=("time_s", safe_median),
                  escalation_rate=("decision", lambda s: float(np.mean(s.astype(str).str.contains("alert", na=False)))*100.0),
                  override_rate=("override", lambda s: float(np.mean(pd.to_numeric(s, errors="coerce").fillna(0)))*100.0),
              ).reset_index())

summary.to_csv(outdir / "trust_summary_by_condition.csv", index=False)
by_analyst.to_csv(outdir / "trust_summary_by_analyst.csv", index=False)

# JSON bundle
with open(outdir / "trust_summary.json", "w") as f:
    json.dump({
        "by_condition": summary.to_dict(orient="records"),
        "by_analyst": by_analyst.to_dict(orient="records"),
        "n_total_rows": int(len(allr)),
    }, f, indent=2)

print(f"[trust] Wrote summaries to {outdir}")
PYANA

  echo "[OK] Trust study analysis complete → ${OUTDIR}"
  exit 0
fi

