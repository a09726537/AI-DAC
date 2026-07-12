import csv
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path.home() / "aidac-ai"
DATA_ROOT = ROOT / "external_datasets"

OUT_CSV = ROOT / "reproducibility/results/benchmarks/public_benchmark_comparability.csv"
OUT_JSON = ROOT / "reproducibility/results/benchmarks/public_benchmark_comparability_metrics.json"
OUT_REPORT = ROOT / "reproducibility/results/benchmarks/public_benchmark_comparability_report.txt"

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

BENCHMARKS = [
    {
        "dataset": "UNSW-NB15",
        "folder": "UNSW-NB15 2024",
        "expected_files": [
            "UNSW_NB15_training-set.csv",
            "UNSW_NB15_testing-set.csv",
        ],
        "reported_f1": 0.941,
        "reported_roc_auc": 0.965,
        "role": "Public intrusion-detection benchmark for comparability",
        "claim_type": "reported_metric",
    },
    {
        "dataset": "NSL-KDD",
        "folder": "NSL-KDD",
        "expected_files": [
            "KDDTrain+.txt",
            "KDDTest+.txt",
        ],
        "reported_f1": 0.957,
        "reported_roc_auc": 0.973,
        "role": "Public intrusion-detection benchmark for comparability",
        "claim_type": "reported_metric",
    },
    {
        "dataset": "CSE-CIC-IDS2018",
        "folder": "CIC-IDS2018",
        "expected_files": [
            "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
            "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
            "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
            "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
            "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
            "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
            "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
            "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
            "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
            "Thursday-20-02-2018_TrafficForML_CICFlowMeter.csv",
        ],
        "reported_f1": None,
        "reported_roc_auc": None,
        "role": "Processed CICFlowMeter benchmark retained for cross-dataset robustness and comparability",
        "claim_type": "comparability_only",
    },
    {
        "dataset": "TON_IoT",
        "folder": "TON_IoT",
        "expected_files": [
            "train_test_network.csv",
        ],
        "reported_f1": None,
        "reported_roc_auc": None,
        "role": "Additional public IoT/network benchmark evidence",
        "claim_type": "dataset_manifest_only",
    },
    {
        "dataset": "BoT-IoT",
        "folder": "BoT-IoT",
        "expected_files": [
            "UNSW_2018_IoT_Botnet_Final_10_best_Training.csv",
            "UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv",
        ],
        "reported_f1": None,
        "reported_roc_auc": None,
        "role": "Additional public botnet benchmark evidence",
        "claim_type": "dataset_manifest_only",
    },
]

def file_size_mb(path):
    return round(path.stat().st_size / (1024 * 1024), 2)

rows = []

for b in BENCHMARKS:
    folder = DATA_ROOT / b["folder"]
    existing_files = []
    missing_files = []
    total_size_mb = 0.0

    for fname in b["expected_files"]:
        fpath = folder / fname
        if fpath.exists():
            existing_files.append(fname)
            total_size_mb += file_size_mb(fpath)
        else:
            missing_files.append(fname)

    status = "OK" if not missing_files else "MISSING_FILES"

    rows.append({
        "dataset": b["dataset"],
        "folder": str(folder.relative_to(ROOT)),
        "expected_file_count": len(b["expected_files"]),
        "found_file_count": len(existing_files),
        "missing_file_count": len(missing_files),
        "total_size_mb": round(total_size_mb, 2),
        "reported_f1": "" if b["reported_f1"] is None else b["reported_f1"],
        "reported_roc_auc": "" if b["reported_roc_auc"] is None else b["reported_roc_auc"],
        "claim_type": b["claim_type"],
        "role": b["role"],
        "status": status,
        "missing_files": "; ".join(missing_files),
    })

with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "dataset",
            "folder",
            "expected_file_count",
            "found_file_count",
            "missing_file_count",
            "total_size_mb",
            "reported_f1",
            "reported_roc_auc",
            "claim_type",
            "role",
            "status",
            "missing_files",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

metrics = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "data_root": str(DATA_ROOT),
    "output_file": str(OUT_CSV.relative_to(ROOT)),
    "benchmarks": rows,
    "reported_metrics": {
        "UNSW-NB15": {
            "f1": 0.941,
            "roc_auc": 0.965,
        },
        "NSL-KDD": {
            "f1": 0.957,
            "roc_auc": 0.973,
        },
    },
    "comparability_only": {
        "CSE-CIC-IDS2018": (
            "Retained as processed CICFlowMeter benchmark for cross-dataset "
            "robustness and comparability; no unsupported headline metric is claimed."
        )
    },
    "note": (
        "This script validates benchmark availability and records the documented "
        "public-benchmark comparability values used in the dissertation. Public "
        "benchmarks are not substitutes for controlled SQL laboratory evidence."
    ),
}

OUT_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

report = []
report.append("Public benchmark comparability report")
report.append("=" * 70)
report.append(f"Dataset root: {DATA_ROOT}")
report.append("")
report.append("Benchmark availability and reported comparability metrics")
report.append("-" * 70)

for r in rows:
    f1 = "---" if r["reported_f1"] == "" else f'{float(r["reported_f1"]):.3f}'
    auc = "---" if r["reported_roc_auc"] == "" else f'{float(r["reported_roc_auc"]):.3f}'

    report.append(
        f"{r['dataset']}: files={r['found_file_count']}/{r['expected_file_count']}, "
        f"size={r['total_size_mb']} MB, "
        f"F1={f1}, ROC-AUC={auc}, "
        f"claim_type={r['claim_type']}, status={r['status']}"
    )

    if r["missing_files"]:
        report.append(f"  missing: {r['missing_files']}")

report.append("")
report.append("Interpretation")
report.append("-" * 70)
report.append("UNSW-NB15 and NSL-KDD provide reported public benchmark metrics.")
report.append("CSE-CIC-IDS2018 is retained for robustness and comparability only.")
report.append("The database-specific thesis claims rest on the controlled SQL laboratory results.")

report.append("")
report.append("Output files")
report.append("-" * 70)
report.append(str(OUT_CSV))
report.append(str(OUT_JSON))

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
print("\n".join(report))