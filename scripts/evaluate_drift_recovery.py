import csv
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path.home() / "aidac-ai"

OUT_CSV = ROOT / "reproducibility/results/drift/drift_recovery_metrics.csv"
OUT_JSON = ROOT / "reproducibility/results/drift/drift_recovery_metrics.json"
OUT_REPORT = ROOT / "reproducibility/results/drift/drift_recovery_report.txt"

rows = [
    {
        "condition": "F1 at pre-drift week 2",
        "full_aidac": 0.950,
        "no_meta_learning": 0.950,
        "improvement": "---"
    },
    {
        "condition": "F1 immediately after first drift week 3",
        "full_aidac": 0.891,
        "no_meta_learning": 0.841,
        "improvement": "+0.050"
    },
    {
        "condition": "Recovery latency events",
        "full_aidac": 312,
        "no_meta_learning": 1847,
        "improvement": "83.1% fewer events"
    },
    {
        "condition": "F1 at stabilization week 4",
        "full_aidac": 0.944,
        "no_meta_learning": 0.903,
        "improvement": "+0.041"
    },
    {
        "condition": "F1 after second drift week 6",
        "full_aidac": 0.941,
        "no_meta_learning": 0.877,
        "improvement": "+0.064"
    },
]

full_latency = 312
no_meta_latency = 1847
latency_reduction_pct = (1 - full_latency / no_meta_latency) * 100

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["condition", "full_aidac", "no_meta_learning", "improvement"]
    )
    writer.writeheader()
    writer.writerows(rows)

metrics = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "event_file": str(OUT_CSV.relative_to(ROOT)),
    "full_aidac_recovery_latency_events": full_latency,
    "no_meta_learning_recovery_latency_events": no_meta_latency,
    "latency_reduction_pct": round(latency_reduction_pct, 6),
    "rounded_for_thesis": {
        "full_aidac_recovery_latency_events": full_latency,
        "no_meta_learning_recovery_latency_events": no_meta_latency,
        "latency_reduction_pct": round(latency_reduction_pct),
        "latency_reduction_pct_1dp": round(latency_reduction_pct, 1)
    },
    "f1_values": {
        "full_aidac": {
            "pre_drift_week_2": 0.950,
            "after_first_drift_week_3": 0.891,
            "stabilization_week_4": 0.944,
            "after_second_drift_week_6": 0.941
        },
        "no_meta_learning": {
            "pre_drift_week_2": 0.950,
            "after_first_drift_week_3": 0.841,
            "stabilization_week_4": 0.903,
            "after_second_drift_week_6": 0.877
        }
    },
    "note": (
        "This script records the documented drift-recovery evidence used in the "
        "dissertation table. If raw temporal drift-run logs are available, they "
        "should be archived and parsed directly."
    )
}

OUT_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

report = []
report.append("Drift recovery report")
report.append("=" * 70)
report.append("")
report.append("Drift recovery comparison")
report.append("-" * 70)

for r in rows:
    report.append(
        f"{r['condition']}: "
        f"Full AI-DAC={r['full_aidac']}, "
        f"No-meta-learning={r['no_meta_learning']}, "
        f"Improvement={r['improvement']}"
    )

report.append("")
report.append(
    f"Recovery latency reduction: "
    f"{full_latency} vs {no_meta_latency} events "
    f"= {latency_reduction_pct:.2f}% fewer events "
    f"-> rounded {round(latency_reduction_pct)}%"
)

report.append("")
report.append("Output files")
report.append("-" * 70)
report.append(str(OUT_CSV))
report.append(str(OUT_JSON))

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
print("\n".join(report))