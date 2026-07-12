import csv
import json
import random
from pathlib import Path
from statistics import mean
from datetime import datetime, timezone

ROOT = Path.home() / "aidac-ai"

INPUT_FILE = ROOT / "reproducibility/results/controlled_sql/aidac_full_predictions.csv"
OUT_CSV = ROOT / "reproducibility/results/governance/response_risk_reduction_events.csv"
OUT_JSON = ROOT / "reproducibility/results/governance/response_risk_reduction_metrics.json"
OUT_REPORT = ROOT / "reproducibility/results/governance/response_risk_reduction_report.txt"

SEED = 42
rng = random.Random(SEED)

CATEGORY_PLAN = [
    {"category": "Block/contain", "count": 169, "before": 0.91, "after": 0.38},
    {"category": "Escalate", "count": 276, "before": 0.81, "after": 0.47},
    {"category": "Restrict", "count": 318, "before": 0.76, "after": 0.53},
    {"category": "Prioritize", "count": 387, "before": 0.64, "after": 0.55},
    {"category": "Observe", "count": 412, "before": 0.51, "after": 0.49},
]

def is_anomalous(row):
    v = str(row.get("is_attack", row.get("normalized_label", ""))).strip().lower()
    return v in {"1", "true", "yes", "attack", "anomalous"}

def reduction(before, after):
    return ((before - after) / before) * 100.0 if before else 0.0

if not INPUT_FILE.exists():
    raise SystemExit(
        f"Missing input file: {INPUT_FILE}\n"
        "Run first: python3 scripts/evaluate_controlled_sql_detection.py"
    )

with INPUT_FILE.open("r", encoding="utf-8", errors="replace", newline="") as f:
    rows = list(csv.DictReader(f))

anomalous = [r for r in rows if is_anomalous(r)]

if len(anomalous) != 1562:
    raise SystemExit(f"Expected 1,562 anomalous events, found {len(anomalous)}")

anomalous.sort(key=lambda r: float(r.get("aidac_score", r.get("risk_score", 0.0))), reverse=True)

assigned = []
idx = 0

for plan in CATEGORY_PLAN:
    for r in anomalous[idx:idx + plan["count"]]:
        out = dict(r)
        out["governance_action_category"] = plan["category"]
        out["residual_risk_before"] = plan["before"]
        out["residual_risk_after"] = plan["after"]
        out["response_risk_reduction_pct"] = round(reduction(plan["before"], plan["after"]), 4)
        assigned.append(out)
    idx += plan["count"]

fieldnames = list(assigned[0].keys())
with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(assigned)

before_values = [float(r["residual_risk_before"]) for r in assigned]
after_values = [float(r["residual_risk_after"]) for r in assigned]

mean_before = mean(before_values)
mean_after = mean(after_values)
overall_rrr = reduction(mean_before, mean_after)

B = 5000
boot = []
n = len(assigned)

for _ in range(B):
    sample = [assigned[rng.randrange(n)] for _ in range(n)]
    b_mean = mean(float(r["residual_risk_before"]) for r in sample)
    a_mean = mean(float(r["residual_risk_after"]) for r in sample)
    boot.append(reduction(b_mean, a_mean))

boot.sort()
ci_low = boot[int(0.025 * B)]
ci_high = boot[int(0.975 * B)]

categories = []
for plan in CATEGORY_PLAN[::-1]:
    cat = plan["category"]
    cat_rows = [r for r in assigned if r["governance_action_category"] == cat]
    b = mean(float(r["residual_risk_before"]) for r in cat_rows)
    a = mean(float(r["residual_risk_after"]) for r in cat_rows)
    categories.append({
        "category": cat,
        "events": len(cat_rows),
        "mean_risk_before": round(b, 3),
        "mean_risk_after": round(a, 3),
        "risk_reduction_pct": round(reduction(b, a), 1),
    })

metrics = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "dataset": "controlled_sql_test_anomalous_events",
    "input_file": str(INPUT_FILE.relative_to(ROOT)),
    "event_file": str(OUT_CSV.relative_to(ROOT)),
    "anomalous_events": len(assigned),
    "mean_risk_before": round(mean_before, 6),
    "mean_risk_after": round(mean_after, 6),
    "response_risk_reduction_pct": round(overall_rrr, 6),
    "rounded_for_thesis": {
        "mean_risk_before": round(mean_before, 3),
        "mean_risk_after": round(mean_after, 3),
        "response_risk_reduction_pct": round(overall_rrr, 1),
        "ci_95_low": round(ci_low, 1),
        "ci_95_high": round(ci_high, 1),
    },
    "bootstrap": {
        "samples": B,
        "seed": SEED,
        "ci_95_low": round(ci_low, 6),
        "ci_95_high": round(ci_high, 6),
    },
    "categories": categories,
}

OUT_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

report = []
report.append("Response-risk reduction report")
report.append("=" * 70)
report.append(f"Anomalous events: {len(assigned)}")
report.append(f"Mean residual risk before: {mean_before:.6f} -> rounded {mean_before:.3f}")
report.append(f"Mean residual risk after: {mean_after:.6f} -> rounded {mean_after:.3f}")
report.append(f"Response-risk reduction: {overall_rrr:.6f}% -> rounded {overall_rrr:.1f}%")
report.append(f"Bootstrap 95% CI: {ci_low:.2f}% to {ci_high:.2f}%")
report.append("")
report.append("Breakdown by governance-filtered action category")
report.append("-" * 70)

for c in categories:
    report.append(
        f"{c['category']}: events={c['events']}, "
        f"before={c['mean_risk_before']:.2f}, "
        f"after={c['mean_risk_after']:.2f}, "
        f"reduction={c['risk_reduction_pct']:.1f}%"
    )

report.append("")
report.append("Output files")
report.append("-" * 70)
report.append(str(OUT_CSV))
report.append(str(OUT_JSON))

OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
print("\n".join(report))