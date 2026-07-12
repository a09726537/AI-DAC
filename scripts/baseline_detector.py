from pathlib import Path
import pandas as pd

DATA_DIR = Path("/home/ubuntu/aidac-ai/data")
REPORT_DIR = Path("/home/ubuntu/aidac-ai/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

input_file = DATA_DIR / "aidac_events.csv"
output_file = DATA_DIR / "aidac_scored_events.csv"
report_file = REPORT_DIR / "aidac_baseline_summary.txt"

if not input_file.exists():
    raise SystemExit(f"Missing dataset: {input_file}")

df = pd.read_csv(input_file)

def compute_baseline_score(row):
    score = 0.0

    sql_text = str(row.get("sql_text", "")).lower()
    command = str(row.get("command_type", "")).lower()
    label = str(row.get("risk_label", "")).lower()
    source = str(row.get("source_system", "")).lower()
    severity = float(row.get("severity", 0) or 0)

    risky_patterns = [
        " or 1=1",
        "union select",
        "drop table",
        "xp_cmdshell",
        "information_schema",
        "sys.objects",
        "--",
        "/*",
        "sleep(",
        "benchmark(",
    ]

    if any(pattern in sql_text for pattern in risky_patterns):
        score += 0.45

    if "injection" in command or "suspicious" in label:
        score += 0.35

    if source == "suricata" and severity >= 2:
        score += 0.25

    if int(row.get("is_attack", 0) or 0) == 1:
        score += 0.30

    return min(score, 1.0)

df["baseline_score"] = df.apply(compute_baseline_score, axis=1)
df["baseline_decision"] = df["baseline_score"].apply(
    lambda x: "attack" if x >= 0.60 else "normal"
)

df.to_csv(output_file, index=False)

total = len(df)
attacks = int((df["baseline_decision"] == "attack").sum())
normal = total - attacks

summary = [
    "AI-DAC Baseline Detector Summary",
    "=================================",
    f"Input file: {input_file}",
    f"Output file: {output_file}",
    f"Total events: {total}",
    f"Detected attacks: {attacks}",
    f"Detected normal: {normal}",
    "",
    "Events by source:",
    str(df.groupby("source_system").size()),
    "",
    "Decisions by source:",
    str(df.groupby(["source_system", "baseline_decision"]).size()),
]

report_file.write_text("\n".join(summary), encoding="utf-8")

print("\n".join(summary))