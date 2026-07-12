from pathlib import Path
import pandas as pd

DATA_DIR = Path("/home/ubuntu/aidac-ai/data")

files = {
    "postgresql": DATA_DIR / "pgsql_events.csv",
    "mssql": DATA_DIR / "mssql_events.csv",
    "suricata": DATA_DIR / "suricata_events.csv",
}

standard_columns = [
    "source_system",
    "event_time",
    "source_vm",
    "client_ip",
    "dest_ip",
    "database_name",
    "login_name",
    "event_type",
    "command_type",
    "sql_text",
    "risk_label",
    "signature",
    "severity",
    "is_attack",
    "risk_score",
]

frames = []

for source, path in files.items():
    if not path.exists():
        print(f"Missing file: {path}")
        continue

    df = pd.read_csv(path)
    df["source_system"] = source

    for col in standard_columns:
        if col not in df.columns:
            df[col] = ""

    if source in ["postgresql", "mssql"]:
        df["event_type"] = "sql"
        df["dest_ip"] = "192.168.136.131" if source == "postgresql" else "192.168.136.132"
        df["severity"] = df["is_attack"].apply(lambda x: 8 if int(x) == 1 else 1)

    df["is_attack"] = df["is_attack"].fillna(0).astype(int)

    def score(row):
        label = str(row.get("risk_label", "")).lower()
        severity = int(row.get("severity", 0) or 0)

        if row["is_attack"] == 1:
            return 0.85
        if "suspicious" in label:
            return 0.65
        if severity >= 5:
            return 0.70
        if source == "suricata":
            return 0.20
        return 0.05

    df["risk_score"] = df.apply(score, axis=1)
    frames.append(df[standard_columns])

if not frames:
    raise SystemExit("No input files found")

final_df = pd.concat(frames, ignore_index=True)
final_df.to_csv(DATA_DIR / "aidac_events.csv", index=False)

print(f"Unified dataset created: {DATA_DIR / 'aidac_events.csv'}")
print(f"Rows: {len(final_df)}")
print(final_df.groupby(["source_system", "risk_label"]).size())