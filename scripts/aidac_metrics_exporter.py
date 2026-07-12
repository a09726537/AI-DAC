from pathlib import Path
import time
import pandas as pd
from prometheus_client import Gauge, start_http_server

DATA_FILE = Path("/home/ubuntu/aidac-ai/data/aidac_scored_events.csv")

total_events = Gauge("aidac_total_events", "Total AI-DAC events")
attack_events = Gauge("aidac_attack_events", "Detected attack events")
normal_events = Gauge("aidac_normal_events", "Detected normal events")
average_risk = Gauge("aidac_average_risk_score", "Average AI-DAC risk score")
postgresql_events = Gauge("aidac_postgresql_events", "PostgreSQL events")
mssql_events = Gauge("aidac_mssql_events", "MSSQL events")
suricata_events = Gauge("aidac_suricata_events", "Suricata events")

def update_metrics():
    if not DATA_FILE.exists():
        return

    df = pd.read_csv(DATA_FILE)

    total = len(df)
    attacks = int((df["baseline_decision"] == "attack").sum())
    normal = total - attacks

    total_events.set(total)
    attack_events.set(attacks)
    normal_events.set(normal)
    average_risk.set(float(df["baseline_score"].mean()) if total else 0)

    postgresql_events.set(int((df["source_system"] == "postgresql").sum()))
    mssql_events.set(int((df["source_system"] == "mssql").sum()))
    suricata_events.set(int((df["source_system"] == "suricata").sum()))

if __name__ == "__main__":
    start_http_server(8000)
    print("AI-DAC metrics exporter running on port 8000")

    while True:
        update_metrics()
        time.sleep(10)