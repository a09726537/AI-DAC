import json
import pandas as pd
from pathlib import Path

input_file = Path("/home/ubuntu/aidac-ai/data/suricata_eve.json")
output_file = Path("/home/ubuntu/aidac-ai/data/suricata_events.csv")

rows = []

with input_file.open("r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        alert = event.get("alert", {})
        dns = event.get("dns", {})

        rows.append({
            "event_time": event.get("timestamp"),
            "source_system": "suricata",
            "event_type": event.get("event_type"),
            "source_vm": "aidac-sensor",
            "client_ip": event.get("src_ip"),
            "dest_ip": event.get("dest_ip"),
            "src_port": event.get("src_port"),
            "dest_port": event.get("dest_port"),
            "proto": event.get("proto"),
            "command_type": event.get("event_type"),
            "sql_text": dns.get("query", ""),
            "risk_label": alert.get("category", "network_event"),
            "signature": alert.get("signature", ""),
            "severity": alert.get("severity", 0),
            "is_attack": 1 if alert else 0,
        })

df = pd.DataFrame(rows)
df.to_csv(output_file, index=False)

print(f"Collected {len(df)} Suricata events")