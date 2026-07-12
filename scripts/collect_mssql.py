import os
import pandas as pd
import pytds

MSSQL_HOST = "192.168.136.132"
MSSQL_DB = "AIDAC_MSSQL"
MSSQL_USER = "aidac_lab"
MSSQL_PASSWORD = "Oracle2020"

OUTPUT_FILE = "/home/ubuntu/aidac-ai/data/mssql_events.csv"

query = """
SELECT event_id, event_time, source_vm, login_name, database_name,
       client_ip, command_type, sql_text, risk_label, is_attack
FROM dbo.sql_audit_events
ORDER BY event_id;
"""

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with pytds.connect(
    server=MSSQL_HOST,
    database=MSSQL_DB,
    user=MSSQL_USER,
    password=MSSQL_PASSWORD,
    port=1433,
) as conn:
    df = pd.read_sql(query, conn)

df["source_system"] = "mssql"
df.to_csv(OUTPUT_FILE, index=False)

print(f"Collected {len(df)} MSSQL events")
print(f"Saved to {OUTPUT_FILE}")