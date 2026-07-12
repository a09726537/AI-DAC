import os
import pandas as pd
import psycopg

PG_HOST = "192.168.136.131"
PG_DB = "aidac_pgsql"
PG_USER = "aidac_lab"
PG_PASSWORD = "Oracle2020"

OUTPUT_FILE = "/home/ubuntu/aidac-ai/data/pgsql_events.csv"

query = """
SELECT event_id, event_time, source_vm, login_name, database_name,
       client_ip, command_type, sql_text, risk_label, is_attack
FROM public.sql_audit_events
ORDER BY event_id;
"""

conninfo = (
    f"host={PG_HOST} "
    f"dbname={PG_DB} "
    f"user={PG_USER} "
    f"password={PG_PASSWORD}"
)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with psycopg.connect(conninfo) as conn:
    df = pd.read_sql(query, conn)

df["source_system"] = "postgresql"
df.to_csv(OUTPUT_FILE, index=False)

print(f"Collected {len(df)} PostgreSQL events")
print(f"Saved to {OUTPUT_FILE}")