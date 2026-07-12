import pandas as pd

PGSQL_FILE = "/home/ubuntu/aidac-ai/data/pgsql_events.csv"
MSSQL_FILE = "/home/ubuntu/aidac-ai/data/mssql_events.csv"
OUTPUT_FILE = "/home/ubuntu/aidac-ai/data/combined_sql_events.csv"

pgsql_df = pd.read_csv(PGSQL_FILE)
mssql_df = pd.read_csv(MSSQL_FILE)

combined_df = pd.concat([pgsql_df, mssql_df], ignore_index=True)
combined_df = combined_df.sort_values(by=["event_time", "source_system", "event_id"])

combined_df.to_csv(OUTPUT_FILE, index=False)

print(f"PostgreSQL events: {len(pgsql_df)}")
print(f"MSSQL events: {len(mssql_df)}")
print(f"Combined SQL events: {len(combined_df)}")
print(f"Saved to {OUTPUT_FILE}")
