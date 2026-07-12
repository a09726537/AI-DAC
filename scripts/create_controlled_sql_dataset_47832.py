import csv
import json
import hashlib
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter

ROOT = Path.home() / "aidac-ai"
RAW_FILE = ROOT / "data/raw/controlled_sql_events.csv"
PROCESSED_FILE = ROOT / "data/processed/controlled_sql_processed.csv"
TRAIN_FILE = ROOT / "data/processed/controlled_sql_train.csv"
VAL_FILE = ROOT / "data/processed/controlled_sql_validation.csv"
TEST_FILE = ROOT / "data/processed/controlled_sql_test.csv"
REPORT_FILE = ROOT / "reproducibility/results/controlled_sql/controlled_sql_dataset_report.txt"
MANIFEST_FILE = ROOT / "reproducibility/manifests/controlled_sql_lab_manifest.json"

SEED = 42
rng = random.Random(SEED)

# Exact thesis counts
COUNTS = {
    "train": {"normal": 26263, "anomalous": 7220},
    "validation": {"normal": 5625, "anomalous": 1550},
    "test": {"normal": 5612, "anomalous": 1562},
}

DATABASES = ["core_banking", "crm", "hr", "inventory", "billing", "auditdb"]
NORMAL_USERS = ["app_user", "report_user", "etl_user", "readonly_user", "dba_admin", "backup_user"]
SERVICE_USERS = ["svc_backup", "svc_etl", "svc_report", "svc_batch"]
ATTACK_USERS = ["unknown_user", "guest", "webapp", "compromised_app", "svc_backup"]
CLIENTS = [f"192.168.136.{i}" for i in range(20, 240)]

NORMAL_TEMPLATES = [
    ("SELECT", "SELECT customer_id, status FROM customers WHERE customer_id = {id};", "read"),
    ("SELECT", "SELECT order_id, amount FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL '7 days';", "read"),
    ("UPDATE", "UPDATE accounts SET last_seen = CURRENT_TIMESTAMP WHERE account_id = {id};", "write"),
    ("INSERT", "INSERT INTO audit_log(user_id, action, created_at) VALUES ({id}, 'LOGIN_OK', CURRENT_TIMESTAMP);", "write"),
    ("SELECT", "SELECT count(*) FROM transactions WHERE status = 'SETTLED';", "reporting"),
    ("BACKUP", "BACKUP DATABASE billing TO DISK = '/backup/billing_full.bak';", "maintenance"),
    ("ALTER", "ALTER INDEX idx_orders_customer REBUILD;", "maintenance"),
]

ANOMALY_TEMPLATES = {
    "sql_injection_evasion": [
        ("SELECT", "SELECT * FROM users WHERE name = 'admin' OR '1'='1' --';"),
        ("SELECT", "SELECT password_hash FROM users WHERE user_id = 1 UNION SELECT token FROM api_keys;"),
        ("SELECT", "SELECT * FROM accounts WHERE id = {id}; DROP TABLE audit_log; --"),
    ],
    "authentication_burst": [
        ("LOGIN", "FAILED LOGIN FOR user='admin' FROM client_ip;"),
        ("LOGIN", "FAILED LOGIN FOR user='sa' FROM client_ip;"),
    ],
    "privilege_escalation": [
        ("GRANT", "GRANT DBA TO app_user;"),
        ("ALTER", "ALTER ROLE readonly_user SUPERUSER;"),
        ("GRANT", "GRANT SELECT, UPDATE, DELETE ON customers TO unknown_user;"),
    ],
    "high_volume_extraction": [
        ("SELECT", "SELECT * FROM customers;"),
        ("SELECT", "SELECT * FROM transactions WHERE amount > 0;"),
        ("COPY", "COPY customers TO '/tmp/export_customers.csv' CSV HEADER;"),
    ],
    "schema_tampering": [
        ("ALTER", "ALTER TABLE payments ADD COLUMN hidden_token TEXT;"),
        ("DROP", "DROP TABLE audit_log;"),
        ("ALTER", "ALTER TABLE users DISABLE TRIGGER ALL;"),
    ],
    "backup_abuse": [
        ("BACKUP", "BACKUP DATABASE core_banking TO DISK = '/tmp/core_banking_copy.bak';"),
        ("COPY", "COPY audit_log TO '/tmp/audit_dump.csv' CSV HEADER;"),
    ],
    "service_account_misuse": [
        ("SELECT", "SELECT * FROM payroll WHERE employee_id > 0;"),
        ("UPDATE", "UPDATE users SET role = 'admin' WHERE username = 'svc_report';"),
    ],
    "slow_rate_privilege_creep": [
        ("GRANT", "GRANT SELECT ON sensitive_accounts TO svc_report;"),
        ("GRANT", "GRANT UPDATE ON customer_flags TO svc_report;"),
        ("ALTER", "ALTER DEFAULT PRIVILEGES GRANT SELECT ON TABLES TO svc_report;"),
    ],
}

ANOMALY_SCENARIOS = list(ANOMALY_TEMPLATES.keys())
LIFECYCLE_PHASES = ["normal_operation", "backup_window", "deployment_window", "maintenance_window", "incident_response"]

FIELDNAMES = [
    "event_id",
    "event_time",
    "partition",
    "source_system",
    "source_vm",
    "client_ip",
    "database_name",
    "login_name",
    "lifecycle_phase",
    "event_type",
    "command_type",
    "object_name",
    "sql_text",
    "query_length",
    "rows_returned",
    "failed_login_count",
    "privilege_changed",
    "maintenance_window",
    "risk_label",
    "severity",
    "is_attack",
    "attack_scenario",
    "risk_score",
    "baseline_score",
    "baseline_decision",
    "normalized_label",
    "dataset_version",
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def normal_row(event_id, event_time, partition):
    command, sql_template, event_type = rng.choice(NORMAL_TEMPLATES)
    db = rng.choice(DATABASES)
    user = rng.choice(NORMAL_USERS + SERVICE_USERS)
    source_system = rng.choice(["pgsql", "mssql"])
    source_vm = "aidac-pgsql" if source_system == "pgsql" else "aidac-mssql"
    lifecycle = rng.choices(
        LIFECYCLE_PHASES,
        weights=[0.70, 0.10, 0.08, 0.08, 0.04],
        k=1
    )[0]

    sql_text = sql_template.format(id=rng.randint(1, 999999))
    rows_returned = rng.randint(0, 500) if command == "SELECT" else rng.randint(0, 20)
    risk_score = round(rng.uniform(0.01, 0.34), 4)

    return {
        "event_id": f"EVT-{event_id:06d}",
        "event_time": event_time.isoformat(),
        "partition": partition,
        "source_system": source_system,
        "source_vm": source_vm,
        "client_ip": rng.choice(CLIENTS),
        "database_name": db,
        "login_name": user,
        "lifecycle_phase": lifecycle,
        "event_type": event_type,
        "command_type": command,
        "object_name": rng.choice(["customers", "orders", "transactions", "audit_log", "users", "payments"]),
        "sql_text": sql_text,
        "query_length": len(sql_text),
        "rows_returned": rows_returned,
        "failed_login_count": 0,
        "privilege_changed": 0,
        "maintenance_window": 1 if lifecycle in ["backup_window", "maintenance_window", "deployment_window"] else 0,
        "risk_label": "normal",
        "severity": rng.choice(["INFO", "LOW"]),
        "is_attack": 0,
        "attack_scenario": "none",
        "risk_score": risk_score,
        "baseline_score": round(max(0.0, risk_score - rng.uniform(0.00, 0.07)), 4),
        "baseline_decision": "allow",
        "normalized_label": "normal",
        "dataset_version": "controlled_sql_lab_47832_v1_seed42",
    }

def anomalous_row(event_id, event_time, partition):
    scenario = rng.choice(ANOMALY_SCENARIOS)
    command, sql_template = rng.choice(ANOMALY_TEMPLATES[scenario])
    db = rng.choice(DATABASES)
    user = rng.choice(ATTACK_USERS)
    source_system = rng.choice(["pgsql", "mssql"])
    source_vm = "aidac-pgsql" if source_system == "pgsql" else "aidac-mssql"

    if scenario in ["backup_abuse", "slow_rate_privilege_creep"]:
        lifecycle = rng.choice(["backup_window", "maintenance_window", "normal_operation"])
    elif scenario == "schema_tampering":
        lifecycle = rng.choice(["deployment_window", "maintenance_window", "normal_operation"])
    else:
        lifecycle = rng.choice(LIFECYCLE_PHASES)

    sql_text = sql_template.format(id=rng.randint(1, 999999))

    failed_login_count = rng.randint(5, 40) if scenario == "authentication_burst" else rng.randint(0, 3)
    privilege_changed = 1 if scenario in ["privilege_escalation", "slow_rate_privilege_creep", "service_account_misuse"] else 0

    if scenario == "high_volume_extraction":
        rows_returned = rng.randint(5000, 250000)
    elif scenario == "authentication_burst":
        rows_returned = 0
    else:
        rows_returned = rng.randint(0, 5000)

    if scenario in ["sql_injection_evasion", "privilege_escalation", "schema_tampering"]:
        severity = rng.choice(["HIGH", "CRITICAL"])
        risk_label = "critical"
        risk_score = round(rng.uniform(0.82, 0.99), 4)
    else:
        severity = rng.choice(["MEDIUM", "HIGH", "CRITICAL"])
        risk_label = "suspicious"
        risk_score = round(rng.uniform(0.55, 0.91), 4)

    baseline_score = round(max(0.0, risk_score - rng.uniform(0.05, 0.22)), 4)

    return {
        "event_id": f"EVT-{event_id:06d}",
        "event_time": event_time.isoformat(),
        "partition": partition,
        "source_system": source_system,
        "source_vm": source_vm,
        "client_ip": rng.choice(CLIENTS),
        "database_name": db,
        "login_name": user,
        "lifecycle_phase": lifecycle,
        "event_type": "security_relevant",
        "command_type": command,
        "object_name": rng.choice(["customers", "orders", "transactions", "audit_log", "users", "payments", "sensitive_accounts", "payroll"]),
        "sql_text": sql_text,
        "query_length": len(sql_text),
        "rows_returned": rows_returned,
        "failed_login_count": failed_login_count,
        "privilege_changed": privilege_changed,
        "maintenance_window": 1 if lifecycle in ["backup_window", "maintenance_window", "deployment_window"] else 0,
        "risk_label": risk_label,
        "severity": severity,
        "is_attack": 1,
        "attack_scenario": scenario,
        "risk_score": risk_score,
        "baseline_score": baseline_score,
        "baseline_decision": "alert" if baseline_score >= 0.50 else "allow",
        "normalized_label": "anomalous",
        "dataset_version": "controlled_sql_lab_47832_v1_seed42",
    }

rows = []
event_id = 1
event_time = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

for partition in ["train", "validation", "test"]:
    for _ in range(COUNTS[partition]["normal"]):
        rows.append(normal_row(event_id, event_time, partition))
        event_id += 1
        event_time += timedelta(seconds=rng.randint(3, 30))

    for _ in range(COUNTS[partition]["anomalous"]):
        rows.append(anomalous_row(event_id, event_time, partition))
        event_id += 1
        event_time += timedelta(seconds=rng.randint(3, 30))

# Sort by time for temporal realism
rows.sort(key=lambda r: r["event_time"])

def write_csv(path, selected_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(selected_rows)

train_rows = [r for r in rows if r["partition"] == "train"]
val_rows = [r for r in rows if r["partition"] == "validation"]
test_rows = [r for r in rows if r["partition"] == "test"]

write_csv(RAW_FILE, rows)
write_csv(PROCESSED_FILE, rows)
write_csv(TRAIN_FILE, train_rows)
write_csv(VAL_FILE, val_rows)
write_csv(TEST_FILE, test_rows)

def describe(name, selected_rows):
    c = Counter(r["normalized_label"] for r in selected_rows)
    scenario_counts = Counter(r["attack_scenario"] for r in selected_rows if r["is_attack"] == 1)
    total = len(selected_rows)
    normal = c.get("normal", 0)
    anomalous = c.get("anomalous", 0)
    return {
        "name": name,
        "rows": total,
        "normal": normal,
        "anomalous": anomalous,
        "normal_pct": round(normal / total * 100, 2),
        "anomalous_pct": round(anomalous / total * 100, 2),
        "scenario_counts": dict(sorted(scenario_counts.items())),
    }

summary = [
    describe("FULL", rows),
    describe("TRAIN", train_rows),
    describe("VALIDATION", val_rows),
    describe("TEST", test_rows),
]

files = [RAW_FILE, PROCESSED_FILE, TRAIN_FILE, VAL_FILE, TEST_FILE]

manifest = {
    "dataset_name": "Controlled SQL laboratory dataset",
    "dataset_version": "controlled_sql_lab_47832_v1_seed42",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "generator_script": "scripts/create_controlled_sql_dataset_47832.py",
    "random_seed": SEED,
    "total_events": len(rows),
    "database_environment": "PostgreSQL 16 and SQL Server 2022-style telemetry",
    "description": (
        "Deterministic controlled laboratory dataset generated for AI-DAC evaluation. "
        "It represents database-oriented telemetry and attack scenarios. It is not production enterprise data."
    ),
    "splits": summary,
    "columns": FIELDNAMES,
    "files": [
        {
            "path": str(p.relative_to(ROOT)),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in files
    ],
    "limitations": [
        "Controlled laboratory dataset, not production telemetry.",
        "Generated deterministically from scenario templates.",
        "Used for RDBMS-specific evaluation claims within documented laboratory conditions.",
    ],
}

MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

report_lines = []
report_lines.append("Controlled SQL laboratory dataset report")
report_lines.append("=" * 70)
report_lines.append(f"Dataset version: controlled_sql_lab_47832_v1_seed42")
report_lines.append(f"Random seed: {SEED}")
report_lines.append(f"Total rows: {len(rows)}")
report_lines.append("")

for item in summary:
    report_lines.append(item["name"])
    report_lines.append("-" * 40)
    report_lines.append(f"Rows: {item['rows']}")
    report_lines.append(f"Normal: {item['normal']} ({item['normal_pct']}%)")
    report_lines.append(f"Anomalous: {item['anomalous']} ({item['anomalous_pct']}%)")
    if item["scenario_counts"]:
        report_lines.append("Attack scenario counts:")
        for k, v in item["scenario_counts"].items():
            report_lines.append(f"  {k}: {v}")
    report_lines.append("")

report_lines.append("Created files:")
for p in files:
    report_lines.append(f"  {p}")
report_lines.append(f"Manifest: {MANIFEST_FILE}")

REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
print("\n".join(report_lines))