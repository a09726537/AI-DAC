import json
import hashlib
import csv
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path.home() / "aidac-ai"
DATA_ROOT = ROOT / "external_datasets"
OUT_DIR = ROOT / "reproducibility" / "manifests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIVE_DATASETS = {
    "BoT-IoT": {
        "path": "BoT-IoT",
        "role": "Comparative benchmark for botnet and high-volume attack behavior",
        "active": True
    },
    "CSE-CIC-IDS2018": {
        "path": "CIC-IDS2018",
        "role": "Comparative benchmark using processed CICFlowMeter CSV files",
        "active": True
    },
    "NSL-KDD": {
        "path": "NSL-KDD",
        "role": "Legacy IDS benchmark for methodological continuity",
        "active": True
    },
    "TON_IoT": {
        "path": "TON_IoT",
        "role": "IoT and network telemetry benchmark",
        "active": True
    },
    "UNSW-NB15": {
        "path": "UNSW-NB15 2024",
        "role": "Modern network intrusion-detection benchmark",
        "active": True
    },
    "LogHub": {
        "path": "LogHub",
        "role": "Available log dataset collection; not used as an active primary evaluation dataset",
        "active": False
    }
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

summary_rows = []

for dataset_name, meta in ACTIVE_DATASETS.items():
    dataset_dir = DATA_ROOT / meta["path"]

    files = []
    if dataset_dir.exists():
        for p in sorted(dataset_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in [".csv", ".txt", ".xls", ".xlsx"]:
                size_bytes = p.stat().st_size
                file_record = {
                    "relative_path": str(p.relative_to(DATA_ROOT)),
                    "file_name": p.name,
                    "size_bytes": size_bytes,
                    "size_mb": round(size_bytes / (1024 * 1024), 2),
                    "sha256": sha256_file(p)
                }
                files.append(file_record)

    manifest = {
        "dataset_name": dataset_name,
        "active_for_evaluation": meta["active"],
        "role_in_dissertation": meta["role"],
        "windows_source_path": "G:\\VMware\\vmshare",
        "linux_mount_path": "/mnt/hgfs/vmshare",
        "project_path": str(DATA_ROOT),
        "manifest_created_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "total_size_mb": round(sum(f["size_bytes"] for f in files) / (1024 * 1024), 2),
        "files": files,
        "limitations": (
            "Public benchmark dataset. Used for comparative intrusion-detection evidence. "
            "Not treated as a substitute for database-specific SQL audit telemetry."
        )
    }

    out_file = OUT_DIR / f"{dataset_name.replace(' ', '_').replace('-', '_')}_manifest.json"
    out_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_rows.append({
        "dataset_name": dataset_name,
        "active_for_evaluation": meta["active"],
        "file_count": len(files),
        "total_size_mb": manifest["total_size_mb"],
        "manifest_file": str(out_file.relative_to(ROOT))
    })

summary_csv = OUT_DIR / "dataset_manifest_summary.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "dataset_name",
            "active_for_evaluation",
            "file_count",
            "total_size_mb",
            "manifest_file"
        ]
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print("Dataset manifests created:")
for row in summary_rows:
    print(row)

print(f"\nSummary written to: {summary_csv}")
