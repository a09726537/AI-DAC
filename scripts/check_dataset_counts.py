import csv
from pathlib import Path
from collections import Counter

ROOT = Path.home() / "aidac-ai"

candidate_files = [
    ROOT / "data/processed/controlled_sql_processed.csv",
    ROOT / "data/processed/controlled_sql_train.csv",
    ROOT / "data/processed/controlled_sql_validation.csv",
    ROOT / "data/processed/controlled_sql_test.csv",
    ROOT / "data/controlled_sql_processed.csv",
    ROOT / "data/controlled_sql_test.csv",
    ROOT / "controlled_sql_processed.csv",
    ROOT / "controlled_sql_test.csv",
]

label_candidates = [
    "is_attack",
    "label",
    "target",
    "class",
    "y",
    "anomaly",
    "attack",
    "is_anomalous",
]

found_any = False

print("AI-DAC dataset count verification")
print("=" * 60)

for path in candidate_files:
    if not path.exists():
        continue

    found_any = True
    print(f"\nFILE: {path}")

    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []

        label_col = None
        for col in label_candidates:
            if col in columns:
                label_col = col
                break

        total_rows = 0
        label_counts = Counter()

        for row in reader:
            total_rows += 1
            if label_col:
                label_counts[row.get(label_col, "")] += 1

    print(f"Rows: {total_rows}")
    print(f"Columns: {columns}")

    if label_col:
        print(f"Label column: {label_col}")
        print("\nCounts:")
        for key, value in label_counts.items():
            print(f"{key}: {value}")

        print("\nPercentages:")
        for key, value in label_counts.items():
            pct = (value / total_rows) * 100 if total_rows else 0
            print(f"{key}: {pct:.2f}%")
    else:
        print("No label column found among:", label_candidates)

if not found_any:
    print("\nNo expected controlled SQL CSV file was found.")
    print("Searching for CSV files under ~/aidac-ai ...\n")

    csv_files = list(ROOT.rglob("*.csv"))
    for f in csv_files[:100]:
        print(f)

    print(f"\nTotal CSV files found: {len(csv_files)}")
    print("\nIf your controlled SQL dataset has another name, copy its path and send it here.")
