from pathlib import Path

ROOT = Path.home() / "aidac-ai"
DATA_ROOT = ROOT / "external_datasets"

print("External dataset inventory")
print("=" * 70)
print(f"Dataset root: {DATA_ROOT}")
print()

if not DATA_ROOT.exists():
    print("ERROR: external_datasets does not exist or is not mounted.")
    raise SystemExit(1)

files = sorted([
    p for p in DATA_ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in [".csv", ".txt", ".xls", ".xlsx"]
])

for p in files:
    size_mb = p.stat().st_size / (1024 * 1024)
    rel = p.relative_to(DATA_ROOT)
    print(f"{rel} | {size_mb:.2f} MB")

print()
print(f"Total dataset files found: {len(files)}")
