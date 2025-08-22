#!/bin/bash
set -euo pipefail
python scripts/train.py --mode prequential "$@"
