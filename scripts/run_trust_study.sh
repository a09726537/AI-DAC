#!/bin/bash
set -euo pipefail
python scripts/eval.py --trust "$@"
