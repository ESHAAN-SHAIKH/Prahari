#!/usr/bin/env bash
# Run the PRAHARI prototype.
#
#   ./run.sh              synthetic sensor model, http://localhost:8000
#   PRAHARI_CKPT=... PRAHARI_CFG=... PRAHARI_SCANS=... ./run.sh
#                          live FRNet checkpoint instead — see README.md
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating .venv ..."
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

echo
echo "PRAHARI starting on http://localhost:8000"
if [ -z "${PRAHARI_CKPT:-}" ]; then
  echo "No PRAHARI_CKPT set — running the synthetic sensor model."
  echo "The dashboard badges every frame as synthetic; see README.md to attach a checkpoint."
fi
echo

exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
