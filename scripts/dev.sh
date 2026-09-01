#!/usr/bin/env bash
#
# Runs the API with autoreload against the local .env.
# Assumes ./scripts/setup.sh has already been run.
#
set -euo pipefail

cd "$(dirname "$0")/.."

[[ -f .env ]] || { echo "No .env found — copy it with: cp .env.example .env"; exit 1; }

exec .venv/bin/uvicorn backend.app.main:app --reload
