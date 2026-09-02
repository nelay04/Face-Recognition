#!/usr/bin/env bash
#
# Runs the API with autoreload against the local .env.
# Assumes ./scripts/setup.sh has already been run.
#
set -euo pipefail

cd "$(dirname "$0")/.."

[[ -f .env ]] || { echo "No .env found — copy it with: cp .env.example .env"; exit 1; }

# HOST/PORT live in Settings for the app to report on, but uvicorn itself
# needs them as CLI flags — read them the same way Settings would, so this
# script and .env never disagree about which port is "the" port.
HOST="$(grep -m1 -E '^HOST=' .env | cut -d= -f2- || true)"
PORT="$(grep -m1 -E '^PORT=' .env | cut -d= -f2- || true)"

exec .venv/bin/uvicorn backend.app.main:app --reload \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8013}"
