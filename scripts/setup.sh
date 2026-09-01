#!/usr/bin/env bash
#
# Creates the virtualenv and installs dependencies in the one order that works.
# Idempotent: safe to re-run.
#
#   ./scripts/setup.sh          # runtime deps only
#   ./scripts/setup.sh --dev    # runtime + tooling
#
set -euo pipefail

cd "$(dirname "$0")/.."

VENV=.venv
REQ=requirements/base.txt
[[ "${1:-}" == "--dev" ]] && REQ=requirements/dev.txt

if [[ ! -d "$VENV" ]]; then
  echo "==> Creating virtualenv at $VENV"
  python3 -m venv "$VENV"
fi

PY="$VENV/bin/python"

echo "==> Upgrading pip"
"$PY" -m pip install --quiet --upgrade pip

echo "==> Installing from $REQ"
"$PY" -m pip install -r "$REQ"

# Must come last and without dependency resolution: face-recognition's metadata
# demands the source-only `dlib` distribution, which would shadow dlib-bin.
echo "==> Installing face-recognition (--no-deps)"
"$PY" -m pip install --no-deps face-recognition==1.3.0

echo "==> Verifying the native stack"
"$PY" - <<'PYCHECK'
import warnings
warnings.filterwarnings("ignore")

import numpy, dlib, cv2, PIL, face_recognition

print(f"    numpy  {numpy.__version__}")
print(f"    dlib   {dlib.__version__}")
print(f"    opencv {cv2.__version__}")
print(f"    pillow {PIL.__version__}")

# Exercise the models end-to-end; a broken install fails here, not in prod.
blank = numpy.zeros((80, 80, 3), dtype=numpy.uint8)
face_recognition.face_encodings(blank)
print("    models loaded OK")
PYCHECK

echo
echo "Done. Activate with:  source $VENV/bin/activate"
