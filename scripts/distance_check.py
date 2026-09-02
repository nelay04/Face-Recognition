#!/usr/bin/env python3
"""Report the distance between two photos under each encoding configuration.

No face photograph ships with this repository, so the accuracy effect of
``ENCODING_MODEL`` and the jitter settings cannot be measured in the test
suite. Point this at your own images instead:

    python scripts/distance_check.py me1.jpg me2.jpg          # same person
    python scripts/distance_check.py me.jpg someone-else.jpg  # different

A lower number for the same person, and a higher one for different people,
is the improvement. Compare against MATCH_TOLERANCE (0.6 by default).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from backend.app.core.config import EncodingModel
from backend.app.services.encoder import FaceEncoder
from backend.app.utils.image import decode_image

# (label, encoding model, jitters). The first row is the old behaviour, so the
# table reads as a before/after rather than a lone number.
CONFIGURATIONS: list[tuple[str, EncodingModel, int]] = [
    ("small, 1 jitter (previous default)", "small", 1),
    ("large, 1 jitter", "large", 1),
    ("large, 10 jitters (enrolment default)", "large", 10),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument(
        "--max-edge",
        type=int,
        default=640,
        help="Downscale cap applied before detection, as the service does.",
    )
    args = parser.parse_args()

    print(f"{'configuration':<40} {'distance':>9} {'ms':>7}")
    print("-" * 58)

    for label, model, jitters in CONFIGURATIONS:
        encoder = FaceEncoder(upsample=1, max_edge=args.max_edge, encoding_model=model)
        started = time.perf_counter()
        try:
            left = encoder.encode_single(_load(args.first), jitters=jitters)
            right = encoder.encode_single(_load(args.second), jitters=jitters)
        except Exception as exc:  # a CLI reports failures, it does not raise
            print(f"{label:<40} {type(exc).__name__}: {exc}")
            continue
        elapsed = (time.perf_counter() - started) * 1000
        distance = float(np.linalg.norm(left.embedding - right.embedding))
        print(f"{label:<40} {distance:>9.4f} {elapsed:>7.0f}")

    return 0


def _load(path: Path) -> np.ndarray:
    # Generous limits: this is a local diagnostic, not an upload endpoint.
    return decode_image(
        path.read_bytes(),
        max_bytes=64 * 1024 * 1024,
        max_pixels=80_000_000,
        min_edge=32,
    )


if __name__ == "__main__":
    sys.exit(main())
