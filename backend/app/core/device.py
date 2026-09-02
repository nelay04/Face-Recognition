"""Choosing between the CPU and GPU face detectors.

dlib exposes two detectors with very different cost profiles: ``hog`` is a
classical CPU detector, and ``cnn`` is a neural detector that is markedly more
accurate on rotated, small or poorly-lit faces but needs CUDA to be tolerable
— on CPU it is roughly an order of magnitude slower than ``hog``.

Whether CUDA is usable is a property of *how dlib was compiled*, not of the
machine: a host can have a perfectly good GPU while the installed dlib wheel
has no CUDA support at all. That distinction is invisible from the outside and
is the single most common source of "why is my GPU idle", so it is surfaced
here in the resolution detail rather than left to be guessed at.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from backend.app.core.config import DetectionModel

logger = logging.getLogger(__name__)

ComputeDevice = Literal["auto", "gpu", "cpu"]


class GpuUnavailableError(RuntimeError):
    """``COMPUTE_DEVICE=gpu`` was demanded but CUDA cannot be used."""


@dataclass(frozen=True, slots=True)
class DeviceChoice:
    """The outcome of resolving a device request."""

    device: Literal["gpu", "cpu"]
    detection_model: DetectionModel
    detail: str

    def __str__(self) -> str:
        return f"{self.device} ({self.detection_model}) — {self.detail}"


def cuda_status() -> tuple[bool, str]:
    """Whether dlib can actually use a GPU, and why not when it cannot.

    Importing dlib is deferred to call time: the module is heavy, and the
    settings layer must stay importable without it.
    """
    try:
        import dlib
    except ImportError as exc:  # pragma: no cover - dlib is a hard dependency
        return False, f"dlib could not be imported ({exc})"

    if not dlib.DLIB_USE_CUDA:
        return False, "the installed dlib wheel was built without CUDA support"

    devices = dlib.cuda.get_num_devices()
    if devices < 1:
        return False, "dlib has CUDA support but no CUDA device is visible"

    return True, f"CUDA enabled, {devices} device(s) visible"


def resolve(
    requested: ComputeDevice,
    *,
    override: DetectionModel | None = None,
) -> DeviceChoice:
    """Pick the detector to run.

    ``override`` is an explicit ``DETECTION_MODEL``; it wins over the device
    preference, because someone naming a model has said what they want. The
    device is still probed so the log line reports where that model will run.

    Raises:
        GpuUnavailableError: ``requested`` is ``"gpu"`` and CUDA is unusable.
            An explicit demand for a GPU fails loudly rather than silently
            running ten times slower on the CPU.
    """
    if requested == "cpu":
        # Trust the request without probing: asking for the CPU is always
        # satisfiable, and importing dlib to confirm that would be pointless.
        return DeviceChoice(
            device="cpu",
            detection_model=override or "hog",
            detail="CPU requested",
        )

    available, reason = cuda_status()

    if requested == "gpu" and not available:
        raise GpuUnavailableError(
            f"COMPUTE_DEVICE=gpu was requested but {reason}. "
            "Set COMPUTE_DEVICE=auto to fall back to the CPU, or install a "
            "CUDA-enabled dlib build."
        )

    if available:
        return DeviceChoice(device="gpu", detection_model=override or "cnn", detail=reason)

    return DeviceChoice(
        device="cpu",
        detection_model=override or "hog",
        detail=f"falling back to CPU: {reason}",
    )
