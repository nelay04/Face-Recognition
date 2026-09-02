"""Unit tests for compute-device resolution.

CUDA availability is patched rather than probed: the suite must give the same
answer on a CI box with no GPU and on a workstation with one.
"""

from __future__ import annotations

import pytest

from backend.app.core import device as device_module
from backend.app.core.device import GpuUnavailableError, resolve


@pytest.fixture
def with_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_module, "cuda_status", lambda: (True, "CUDA enabled, 1 device(s)"))


@pytest.fixture
def without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device_module, "cuda_status", lambda: (False, "built without CUDA"))


def test_auto_uses_the_gpu_when_cuda_is_available(with_cuda: None) -> None:
    choice = resolve("auto")
    assert choice.device == "gpu"
    assert choice.detection_model == "cnn"


def test_auto_falls_back_to_cpu_without_cuda(without_cuda: None) -> None:
    """A missing GPU must not stop the service from starting."""
    choice = resolve("auto")
    assert choice.device == "cpu"
    assert choice.detection_model == "hog"
    assert "built without CUDA" in choice.detail


def test_explicit_gpu_refuses_to_fall_back(without_cuda: None) -> None:
    """Asking for a GPU and silently getting a 10x slower CPU run is a trap."""
    with pytest.raises(GpuUnavailableError, match="built without CUDA"):
        resolve("gpu")


def test_explicit_cpu_never_probes_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode() -> tuple[bool, str]:
        raise AssertionError("cuda_status must not be called for COMPUTE_DEVICE=cpu")

    monkeypatch.setattr(device_module, "cuda_status", explode)
    assert resolve("cpu").detection_model == "hog"


def test_an_explicit_model_overrides_the_device_default(with_cuda: None) -> None:
    """Naming DETECTION_MODEL is a statement of intent; the device yields."""
    choice = resolve("auto", override="hog")
    assert choice.device == "gpu"
    assert choice.detection_model == "hog"


def test_cuda_status_reports_a_reason_when_unavailable() -> None:
    """Whatever the real answer is, it must come with an explanation."""
    available, reason = device_module.cuda_status()
    assert isinstance(available, bool)
    assert reason
