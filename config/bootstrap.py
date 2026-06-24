"""Runtime compatibility shims — import before chromadb / ONNX protobuf users."""

from __future__ import annotations

import os

_applied = False


def apply_runtime_compat() -> None:
    """Apply env fixes once per process (Chroma ONNX + protobuf 4+)."""
    global _applied
    if _applied:
        return
    # Avoid: "Descriptors cannot be created directly" from onnx/chromadb on protobuf 4+.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    _applied = True


def apply_settings_env(settings) -> None:
    """Apply optional .env-driven runtime flags."""
    impl = getattr(settings, "protocol_buffers_python_implementation", "").strip()
    if impl:
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", impl)
    apply_runtime_compat()


apply_runtime_compat()
