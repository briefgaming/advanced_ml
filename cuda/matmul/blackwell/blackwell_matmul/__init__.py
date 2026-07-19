"""Blackwell GEMM on Modal (CuTe DSL in-container only).

Your laptop needs **Modal + NumPy** only; CUDA/CUTLASS/PyTorch are installed in the remote image.
Sandbox orchestration comes from the shared ``modal_kernel`` package.

Prefer::

    from blackwell_matmul import blackwell_matmul_modal
"""

from __future__ import annotations

__all__ = [
    "blackwell_matmul_modal",
    "blackwell_modal_image",
]


def __getattr__(name: str):
    if name == "blackwell_matmul_modal":
        from blackwell_matmul.modal_sandbox import blackwell_matmul_modal

        return blackwell_matmul_modal
    if name == "blackwell_modal_image":
        from blackwell_matmul.modal_sandbox import blackwell_modal_image

        return blackwell_modal_image
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
