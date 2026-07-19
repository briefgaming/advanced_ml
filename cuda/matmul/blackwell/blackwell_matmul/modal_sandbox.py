"""Modal client for Blackwell CuTe GEMM — orchestration lives in ``modal_kernel``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from modal_kernel import build_image, load_npy, run_worker

_PIP = (
    "numpy>=1.26",
    "torch",
    "nvidia-cutlass-dsl[cu13]>=4.5.0",
)


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def blackwell_modal_image():
    """Image with PyTorch, CUTLASS DSL (CUDA 13 wheels), and this package under ``/workspace``."""
    return build_image(
        local_dirs=[_package_root()],
        pip_packages=_PIP,
    )


def blackwell_matmul_modal(
    a: np.ndarray,
    b: np.ndarray,
    *,
    opts: Dict[str, Any] | None = None,
    app_name: str = "blackwell-matmul",
    gpu: str = "B200",
    timeout: int = 900,
    stream_build_logs: bool = False,
) -> np.ndarray:
    """
    Upload ``a`` and ``b``, run :mod:`blackwell_matmul.sandbox_worker` on a Modal Sandbox,
    and download ``c`` as a float32 NumPy array.

    Requires ``modal`` CLI auth (``modal token new``) and a quota that includes ``gpu``.

    Parameters
    ----------
    opts
        Optional JSON-serializable keywords for dtypes and tiling, for example::

            {"ab_dtype": "Float16", "acc_dtype": "Float32", "mma_tiler_mn": [256, 128]}
    """
    results = run_worker(
        module="blackwell_matmul.sandbox_worker",
        args=["/tmp/a.npy", "/tmp/b.npy", "/tmp/c.npy", "/tmp/opts.json"],
        uploads={
            "/tmp/a.npy": np.asarray(a),
            "/tmp/b.npy": np.asarray(b),
            "/tmp/opts.json": dict(opts or {}),
        },
        downloads=["/tmp/c.npy"],
        app_name=app_name,
        gpu=gpu,
        timeout=timeout,
        stream_build_logs=stream_build_logs,
        image=blackwell_modal_image(),
    )
    return load_npy(results["/tmp/c.npy"])


__all__ = ["blackwell_modal_image", "blackwell_matmul_modal"]
