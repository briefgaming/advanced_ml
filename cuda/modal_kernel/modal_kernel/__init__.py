"""Run CUDA kernel workers on Modal Sandboxes.

Client machine needs **Modal + NumPy**. CUDA / CUTLASS / PyTorch live in the image.

Typical usage from a kernel package::

    from pathlib import Path
    from modal_kernel import build_image, load_npy, run_worker

    ROOT = Path(__file__).resolve().parents[1]

    def my_kernel_modal(q, k, v, *, opts=None, gpu="B200", **kw):
        out = run_worker(
            module="flashattention1.sandbox_worker",
            args=["/tmp/q.npy", "/tmp/k.npy", "/tmp/v.npy", "/tmp/out.npy", "/tmp/opts.json"],
            uploads={
                "/tmp/q.npy": q,
                "/tmp/k.npy": k,
                "/tmp/v.npy": v,
                "/tmp/opts.json": opts or {},
            },
            downloads=["/tmp/out.npy"],
            app_name="flashattention1",
            gpu=gpu,
            local_dirs=[ROOT],
            pip_packages=["numpy>=1.26", "torch", "nvidia-cutlass-dsl[cu13]>=4.5.0"],
            **kw,
        )
        return load_npy(out["/tmp/out.npy"])
"""

from __future__ import annotations

from modal_kernel.image import build_image
from modal_kernel.runner import run_worker
from modal_kernel.serialize import load_npy, materialize_upload

__all__ = [
    "build_image",
    "load_npy",
    "materialize_upload",
    "run_worker",
]
