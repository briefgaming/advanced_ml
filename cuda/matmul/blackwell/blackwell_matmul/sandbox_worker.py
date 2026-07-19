"""Sandbox entrypoint: runs :func:`blackwell_matmul.gemm.blackwell_matmul` on Modal GPU."""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict

import cutlass
import numpy as np
import torch


_CUTLASS_NAMES = (
    "Float16",
    "BFloat16",
    "Float32",
    "TFloat32",
    "Float8E4M3FN",
    "Float8E5M2",
    "Int8",
    "Uint8",
    "Int32",
)


def _dtype(name: str) -> type:
    if not hasattr(cutlass, name):
        raise ValueError(f"cutlass has no dtype {name!r}")
    if name not in _CUTLASS_NAMES:
        raise ValueError(f"Unsupported CUTLASS dtype name {name!r}")
    return getattr(cutlass, name)


def _run_from_disk(a_path: str, b_path: str, c_path: str, opts: Dict[str, Any]) -> None:
    ab_dtype = _dtype(opts.get("ab_dtype", "Float16"))
    c_dtype_name = opts.get("c_dtype")
    c_dtype = _dtype(c_dtype_name) if c_dtype_name else None
    acc_dtype = _dtype(opts.get("acc_dtype", "Float32"))

    a_np = np.load(a_path, mmap_mode=None)
    b_np = np.load(b_path, mmap_mode=None)
    a = torch.from_numpy(np.asarray(a_np, copy=True)).cuda(non_blocking=True)
    b = torch.from_numpy(np.asarray(b_np, copy=True)).cuda(non_blocking=True)

    kwargs = dict(
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        acc_dtype=acc_dtype,
        a_major=opts.get("a_major", "k"),
        b_major=opts.get("b_major", "n"),
        c_major=opts.get("c_major", "n"),
        mma_tiler_mn=tuple(opts.get("mma_tiler_mn", [256, 128])),
        cluster_shape_mn=tuple(opts.get("cluster_shape_mn", [2, 1])),
        use_2cta_instrs=bool(opts.get("use_2cta_instrs", True)),
        use_tma_store=bool(opts.get("use_tma_store", True)),
        check_device_arch=bool(opts.get("check_device_arch", True)),
    )

    from blackwell_matmul.gemm import blackwell_matmul

    c_t = blackwell_matmul(a, b, **kwargs)
    c_np = c_t.detach().float().cpu().numpy()
    np.save(c_path, c_np, allow_pickle=False)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 3:
        print(
            "usage: sandbox_worker.py <a.npy> <b.npy> <c_out.npy> [opts.json]",
            file=sys.stderr,
        )
        return 2
    a_path, b_path, c_path = argv[0], argv[1], argv[2]
    opts: Dict[str, Any] = {}
    if len(argv) > 3:
        opts = json.loads(Path(argv[3]).read_text(encoding="utf-8"))

    _run_from_disk(a_path, b_path, c_path, opts)

    summary = {
        "status": "ok",
        "c_path": c_path,
        "out_dtype": "float32",
        "note": "Result saved as float32 .npy for stable interchange (including FP8 outputs).",
    }
    print(json.dumps(summary))
    return 0


def run_payload_json(payload: str) -> Dict[str, Any]:
    """Decode JSON with base64-encoded ``.npy`` payloads (``a_b64``, ``b_b64``)."""
    data = json.loads(payload)
    opts = data.get("opts") or {}
    a_np = np.load(io.BytesIO(base64.b64decode(data["a_b64"])), allow_pickle=False)
    b_np = np.load(io.BytesIO(base64.b64decode(data["b_b64"])), allow_pickle=False)
    np.save("/tmp/a_payload.npy", a_np, allow_pickle=False)
    np.save("/tmp/b_payload.npy", b_np, allow_pickle=False)

    _run_from_disk("/tmp/a_payload.npy", "/tmp/b_payload.npy", "/tmp/c_payload.npy", opts)

    c_np = np.load("/tmp/c_payload.npy", mmap_mode=None)
    buf = io.BytesIO()
    np.save(buf, c_np, allow_pickle=False)
    return {
        "status": "ok",
        "c_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "summary": {"shape": list(c_np.shape), "dtype": str(c_np.dtype)},
    }


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--payload-stdin":
        print(json.dumps(run_payload_json(sys.stdin.read())))
        sys.exit(0)
    sys.exit(main())
