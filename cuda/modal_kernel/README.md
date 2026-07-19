# modal_kernel

Shared Modal Sandbox runner for CUDA kernel packages under `cuda/`.

Your laptop only needs **Modal** and **NumPy**. Each kernel package supplies:

1. An in-sandbox **worker** (`python -m your_pkg.sandbox_worker …`)
2. Kernel code imported only inside that worker
3. A thin client wrapper that calls `run_worker`

## Install

```bash
cd cuda/modal_kernel
pip install -e .
# or: pip install -r requirements.txt && export PYTHONPATH="$PWD:$PYTHONPATH"
```

Authenticate: [Modal docs](https://modal.com/docs/guide) (`modal token new`).

## API

| Function | Role |
|---|---|
| `build_image(...)` | Debian slim + pip + local dirs on `PYTHONPATH=/workspace` |
| `run_worker(...)` | Create sandbox → upload → `python -m module args` → download |
| `load_npy(bytes)` | Decode a downloaded `.npy` |

Uploads accept NumPy arrays (saved as `.npy`), dict/list (JSON), paths, bytes, or strings.

## Kernel package checklist

```text
your_kernel/
  your_kernel/
    __init__.py          # export your_kernel_modal
    modal_client.py      # thin wrap of run_worker (optional name)
    sandbox_worker.py    # runs on Modal GPU only
    kernel.py            # CUTLASS / CUDA / torch — sandbox only
  requirements.txt       # local: modal-kernel (+ numpy if needed)
```

Minimal client:

```python
from pathlib import Path
from modal_kernel import load_npy, run_worker

ROOT = Path(__file__).resolve().parents[1]

def flash_attn_modal(q, k, v, *, opts=None, app_name="flashattention1", gpu="B200", **kw):
    results = run_worker(
        module="flashattention1.sandbox_worker",
        args=["/tmp/q.npy", "/tmp/k.npy", "/tmp/v.npy", "/tmp/out.npy", "/tmp/opts.json"],
        uploads={
            "/tmp/q.npy": q,
            "/tmp/k.npy": k,
            "/tmp/v.npy": v,
            "/tmp/opts.json": opts or {},
        },
        downloads=["/tmp/out.npy"],
        app_name=app_name,
        gpu=gpu,
        local_dirs=[ROOT],
        pip_packages=["numpy>=1.26", "torch", "nvidia-cutlass-dsl[cu13]>=4.5.0"],
        **kw,
    )
    return load_npy(results["/tmp/out.npy"])
```

See `matmul/blackwell-matmul` for a full example.

GPU SKUs: https://modal.com/docs/guide/gpu
