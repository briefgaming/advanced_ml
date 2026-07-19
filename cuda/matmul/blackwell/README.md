# Blackwell

CuTe DSL GEMM runs **inside a Modal Sandbox on B200**. Your machine only needs **Modal** and **NumPy** to submit jobs.

Modal orchestration is shared via `[cuda/modal_kernel](../../modal_kernel)` — this package only defines the GEMM worker and a thin client wrapper.

## Layout

- `blackwell_matmul/modal_sandbox.py` — `blackwell_matmul_modal()` (calls `modal_kernel.run_worker`)
- `blackwell_matmul/sandbox_worker.py` — entrypoint inside the sandbox
- `blackwell_matmul/gemm.py` + `vendor/` — loaded **only in the sandbox** (CUTLASS, CUDA PyTorch)

## Setup

1. [Modal account & CLI auth](https://modal.com/docs/guide)
2. Install the shared runner + this package's local deps:

```bash
pip install -e ../../modal_kernel
pip install -r requirements.txt
```

## Run

```bash
cd blackwell
PYTHONPATH=. python -c "
import numpy as np
from blackwell_matmul import blackwell_matmul_modal

L, M, K, N = 8, 512, 512, 512
a = np.random.randn(L, M, K).astype('float16')
b = np.random.randn(L, K, N).astype('float16')
c = blackwell_matmul_modal(a, b, stream_build_logs=True)
print(c.shape, c.dtype)
"
```

Optional tiling/dtypes via `opts=` (same keys as `sandbox_worker`).

GPU SKUs: [https://modal.com/docs/guide/gpu](https://modal.com/docs/guide/gpu)