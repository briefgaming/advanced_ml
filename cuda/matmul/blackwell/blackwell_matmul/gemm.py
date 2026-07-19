"""In-sandbox GEMM only — CUTLASS CuTe persistent kernel (not intended for local laptops)."""

from __future__ import annotations

from typing import Any, Literal, Tuple, Type


def _identity_epilogue(x):
    return x


def assert_blackwell(device: "torch.device | int | None" = None) -> None:
    """Raise if the active CUDA device is not Blackwell-class (compute capability >= 10.0)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Blackwell GEMM.")
    if device is None:
        dev_index = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        dev_index = device.index if device.index is not None else torch.cuda.current_device()
    else:
        dev_index = int(device)
    major, minor = torch.cuda.get_device_capability(dev_index)
    if major < 10:
        raise RuntimeError(
            "This kernel targets NVIDIA Blackwell (SM100, compute capability 10.x). "
            f"Device {dev_index} reports capability {major}.{minor}."
        )


def _as_batched(
    a: "torch.Tensor", b: "torch.Tensor"
) -> Tuple["torch.Tensor", "torch.Tensor", Tuple[int, int, int, int]]:
    """Normalize to (L,M,K) @ (L,K,N) and return (mnkl) as (M,N,K,L)."""
    import torch

    if a.dim() == 2:
        a = a.unsqueeze(0)
    if b.dim() == 2:
        b = b.unsqueeze(0)
    if a.dim() != 3 or b.dim() != 3:
        raise ValueError(f"Expected 2D or 3D tensors; got {a.shape=} {b.shape=}")
    l, m, ka = a.shape
    lb, kb, n = b.shape
    if l != lb or ka != kb:
        raise ValueError(f"Incompatible shapes for bmm: {a.shape}, {b.shape}")
    return a, b, (m, n, ka, l)


def blackwell_matmul(
    a: "torch.Tensor",
    b: "torch.Tensor",
    *,
    out: "torch.Tensor | None" = None,
    ab_dtype: Type[Any] | None = None,
    c_dtype: Type[Any] | None = None,
    acc_dtype: Type[Any] | None = None,
    a_major: Literal["k", "m"] = "k",
    b_major: Literal["k", "n"] = "n",
    c_major: Literal["n", "m"] = "n",
    mma_tiler_mn: Tuple[int, int] = (256, 128),
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    use_2cta_instrs: bool = True,
    use_tma_store: bool = True,
    check_device_arch: bool = True,
) -> "torch.Tensor":
    """
    Fused batched GEMM on Blackwell using tcgen05.mma (CuTe persistent scheduler).

    Tensor layouts match PyTorch :func:`torch.bmm`:
      - ``a``: ``(..., M, K)``
      - ``b``: ``(..., K, N)``
      - returns ``(..., M, N)``

    Defaults use FP16 inputs, FP32 accumulation, FP16 output, 2CTA MMA, TMA store,
    and a ``(256,128)`` MMA tile with cluster ``(2,1)`` — a strong baseline on SM100.

    Parameters mirror the vendored NVIDIA ``dense_gemm_persistent`` example; see that file
    for dtype and tiling constraints.
    """
    import torch

    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.utils as utils
    from cutlass.torch import dtype as torch_dtype
    from cutlass.utils import create_cute_tensor_for_fp8, is_fp8_dtype

    from blackwell_matmul.vendor.dense_gemm_persistent import compile_bmm

    if ab_dtype is None:
        ab_dtype = cutlass.Float16
    if acc_dtype is None:
        acc_dtype = cutlass.Float32

    if check_device_arch:
        assert_blackwell(a.device)

    cdtype = c_dtype or ab_dtype
    a_t, b_t, mnkl = _as_batched(a, b)
    m, n, k, batch = mnkl

    device = a_t.device
    torch_stream = torch.cuda.current_stream(device)
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    cluster_ctas = cluster_shape_mn[0] * cluster_shape_mn[1]
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(cluster_ctas)

    out_dtype_torch = torch_dtype(cdtype)
    if out is None:
        c_t = torch.empty(batch, m, n, device=device, dtype=out_dtype_torch)
    else:
        if out.shape != (batch, m, n):
            raise ValueError(f"{out.shape=} expected {(batch, m, n)}")
        if out.device != device:
            raise ValueError("out must be on the same device as inputs")
        if out.dtype != out_dtype_torch:
            raise ValueError(f"out.dtype={out.dtype} expected {out_dtype_torch} for c_dtype={cdtype}")
        c_t = out

    leading_dim_a = 2 if a_major == "k" else 1
    leading_dim_b = 1 if b_major == "k" else 2
    leading_dim_c = 2 if c_major == "n" else 1

    ab_torch_dtype = torch_dtype(ab_dtype)
    a_storage = a_t.detach().to(dtype=ab_torch_dtype).contiguous()
    b_storage = b_t.detach().to(dtype=ab_torch_dtype).contiguous()
    c_storage = c_t.detach().contiguous()

    a_src = a_t.detach().float() if is_fp8_dtype(ab_dtype) else None
    b_src = b_t.detach().float() if is_fp8_dtype(ab_dtype) else None
    c_src = (
        torch.zeros(batch, m, n, device=device, dtype=torch.float32)
        if is_fp8_dtype(cdtype)
        else None
    )

    a_ = create_cute_tensor_for_fp8(a_storage, ab_dtype, leading_dim_a, a_src)
    b_ = create_cute_tensor_for_fp8(b_storage, ab_dtype, leading_dim_b, b_src)
    c_ = create_cute_tensor_for_fp8(c_storage, cdtype, leading_dim_c, c_src)

    compiled_fn = compile_bmm(
        mnkl,
        a_,
        b_,
        c_,
        acc_dtype,
        a_major,
        b_major,
        c_major,
        mma_tiler_mn,
        cluster_shape_mn,
        max_active_clusters,
        use_2cta_instrs,
        use_tma_store,
        _identity_epilogue,
    )
    compiled_fn(a_, b_, c_, current_stream)

    if out is None:
        return c_storage
    return out
