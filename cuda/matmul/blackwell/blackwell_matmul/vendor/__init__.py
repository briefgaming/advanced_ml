"""Vendored CUTLASS CuTe DSL Blackwell persistent GEMM (upstream: NVIDIA CUTLASS)."""

from .dense_gemm_persistent import compile_bmm, PersistentDenseGemmKernel

__all__ = ["compile_bmm", "PersistentDenseGemmKernel"]
