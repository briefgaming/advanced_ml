"""Orchestrate Modal Sandbox jobs: upload inputs, run a worker module, download outputs."""

from __future__ import annotations

import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

from modal_kernel.image import build_image
from modal_kernel.serialize import UploadValue, materialize_upload


def run_worker(
    *,
    module: str,
    args: Sequence[str],
    uploads: Mapping[str, UploadValue],
    downloads: Sequence[str],
    app_name: str,
    gpu: str = "B200",
    timeout: int = 900,
    stream_build_logs: bool = False,
    image: Any | None = None,
    local_dirs: Sequence[Path | str] | Mapping[Path | str, str] = (),
    pip_packages: Sequence[str] = (),
    python_version: str = "3.12",
    env: Mapping[str, str] | None = None,
    workspace: str = "/workspace",
    exec_timeout: int | None = None,
) -> dict[str, bytes]:
    """
    Run ``python -m <module> <args...>`` inside a Modal Sandbox and return downloads.

    Parameters
    ----------
    module
        Importable worker module inside the image (e.g. ``blackwell_matmul.sandbox_worker``).
    args
        CLI args passed to the worker (typically absolute sandbox paths under ``/tmp``).
    uploads
        Map of sandbox destination path → value (``.npy`` arrays, JSON dicts, files, …).
    downloads
        Sandbox paths to read after a successful run.
    image
        Prebuilt Modal image. If omitted, built from ``local_dirs`` / ``pip_packages``.
    exec_timeout
        Timeout for ``sb.exec`` only. Defaults to ``min(timeout, max(timeout - 120, 60))``.

    Returns
    -------
    dict[str, bytes]
        Raw bytes for each path in ``downloads``.
    """
    import modal

    if image is None:
        image = build_image(
            local_dirs=local_dirs,
            pip_packages=pip_packages,
            python_version=python_version,
            env=env,
            workspace=workspace,
        )

    sb_app = modal.App.lookup(app_name, create_if_missing=True)
    log_cm = modal.enable_output() if stream_build_logs else nullcontext()
    with log_cm:
        sb = modal.Sandbox.create(
            app=sb_app,
            image=image,
            gpu=gpu,
            timeout=timeout,
        )

    try:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            for i, (remote_path, value) in enumerate(uploads.items()):
                local_path = base / f"upload_{i}{Path(remote_path).suffix or '.bin'}"
                materialize_upload(value, local_path)
                sb.filesystem.copy_from_local(str(local_path), remote_path)

        if exec_timeout is None:
            exec_timeout = min(timeout, max(timeout - 120, 60))

        proc = sb.exec(
            "python",
            "-m",
            module,
            *list(args),
            timeout=exec_timeout,
        )
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"Sandbox worker {module!r} failed rc={proc.returncode}\n"
                f"stdout:\n{proc.stdout.read()}\nstderr:\n{proc.stderr.read()}"
            )

        return {path: sb.filesystem.read_bytes(path) for path in downloads}
    finally:
        sb.terminate(wait=True)
        sb.detach()
