"""Modal image builders for CUDA kernel sandboxes."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence


def build_image(
    *,
    local_dirs: Sequence[Path | str] | Mapping[Path | str, str] = (),
    pip_packages: Sequence[str] = (),
    python_version: str = "3.12",
    env: Mapping[str, str] | None = None,
    workspace: str = "/workspace",
):
    """
    Build a Modal image with pip deps and local package trees under ``workspace``.

    ``local_dirs`` may be:

    - a sequence of local paths, each copied to ``workspace`` (same basename layout
      as ``add_local_dir(..., remote_path=workspace)`` — put the package root that
      contains your importable module, e.g. ``flashattention1/`` with
      ``flashattention1/flash_attn/...``), or
    - a mapping ``{local_path: remote_path}`` for explicit placement.
    """
    import modal

    image = modal.Image.debian_slim(python_version=python_version)
    if pip_packages:
        image = image.pip_install(*pip_packages)

    merged_env = {"PYTHONPATH": workspace, **(dict(env) if env else {})}
    image = image.env(merged_env)

    if isinstance(local_dirs, Mapping):
        items = [(Path(local), remote) for local, remote in local_dirs.items()]
    else:
        items = [(Path(local), workspace) for local in local_dirs]

    for local, remote in items:
        image = image.add_local_dir(str(local.resolve()), remote_path=remote, copy=True)

    return image
