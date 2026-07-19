"""Serialize client-side uploads / deserialize sandbox downloads."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

UploadValue = np.ndarray | Mapping[str, Any] | Sequence[Any] | Path | bytes | str


def materialize_upload(value: UploadValue, dest: Path) -> None:
    """Write ``value`` to ``dest`` for ``Sandbox.filesystem.copy_from_local``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, np.ndarray):
        np.save(dest, value, allow_pickle=False)
        return
    if isinstance(value, Path) or (isinstance(value, str) and Path(value).is_file()):
        shutil.copyfile(value, dest)
        return
    if isinstance(value, bytes):
        dest.write_bytes(value)
        return
    if isinstance(value, (dict, list, tuple)):
        dest.write_text(json.dumps(value), encoding="utf-8")
        return
    if isinstance(value, str):
        dest.write_text(value, encoding="utf-8")
        return
    raise TypeError(
        f"Unsupported upload type {type(value)!r}; "
        "expected ndarray, dict/list, Path, bytes, or str"
    )


def load_npy(data: bytes) -> np.ndarray:
    """Decode a ``.npy`` payload downloaded from the sandbox."""
    return np.load(io.BytesIO(data), allow_pickle=False)
