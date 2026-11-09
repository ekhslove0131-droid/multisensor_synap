"""Stable digest over the shared numeric runtime package source."""

from __future__ import annotations

import hashlib
from pathlib import Path


def runtime_kernel_sha256() -> str:
    root = Path(__file__).resolve().parent
    material = b"".join(
        path.name.encode() + b"\0" + path.read_bytes()
        for path in sorted(root.glob("*.py"))
    )
    return hashlib.sha256(material).hexdigest()
