"""Small parsing and digest helpers for the wire16 closure receipt."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from synapbridge_beta_runtime.wire_contract import WIRE16


def wire_pins(path: Path) -> dict[str, str]:
    pattern = re.compile(r"^\+?\| ([^|]+) \| ([0-9a-f]{64}) \|$")
    values = {}
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            values[match.group(1).strip()] = match.group(2)
    expected = {
        (f"schemas/synthetic_beta_v1/{name}"
         if kind == "synthetic_beta_v1"
         else f"services/inference/tests/synthetic_beta/golden/{name}"): digest
        for kind, name, digest in WIRE16
    }
    if values != expected:
        raise ValueError("wire16_pin_contract_invalid")
    return values


def wire_paths(refs: list[dict[str, str]]) -> dict[str, str]:
    return {
        (f"schemas/synthetic_beta_v1/{ref['name']}"
         if ref["kind"] == "synthetic_beta_v1"
         else f"services/inference/tests/synthetic_beta/golden/{ref['name']}"):
        ref["sha256"]
        for ref in refs
    }


def wheel_kernel_sha(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name for name in archive.namelist()
            if name.startswith("synapbridge_beta_runtime/") and name.endswith(".py")
        )
        material = b"".join(
            Path(name).name.encode() + b"\0" + archive.read(name) for name in members
        )
    return hashlib.sha256(material).hexdigest()


def local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc:
        raise ValueError("local_ref_invalid")
    return Path(unquote(parsed.path))
