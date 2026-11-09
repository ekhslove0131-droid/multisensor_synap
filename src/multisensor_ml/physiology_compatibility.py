"""Pinned descriptive-schema consumer and independent normalized-signal math.

No raw admission, source timestamp verification, fitting, or cohort creation.
Cloud must verify native sampling/contact/clock before using the math helpers.
"""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

SCHEMA_HASHES = {
    "watch": "4e949a5033e36b9c2f0ec3d348c52088717a0904dc774a923941f468061bc6fb",
    "h10": "055bfcee108dd6e059ddd930eb682ef43760c92bc62ef68cb004953a2ae92ae2",
    "muse": "97139686fab24c88ed844ba2d69ea81384b76711a7217a954c994e4eaa334ce9",
}


def load_schema(document: Mapping[str, Any], source: str) -> dict[str, Any]:
    encoded = json.dumps(document, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    if source not in SCHEMA_HASHES or hashlib.sha256(encoded).hexdigest() != SCHEMA_HASHES[source]:
        raise ValueError("schema hash mismatch")
    return dict(json.loads(encoded))


def consume_descriptors(
    schema: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[float, ...] | None:
    """Validate raw-free descriptor projection; a returned vector is not training ready.

    Projection contains code/unit/value/readiness/reason only. Optional absent
    descriptors never expand the ordered core. Revalidate schema on every call.
    """
    document = load_schema(schema, schema.get("source_kind", ""))
    specs = {d["code"]: d for d in document["descriptors"]}
    values: dict[str, float | None] = {}
    for row in rows:
        if set(row) != {"code", "unit", "value", "readiness", "reason"}:
            raise ValueError("descriptor fields mismatch")
        code = row["code"]
        if not isinstance(code, str) or code not in specs or code in values:
            raise ValueError("unknown or duplicate descriptor")
        if row["unit"] != specs[code]["unit"]:
            raise ValueError("descriptor unit mismatch")
        value = row["value"]
        if row["readiness"] == "AVAILABLE":
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or row["reason"] is not None):
                raise ValueError("available descriptor invalid")
            domain = specs[code]["domain"]
            if not {
                "POSITIVE": value > 0,
                "NONNEGATIVE": value >= 0,
                "UNIT_INTERVAL": 0 <= value <= 1,
                "SIGNED_FINITE": True,
            }.get(domain, False):
                raise ValueError("descriptor domain violation")
            values[code] = float(value)
        elif row["readiness"] == "INSUFFICIENT":
            if value is not None or not isinstance(row["reason"], str) or not row["reason"]:
                raise ValueError("insufficient descriptor invalid")
            values[code] = None
        else:
            raise ValueError("unknown descriptor readiness")
    core = [values.get(name) for name in document["core_features"]]
    return None if any(v is None for v in core) else tuple(float(v) for v in core if v is not None)


def watch_core(
    hr: Sequence[float], eda: Sequence[float], eda_times_seconds: Sequence[float]
) -> tuple[float, float, float]:
    """Math-only parity helper, after source support/quality validation."""
    h, e, t = (np.asarray(v, dtype=float) for v in (hr, eda, eda_times_seconds))
    if (any(v.ndim != 1 or not len(v) or not np.isfinite(v).all() for v in (h, e, t))
            or len(e) != len(t) or len(t) < 2 or np.any(np.diff(t) <= 0)):
        raise ValueError("invalid normalized watch values")
    centered = t - t.mean()
    slope = float(np.dot(centered, e - e.mean()) / np.dot(centered, centered))
    return float(np.median(h)), float(np.median(e)), slope


def rr_rmssd(
    values: Sequence[float], valid: Sequence[bool], segments: Sequence[int],
    sequence: Sequence[int],
) -> float | None:
    n = len(values)
    if (any(len(v) != n for v in (valid, segments, sequence))
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or any(type(v) is not bool for v in valid)
            or any(type(v) is not int or v < 0 for v in (*segments, *sequence))
            or any(sequence[i] <= sequence[i-1] or segments[i] < segments[i-1]
                   for i in range(1, n))):
        raise ValueError("invalid RR sequence evidence")
    differences = [(values[i] - values[i-1])**2 for i in range(1, n)
                   if valid[i] and valid[i-1] and values[i] > 0 and values[i-1] > 0
                   and segments[i] == segments[i-1] and sequence[i] == sequence[i-1] + 1]
    return math.sqrt(sum(differences)/len(differences)) if differences else None


def eeg_descriptors(values: Sequence[float], fs: float) -> tuple[float, ...]:
    """Independent Welch calculation on an already verified native 10-second signal."""
    x = np.asarray(values, dtype=float)
    if (type(fs) not in (int, float) or not math.isfinite(fs) or not 100 <= fs <= 10000
            or (10*fs) % 1 or (4*fs) % 1 or int(4*fs) % 2
            or x.ndim != 1 or len(x) != int(10*fs) or not np.isfinite(x).all()
            or np.var(x) == 0):
        raise ValueError("insufficient native EEG")
    size = int(4*fs)
    hann = 0.5 - 0.5*np.cos(2*np.pi*np.arange(size)/size)
    blocks = np.lib.stride_tricks.sliding_window_view(x, size)[::size//2]
    transformed = np.fft.rfft((blocks - blocks.mean(axis=1, keepdims=True))*hann, axis=1)
    psd = (np.abs(transformed)**2).mean(axis=0) / (fs*np.sum(hann**2))
    psd[1:-1] *= 2
    frequency = np.fft.rfftfreq(size, 1/fs)
    total = psd[(frequency >= 1) & (frequency < 45)]
    energy = float(total.sum())
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError("no EEG band energy")
    # Common df cancels in relative band power and normalized bin probabilities.
    bands = tuple(float(psd[(frequency >= lo) & (frequency < hi)].sum()/energy)
                  for lo, hi in ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45)))
    p = total[total > 0]/energy
    entropy = float(-np.sum(p*np.log(p))/np.log(len(total)))
    return (*bands, entropy)
