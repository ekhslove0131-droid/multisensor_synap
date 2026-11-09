"""Strict JSON numeric conversion that excludes booleans and non-finite values."""

from __future__ import annotations

import math
from typing import cast


def strict_float(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("strict_float_invalid")
    number = cast(int | float, value)
    if not math.isfinite(float(number)):
        raise ValueError("strict_float_invalid")
    return float(number)
