"""Causal 30/60/180-second feature derivation from native parts."""

from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np

from .cardiac_features import ecg_rr_ms, interval_features, wave_rr_ms
from .feature_catalog import LOOKBACKS, feature_order, schema_sha
from .feature_types import FeatureFrame
from .types import NativePart, SignalSamples
from .wave_features import amplitude, change_rate, eeg_features, slope


def derive_features(parts: list[NativePart] | tuple[NativePart, ...],
                    decision_ms: int) -> tuple[FeatureFrame, ...]:
    eligible = [part for part in parts if part.end_ms <= decision_ms]
    if not eligible:
        return ()
    family = eligible[0].family
    scope = eligible[0].scope
    source_uuid = eligible[0].source_uuid
    if any(part.family != family or part.scope != scope
           or part.source_uuid != source_uuid for part in eligible):
        raise ValueError("feature_scope_mismatch")
    quality = _current_quality(eligible[-1], decision_ms)
    if quality is None:
        return ()
    values: dict[str, float] = {}
    for window in LOOKBACKS:
        selected = [part for part in eligible if part.end_ms > decision_ms - window * 1000]
        try:
            values.update(_family_features(family, selected, decision_ms, window))
        except ValueError:
            continue
    ordered = feature_order(family)
    finite = {name: float(values[name]) for name in ordered
              if name in values and math.isfinite(values[name])}
    required = [name for name in finite if name.endswith("_180") and "motion" not in name]
    if len(required) < 2:
        return ()
    support = (decision_ms - 10_000, decision_ms)
    return (FeatureFrame(scope, decision_ms, family, MappingProxyType(finite),
                         support, quality,
                         schema_sha(family)),)


def _family_features(family: str, parts: list[NativePart], decision: int,
                     window: int) -> dict[str, float]:
    if not parts:
        raise ValueError("feature_window_empty")
    if family == "watch":
        eda, eda_time = _signal(parts, "eda", decision, window)
        ppg, ppg_time = _signal(parts, "ppg_green", decision, window)
        temp, temp_time = _signal(parts, "skin_temp", decision, window)
        ibi = wave_rr_ms(ppg, ppg_time, 100.0, prominence=0.25)
        rmssd, _, _ = interval_features(ibi)
        result = {"hr_median": 60_000 / np.median(ibi), "ibi_rmssd": rmssd,
                  "eda_median": np.median(eda), "eda_slope": slope(eda, eda_time),
                  "ppg_amp": amplitude(ppg), "temp_median": np.median(temp),
                  "temp_change": slope(temp, temp_time)}
    elif family == "h10":
        ecg, ecg_time = _signal(parts, "ecg", decision, window)
        rr = ecg_rr_ms(ecg, ecg_time, 130.0)
        rmssd, sdnn, change = interval_features(rr)
        result = {"hr_median": 60_000 / np.median(rr), "rr_rmssd": rmssd,
                  "rr_sdnn": sdnn, "rr_change": change}
    else:
        result = {}
        for channel in range(1, 5):
            eeg, _ = _signal(parts, f"eeg_{channel}", decision, window)
            theta, alpha, beta, log_power = eeg_features(eeg, 256.0)
            midpoint = len(eeg) // 2
            prior = eeg_features(eeg[:midpoint], 256.0)[3]
            current = eeg_features(eeg[midpoint:], 256.0)[3]
            result.update({f"eeg_{channel}_theta_rel": theta,
                           f"eeg_{channel}_alpha_rel": alpha,
                           f"eeg_{channel}_beta_rel": beta,
                           f"eeg_{channel}_log_power": log_power,
                           f"eeg_{channel}_power_change": change_rate(
                               current, prior, window / 2
                           )})
    motion = _motion(parts, decision, window)
    if motion is not None:
        result["motion_var"] = motion
    return {f"{name}_{window}": float(value) for name, value in result.items()}


def _signal(parts: list[NativePart], name: str, decision: int,
            window: int) -> tuple[np.ndarray, np.ndarray]:
    samples = [part.signals[name] for part in parts if name in part.signals]
    values, times, available, valid = _concat(samples)
    lower = (decision - window * 1000) * 1000
    mask = (valid & (available <= decision) & (times >= lower)
            & (times < decision * 1000))
    values, times = values[mask], times[mask]
    if not len(values):
        raise ValueError("signal_unavailable")
    nominal_fs = samples[0].nominal_fs_hz
    if nominal_fs is not None and len(values) < math.ceil(window * nominal_fs * 0.8):
        raise ValueError("signal_coverage_low")
    period_us = int(1_000_000 / nominal_fs) if nominal_fs else 0
    if ((len(times) > 1 and np.max(np.diff(times)) >= 6_000_000)
            or times[0] - lower >= 6_000_000
            or decision * 1000 - times[-1] - period_us >= 6_000_000):
        raise ValueError("signal_gap_exceeded")
    return values.astype(np.float64), times


def _concat(
    samples: list[SignalSamples],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (np.concatenate([sample.values for sample in samples]),
            np.concatenate([sample.time_us for sample in samples]),
            np.concatenate([sample.available_ms for sample in samples]),
            np.concatenate([sample.valid for sample in samples]))


def _motion(parts: list[NativePart], decision: int, window: int) -> float | None:
    source = parts[0].auxiliary if parts[0].family == "muse" else parts[0].signals
    names = ("acc_x", "acc_y", "acc_z")
    if not all(name in source for name in names):
        return None
    axes = []
    for name in names:
        samples = [part.auxiliary[name] if part.family == "muse" else part.signals[name]
                   for part in parts]
        values, times, available, valid = _concat(samples)
        lower = (decision - window * 1000) * 1000
        mask = (valid & (available <= decision) & (times >= lower)
                & (times < decision * 1000))
        selected = values[mask]
        expected = window * 25
        selected_times = times[mask]
        if (len(selected) < math.ceil(expected * 0.8)
                or (len(selected_times) > 1
                    and np.max(np.diff(selected_times)) >= 6_000_000)):
            return None
        axes.append(selected)
    count = min(map(len, axes))
    return float(np.var(np.sqrt(sum(axis[:count] ** 2 for axis in axes)))) if count else 0.0


def _current_quality(part: NativePart, decision: int) -> float | None:
    required = {
        "watch": ("eda", "ppg_green", "skin_temp"),
        "h10": ("ecg",),
        "muse": ("eeg_1", "eeg_2", "eeg_3", "eeg_4"),
    }[part.family]
    qualities = []
    for name in required:
        sample = (part.auxiliary.get(name)
                  if part.family == "muse" and name in part.auxiliary
                  else part.signals.get(name))
        if sample is None:
            return None
        expected = int((sample.nominal_fs_hz or 1.0) * 10)
        mask = (sample.valid & (sample.available_ms <= decision)
                & (sample.time_us >= (decision - 10_000) * 1000)
                & (sample.time_us < decision * 1000))
        times = sample.time_us[mask]
        quality = len(times) / max(expected, 1)
        start_us = (decision - 10_000) * 1000
        period_us = int(1_000_000 / (sample.nominal_fs_hz or 1.0))
        if (quality < 0.8 or (len(times) > 1 and np.max(np.diff(times)) >= 6_000_000)
                or not len(times) or times[0] - start_us >= 6_000_000
                or decision * 1000 - times[-1] - period_us >= 6_000_000):
            return None
        qualities.append(quality)
    if part.family == "muse" and "contact" in part.auxiliary:
        contact = part.auxiliary["contact"]
        mask = contact.valid & (contact.available_ms <= decision) & (contact.values == 1.0)
        quality = float(np.sum(mask)) / 10.0
        if quality < 0.8:
            return None
        qualities.append(quality)
    return float(min(qualities))
