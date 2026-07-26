from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from multisensor_synth.config.loader import load_config

ROOT = Path(__file__).parents[2]
QUICK_CONFIG = ROOT / "configs" / "quick.yaml"


def _write_changed_config(
    tmp_path: Path, change: Callable[[dict[str, object]], None]
) -> Path:
    loaded = load_config(QUICK_CONFIG)
    payload = loaded.config.model_dump(mode="json", by_alias=True)
    change(payload)
    path = tmp_path / "changed.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_quick_config_loads_with_fixed_contracts() -> None:
    loaded = load_config(QUICK_CONFIG)

    assert loaded.config.run.participant_count == 3
    assert loaded.config.run.duration_sec == 21_600
    assert loaded.config.run.canonical_rate_hz == 1
    assert loaded.config.devices.muse_s.eeg.sample_rate_hz == 256
    assert loaded.config.devices.polar_h10.ecg.sample_rate_hz == 130
    assert loaded.config.devices.galaxy_watch8.eda.sample_rate_hz == 1


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    path = _write_changed_config(tmp_path, lambda payload: payload.update({"surprise": True}))

    with pytest.raises(ValidationError, match="surprise"):
        load_config(path)


def test_invalid_device_rate_is_rejected(tmp_path: Path) -> None:
    def change(payload: dict[str, object]) -> None:
        payload["devices"]["muse_s"]["eeg"]["sample_rate_hz"] = 128  # type: ignore[index]

    path = _write_changed_config(tmp_path, change)

    with pytest.raises(ValidationError, match="Muse S EEG"):
        load_config(path)


def test_probability_sum_is_rejected(tmp_path: Path) -> None:
    def change(payload: dict[str, object]) -> None:
        payload["population"]["response_archetypes"]["eeg_first"] = 0.9  # type: ignore[index]

    path = _write_changed_config(tmp_path, change)

    with pytest.raises(ValidationError, match="response archetype"):
        load_config(path)


def test_temperature_units_alias_normalizes_to_unit() -> None:
    loaded = load_config(QUICK_CONFIG)
    resolved = loaded.config.model_dump(mode="json")
    temperature = resolved["devices"]["galaxy_watch8"]["skin_temperature"]

    assert temperature["unit"] == "C"
    assert "units" not in temperature


def test_run_start_must_be_utc(tmp_path: Path) -> None:
    def change(payload: dict[str, object]) -> None:
        payload["run"]["start_time_utc"] = "2026-01-01T09:00:00+09:00"  # type: ignore[index]

    path = _write_changed_config(tmp_path, change)

    with pytest.raises(ValidationError, match="UTC"):
        load_config(path)


def test_negative_phase_duration_is_rejected(tmp_path: Path) -> None:
    def change(payload: dict[str, object]) -> None:
        payload["events"]["phases"]["pre_early_sec"] = {  # type: ignore[index]
            "min": -1,
            "max": 10,
        }

    path = _write_changed_config(tmp_path, change)

    with pytest.raises(ValidationError, match="phase durations must be non-negative"):
        load_config(path)


def test_polar_allowed_rate_contract_cannot_be_narrowed(tmp_path: Path) -> None:
    def change(payload: dict[str, object]) -> None:
        payload["devices"]["polar_h10"]["accelerometer"][  # type: ignore[index]
            "allowed_sample_rates_hz"
        ] = [25, 50, 100]

    path = _write_changed_config(tmp_path, change)

    with pytest.raises(ValidationError, match="allowed ACC rates"):
        load_config(path)


def test_all_modality_delay_keys_are_required(tmp_path: Path) -> None:
    def change(payload: dict[str, object]) -> None:
        del payload["events"]["modality_delays_sec"]["motor"]  # type: ignore[index]

    path = _write_changed_config(tmp_path, change)

    with pytest.raises(ValidationError, match="modality delay keys"):
        load_config(path)


def test_all_eeg_frequency_bands_are_required(tmp_path: Path) -> None:
    def change(payload: dict[str, object]) -> None:
        del payload["features"]["eeg"]["frequency_bands_hz"]["gamma"]  # type: ignore[index]

    path = _write_changed_config(tmp_path, change)

    with pytest.raises(ValidationError, match="EEG frequency band keys"):
        load_config(path)


def test_all_device_clock_keys_are_required(tmp_path: Path) -> None:
    def change(payload: dict[str, object]) -> None:
        del payload["artifacts"]["clocks"]["drift_ppm"]["muse_s"]  # type: ignore[index]

    path = _write_changed_config(tmp_path, change)

    with pytest.raises(ValidationError, match="clock drift device keys"):
        load_config(path)


def test_invalid_device_unit_is_rejected(tmp_path: Path) -> None:
    def change(payload: dict[str, object]) -> None:
        payload["devices"]["polar_h10"]["ecg"]["unit"] = "mV"  # type: ignore[index]

    path = _write_changed_config(tmp_path, change)

    with pytest.raises(ValidationError, match="Polar H10 ECG"):
        load_config(path)


def test_invalid_split_fraction_is_rejected(tmp_path: Path) -> None:
    def change(payload: dict[str, object]) -> None:
        payload["splits"]["test_fraction"] = 0.5  # type: ignore[index]

    path = _write_changed_config(tmp_path, change)

    with pytest.raises(ValidationError, match="split fractions"):
        load_config(path)


def test_watch_on_demand_streams_cannot_both_be_enabled(tmp_path: Path) -> None:
    def change(payload: dict[str, object]) -> None:
        on_demand = payload["devices"]["galaxy_watch8"]["on_demand"]  # type: ignore[index]
        on_demand["ecg_500hz"]["enabled"] = True
        on_demand["ppg_100hz"]["enabled"] = True

    path = _write_changed_config(tmp_path, change)

    with pytest.raises(ValidationError, match="cannot both be enabled"):
        load_config(path)


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    def change(payload: dict[str, object]) -> None:
        payload["events"]["phases"]["hidden_phase"] = {  # type: ignore[index]
            "min": 1,
            "max": 2,
        }

    path = _write_changed_config(tmp_path, change)

    with pytest.raises(ValidationError, match="hidden_phase"):
        load_config(path)
