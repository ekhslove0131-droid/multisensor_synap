from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml
from multisensor_synth.config.loader import load_config
from multisensor_synth.orchestration.pipeline import generate_truth_run, stable_run_id

from multisensor_ml.registry import RegisteredSeries, register_series
from multisensor_ml.settings import Goal15Config


def _deep_merge(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    result = dict(left)
    for key, value in right.items():
        previous = result.get(key)
        if isinstance(previous, dict) and isinstance(value, dict):
            result[key] = _deep_merge(
                cast(dict[str, object], previous),
                cast(dict[str, object], value),
            )
        else:
            result[key] = value
    return result


def write_derived_generator_config(
    base_config: Path,
    destination: Path,
    *,
    seed: int,
    overrides: dict[str, object] | None = None,
) -> Path:
    """Create an auditable generator config that inherits the read-only source."""

    payload: dict[str, object] = {
        "extends": str(base_config.resolve()),
        "run": {"seed": seed},
    }
    if overrides:
        payload = _deep_merge(payload, overrides)
        run = cast(dict[str, object], payload.setdefault("run", {}))
        run["seed"] = seed
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return destination


def materialize_synthetic(config: Goal15Config) -> RegisteredSeries:
    """Generate/reuse configured truth runs, then register read-only lineage references."""

    series_raw = config.data_root / "raw" / config.series_id
    config_dir = series_raw / "generator_configs"
    run_root = series_raw / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    run_dirs: list[Path] = []
    for seed in config.seeds:
        derived = write_derived_generator_config(
            config.generator_config,
            config_dir / f"seed-{seed}.yaml",
            seed=seed,
            overrides=config.generator_overrides,
        )
        loaded = load_config(derived)
        expected = run_root / stable_run_id(loaded)
        if expected.exists():
            run_dirs.append(expected)
            continue
        generated = generate_truth_run(derived, output_root=run_root)
        run_dirs.append(generated.run_dir)

    registry_root = config.data_root / "registry"
    existing = registry_root / config.series_id
    if existing.exists():
        return RegisteredSeries(
            series_id=config.series_id,
            registry_dir=existing,
            records_jsonl=existing / "datasets.jsonl",
            split_parquet=existing / "splits.parquet",
            manifest_json=existing / "manifest.json",
        )
    return register_series(config.series_id, run_dirs, registry_root)
