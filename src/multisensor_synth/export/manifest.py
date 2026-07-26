from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from multisensor_synth import __version__
from multisensor_synth.config.loader import LoadedConfig
from multisensor_synth.domain.seeds import SeedTree

DETERMINISTIC_MANIFEST_KEYS = (
    "schema_version",
    "generator_version",
    "run_id",
    "profile",
    "seed",
    "config_sha256",
    "run_contract",
    "deterministic",
    "synthetic",
    "non_diagnostic",
    "not_for_clinical_use",
    "implemented_goal",
    "implemented_layers",
    "deferred_layers",
    "child_seed_namespaces",
    "tables",
)


def deterministic_manifest_subset(manifest: dict[str, object]) -> dict[str, object]:
    return {
        key: manifest[key]
        for key in DETERMINISTIC_MANIFEST_KEYS
        if key in manifest
    }


def _git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _dependency_versions() -> dict[str, str]:
    packages = ("numpy", "pyarrow", "pydantic", "PyYAML")
    result: dict[str, str] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def build_manifest(
    loaded: LoadedConfig,
    run_id: str,
    seeds: SeedTree,
    tables: dict[str, dict[str, object]],
    project_root: Path,
) -> dict[str, object]:
    return {
        "schema_version": loaded.config.schema_version,
        "generator_version": __version__,
        "run_id": run_id,
        "profile": loaded.config.profile,
        "seed": loaded.config.run.seed,
        "config_sha256": loaded.config_sha256,
        "run_contract": {
            "participant_count": loaded.config.run.participant_count,
            "duration_sec": loaded.config.run.duration_sec,
            "canonical_rate_hz": loaded.config.run.canonical_rate_hz,
            "minimum_gap_between_target_sec": (
                loaded.config.events.minimum_gap_between_target_sec
            ),
        },
        "git_commit": _git_commit(project_root),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "deterministic": loaded.config.run.deterministic,
        "synthetic": True,
        "non_diagnostic": True,
        "not_for_clinical_use": True,
        "implemented_goal": 1,
        "implemented_layers": ["truth"],
        "deferred_layers": ["observed", "model_ready"],
        "child_seed_namespaces": seeds.used_namespaces,
        "tables": tables,
        "dependency_versions": _dependency_versions(),
    }


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
