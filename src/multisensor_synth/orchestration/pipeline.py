from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from multisensor_synth import __version__
from multisensor_synth.config.loader import LoadedConfig, load_config
from multisensor_synth.domain.seeds import SeedTree
from multisensor_synth.events.scheduler import schedule_events
from multisensor_synth.export.manifest import build_manifest, write_manifest
from multisensor_synth.export.parquet import (
    DAILY_CONTEXT_SCHEMA,
    EVENT_SCHEMA,
    PARTICIPANT_SCHEMA,
    LatentParquetWriter,
    table_summary,
    write_records,
)
from multisensor_synth.latent.daily_context import generate_daily_contexts
from multisensor_synth.latent.state_process import generate_person_timeline
from multisensor_synth.population.generator import generate_participants
from multisensor_synth.validation.invariants import validate_truth_run
from multisensor_synth.validation.report import (
    ValidationReport,
    write_validation_report,
)


@dataclass(frozen=True, slots=True)
class GeneratedRun:
    run_id: str
    run_dir: Path
    report: ValidationReport
    manifest: dict[str, object]


def stable_run_id(loaded: LoadedConfig) -> str:
    material = (
        f"{loaded.config.profile}:{loaded.config.run.seed}:"
        f"{loaded.config_sha256}:{__version__}"
    )
    suffix = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    return f"{loaded.config.profile}-{suffix}"


def _compression(value: str) -> str | None:
    return None if value == "none" else value


def generate_truth_run(
    config_path: str | Path, output_root: str | Path | None = None
) -> GeneratedRun:
    loaded = load_config(config_path)
    config = loaded.config
    run_id = stable_run_id(loaded)
    root = Path(output_root) if output_root is not None else config.storage.output_root
    run_dir = root.resolve() / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    truth_dir = run_dir / "truth"
    snapshot_dir = run_dir / "config_snapshot"
    reports_dir = run_dir / "reports"
    for directory in (truth_dir, snapshot_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=False)

    (snapshot_dir / "source.yaml").write_text(loaded.source_text, encoding="utf-8")
    (snapshot_dir / "resolved.yaml").write_text(
        yaml.safe_dump(loaded.resolved_data, sort_keys=False),
        encoding="utf-8",
    )

    seeds = SeedTree(config.run.seed)
    participants = generate_participants(config, seeds, run_id)
    contexts = generate_daily_contexts(config, participants, seeds, run_id)
    events = schedule_events(config, participants, seeds, run_id)
    compression = _compression(config.storage.parquet_compression)
    write_records(
        truth_dir / "participants.parquet",
        participants,
        PARTICIPANT_SCHEMA,
        compression,
    )
    write_records(
        truth_dir / "daily_context.parquet",
        contexts,
        DAILY_CONTEXT_SCHEMA,
        compression,
    )
    write_records(
        truth_dir / "events.parquet",
        events,
        EVENT_SCHEMA,
        compression,
    )
    with LatentParquetWriter(
        truth_dir / "latent_timeline.parquet", compression
    ) as writer:
        for participant in participants:
            timeline = generate_person_timeline(
                config, participant, contexts, events, seeds, run_id
            )
            writer.write(timeline, config.run.chunk_duration_sec)

    table_paths = (
        "truth/participants.parquet",
        "truth/daily_context.parquet",
        "truth/latent_timeline.parquet",
        "truth/events.parquet",
    )
    tables = {relative: table_summary(run_dir / relative) for relative in table_paths}
    tables["truth/latent_timeline.parquet"]["rows_per_person"] = config.run.duration_sec
    manifest = build_manifest(
        loaded,
        run_id,
        seeds,
        tables,
        Path(__file__).parents[3],
    )
    write_manifest(run_dir / "manifest.json", manifest)
    report = validate_truth_run(run_dir)
    write_validation_report(reports_dir / "truth_validation_report.json", report)
    return GeneratedRun(
        run_id=run_id,
        run_dir=run_dir,
        report=report,
        manifest=manifest,
    )
