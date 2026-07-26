from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from multisensor_ml.knime import export_knime_artifacts
from multisensor_ml.materialize import materialize_synthetic
from multisensor_ml.pipeline import (
    PreparedSeries,
    evaluate_prepared_bundle,
    prepare_series,
    train_prepared,
)
from multisensor_ml.settings import load_goal15_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multisensor-ml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize-synthetic")
    materialize.add_argument("--config", type=Path, required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--series", required=True)
    prepare.add_argument("--project-root", type=Path, default=Path.cwd())

    train = subparsers.add_parser("train")
    train.add_argument("--experiment", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument(
        "--role",
        choices=("validation", "locked_test"),
        required=True,
    )
    evaluate.add_argument("--audit-reason")
    evaluate.add_argument("--project-root", type=Path, default=Path.cwd())

    export = subparsers.add_parser("export-knime")
    export.add_argument("--experiment", required=True)
    export.add_argument("--project-root", type=Path, default=Path.cwd())

    run_all = subparsers.add_parser("run-all")
    run_all.add_argument("--config", type=Path, required=True)
    return parser


def _prepared_from_root(root: Path) -> PreparedSeries:
    return PreparedSeries(
        root=root,
        manifest_json=root / "manifest.json",
        global_baseline_json=root / "global_baseline.json",
        personal_baseline_parquet=root / "personal_baseline.parquet",
    )


def _emit(**payload: object) -> None:
    print(json.dumps(payload, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "materialize-synthetic":
        config = load_goal15_config(args.config)
        registered = materialize_synthetic(config)
        _emit(series_id=registered.series_id, registry=str(registered.registry_dir))
        return 0

    if args.command == "prepare":
        project = args.project_root.resolve()
        prepared = prepare_series(
            project / "data" / "registry" / args.series,
            project / "data" / "prepared" / args.series,
        )
        _emit(series_id=args.series, prepared=str(prepared.root))
        return 0

    if args.command == "train":
        config = load_goal15_config(args.experiment)
        bundle = train_prepared(
            config.data_root / "prepared" / config.series_id,
            config.artifact_root / config.experiment_id,
            random_state=config.random_state,
            locked_test_audit_reason=config.locked_test_audit_reason,
            uv_lock=Path(__file__).parents[2] / "uv.lock",
        )
        _emit(experiment_id=config.experiment_id, bundle=str(bundle))
        return 0

    if args.command == "evaluate":
        project = args.project_root.resolve()
        dataset = Path(args.dataset)
        prepared_root = (
            dataset.resolve()
            if dataset.exists()
            else project / "data" / "prepared" / args.dataset
        )
        report = evaluate_prepared_bundle(
            args.bundle.resolve(),
            prepared_root,
            role=args.role,
            audit_reason=args.audit_reason,
        )
        _emit(role=args.role, report=str(report))
        return 0

    if args.command == "export-knime":
        project = args.project_root.resolve()
        bundle = project / "artifacts" / args.experiment
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        lineage = cast(dict[str, object], manifest["lineage"])
        prepared_root = Path(str(lineage["prepared_root"]))
        export = export_knime_artifacts(
            bundle,
            prepared_root,
            project / "knime" / "exports" / args.experiment,
        )
        _emit(experiment_id=args.experiment, knime_export=str(export))
        return 0

    if args.command == "run-all":
        config = load_goal15_config(args.config)
        registered = materialize_synthetic(config)
        prepared_root = config.data_root / "prepared" / config.series_id
        prepared = (
            _prepared_from_root(prepared_root)
            if prepared_root.exists()
            else prepare_series(registered.registry_dir, prepared_root)
        )
        bundle_root = config.artifact_root / config.experiment_id
        bundle = (
            bundle_root
            if bundle_root.exists()
            else train_prepared(
                prepared.root,
                bundle_root,
                random_state=config.random_state,
                locked_test_audit_reason=config.locked_test_audit_reason,
                uv_lock=Path(__file__).parents[2] / "uv.lock",
            )
        )
        export_root = Path(__file__).parents[2] / "knime" / "exports" / config.experiment_id
        export = (
            export_root
            if export_root.exists()
            else export_knime_artifacts(bundle, prepared.root, export_root)
        )
        _emit(
            series_id=config.series_id,
            experiment_id=config.experiment_id,
            registry=str(registered.registry_dir),
            prepared=str(prepared.root),
            bundle=str(bundle),
            knime_export=str(export),
            status="oracle/sanity",
            real_data_status="NOT VERIFIED",
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")
