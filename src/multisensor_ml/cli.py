from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import pandas as pd

from multisensor_ml.factory import run_synthetic_factory
from multisensor_ml.kaggle_model_contracts import load_kaggle_model_package_config
from multisensor_ml.kaggle_model_package import (
    build_kaggle_model_package,
    verify_kaggle_model_package,
)
from multisensor_ml.kaggle_reproduce import (
    compare_expected,
    load_hierarchical_package,
    predict_hierarchical,
)
from multisensor_ml.knime import export_knime_artifacts
from multisensor_ml.materialize import materialize_synthetic
from multisensor_ml.model_registry import ModelRegistry, import_oracle_bundle
from multisensor_ml.neon_adapter import (
    derive_watch_features,
    fit_personal_baseline_adapter,
    personal_adapter_manifest,
    transform_with_personal_adapter,
)
from multisensor_ml.phase3_pipeline import (
    prepare_phase3_source,
    run_phase3_from_config,
)
from multisensor_ml.pipeline import (
    PreparedSeries,
    evaluate_prepared_bundle,
    prepare_series,
    train_prepared,
)
from multisensor_ml.receipts import validate_stage_receipt
from multisensor_ml.registry_workflow import (
    RegistryRunResult,
    export_registry_knime_tables,
    run_registry_all,
    run_registry_stage,
)
from multisensor_ml.sensor_availability import (
    AVAILABILITY_PROFILES,
    build_sensor_availability_package,
    verify_sensor_availability_package,
)
from multisensor_ml.settings import (
    load_factory_config,
    load_goal15_config,
    load_phase3_config,
    load_training_registry_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multisensor-ml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize-synthetic")
    materialize.add_argument("--config", type=Path, required=True)

    factory = subparsers.add_parser("factory")
    factory_commands = factory.add_subparsers(dest="factory_command", required=True)
    factory_run = factory_commands.add_parser("run")
    factory_run.add_argument("--config", type=Path, required=True)
    factory_run.add_argument("--output-receipt", type=Path, required=True)
    factory_validate = factory_commands.add_parser("validate")
    factory_validate.add_argument("--receipt", type=Path, required=True)

    registry = subparsers.add_parser("registry")
    registry_commands = registry.add_subparsers(dest="registry_command", required=True)
    registry_init = registry_commands.add_parser("init")
    registry_init.add_argument("--config", type=Path, required=True)
    registry_import = registry_commands.add_parser("import-oracle")
    registry_import.add_argument("--experiment", required=True)
    registry_import.add_argument("--project-root", type=Path, default=Path.cwd())
    registry_import.add_argument(
        "--registry-path",
        type=Path,
        default=Path("data/model_registry/goal15.sqlite"),
    )
    registry_stage = registry_commands.add_parser("run-stage")
    registry_stage.add_argument(
        "--stage",
        choices=(
            "labels",
            "baseline",
            "types",
            "stage-model",
            "behavior-model",
            "evaluate",
            "predict",
            "route-ko",
        ),
        required=True,
    )
    registry_stage.add_argument("--run-id", required=True)
    registry_stage.add_argument("--input-receipt", type=Path, required=True)
    registry_stage.add_argument("--output-receipt", type=Path, required=True)
    registry_stage.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training_registry.yaml"),
    )
    registry_run_all = registry_commands.add_parser("run-all")
    registry_run_all.add_argument("--config", type=Path, required=True)
    registry_compare = registry_commands.add_parser("compare")
    registry_compare.add_argument("--target", required=True)
    registry_compare.add_argument(
        "--registry-path",
        type=Path,
        default=Path("data/model_registry/goal15.sqlite"),
    )
    registry_promote = registry_commands.add_parser("promote")
    registry_promote.add_argument("--release", required=True)
    registry_promote.add_argument("--audit-reason", required=True)
    registry_promote.add_argument(
        "--registry-path",
        type=Path,
        default=Path("data/model_registry/goal15.sqlite"),
    )
    registry_predict = registry_commands.add_parser("predict")
    registry_predict.add_argument("--release", required=True)
    registry_predict.add_argument("--dataset", required=True)
    registry_predict.add_argument(
        "--registry-path",
        type=Path,
        default=Path("data/model_registry/goal15.sqlite"),
    )
    registry_export = registry_commands.add_parser("export-knime")
    registry_export.add_argument("--run-id", required=True)
    registry_export.add_argument("--project-root", type=Path, default=Path.cwd())
    registry_export.add_argument(
        "--registry-path",
        type=Path,
        default=Path("data/model_registry/goal15.sqlite"),
    )

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

    onnx = subparsers.add_parser("onnx")
    onnx_commands = onnx.add_subparsers(dest="onnx_command", required=True)
    onnx_tune = onnx_commands.add_parser("tune")
    onnx_tune.add_argument("--project-root", type=Path, required=True)
    onnx_tune.add_argument("--series", default="mvp3-oracle-v1")
    onnx_tune.add_argument(
        "--output",
        type=Path,
        default=Path("services/onnx_api/models/goal15-final-v1"),
    )
    onnx_tune.add_argument("--trials", type=int, default=12)
    onnx_tune.add_argument("--max-train-rows", type=int, default=600_000)
    onnx_tune.add_argument("--max-validation-rows", type=int, default=120_000)
    onnx_tune.add_argument("--seed", type=int, default=20260725)

    kaggle_model = subparsers.add_parser("kaggle-model")
    kaggle_model_commands = kaggle_model.add_subparsers(
        dest="kaggle_model_command", required=True
    )
    kaggle_package = kaggle_model_commands.add_parser("package")
    kaggle_package.add_argument("--config", type=Path, required=True)
    kaggle_package.add_argument("--wheel", type=Path, required=True)
    kaggle_package.add_argument("--source-project-root", type=Path)
    kaggle_verify = kaggle_model_commands.add_parser("verify")
    kaggle_verify.add_argument("--package", type=Path, required=True)
    kaggle_reproduce = kaggle_model_commands.add_parser("reproduce")
    kaggle_reproduce.add_argument("--model-root", type=Path, required=True)
    kaggle_reproduce.add_argument("--input", type=Path, required=True)
    kaggle_reproduce.add_argument("--output", type=Path, required=True)
    kaggle_reproduce.add_argument("--expected", type=Path)

    availability_model = subparsers.add_parser("availability-model")
    availability_commands = availability_model.add_subparsers(
        dest="availability_model_command", required=True
    )
    availability_package = availability_commands.add_parser("package")
    availability_package.add_argument("--project-root", type=Path, default=Path.cwd())
    availability_package.add_argument("--series", required=True)
    availability_package.add_argument("--output", type=Path, required=True)
    availability_package.add_argument("--wheel", type=Path)
    availability_package.add_argument("--max-rows-per-person", type=int, default=20_000)
    availability_verify = availability_commands.add_parser("verify")
    availability_verify.add_argument("--package", type=Path, required=True)

    neon_adapter = subparsers.add_parser("neon-adapter")
    neon_adapter_commands = neon_adapter.add_subparsers(
        dest="neon_adapter_command", required=True
    )
    neon_derive = neon_adapter_commands.add_parser("derive")
    neon_derive.add_argument("--input", type=Path, required=True)
    neon_derive.add_argument("--output", type=Path, required=True)
    neon_derive.add_argument("--clock-offset-ms", type=float)
    neon_derive.add_argument("--physiological-lag-ms", type=float)
    neon_fit = neon_adapter_commands.add_parser("fit")
    neon_fit.add_argument("--input", type=Path, required=True)
    neon_fit.add_argument("--output", type=Path, required=True)
    neon_fit.add_argument("--person-id", required=True)
    neon_fit.add_argument("--model-version", default="hierarchical-v5")
    neon_fit.add_argument("--warmup-seconds", type=int, default=1800)
    neon_fit.add_argument("--weight-cap", type=float, default=0.75)

    phase3 = subparsers.add_parser("phase3")
    phase3_commands = phase3.add_subparsers(dest="phase3_command", required=True)
    for phase3_command in ("prepare", "train-validate", "report-input"):
        command_parser = phase3_commands.add_parser(phase3_command)
        command_parser.add_argument("--config", type=Path, required=True)
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
    if args.command == "onnx":
        if args.onnx_command != "tune":
            raise AssertionError(f"unhandled ONNX command: {args.onnx_command}")
        from multisensor_ml.optuna_dual_sensor import (
            DualSensorTuningConfig,
            tune_dual_sensor_models,
        )

        tuning_config = DualSensorTuningConfig(
            project_root=args.project_root,
            series_id=args.series,
            output_root=args.output,
            n_trials=args.trials,
            max_train_rows=args.max_train_rows,
            max_validation_rows=args.max_validation_rows,
            seed=args.seed,
        )
        result = tune_dual_sensor_models(tuning_config)
        _emit(
            status=result["status"],
            output=str(args.output.resolve()),
            variants=list(cast(list[object], result["variants"])),
            primary_metric=result["primary_metric"],
            data_status=result["model_scope"],
            real_data_status=result["real_data_status"],
            locked_test_read=result["locked_test_read"],
        )
        return 0
    if args.command == "kaggle-model":
        if args.kaggle_model_command == "package":
            config = load_kaggle_model_package_config(args.config)
            if args.source_project_root is not None:
                config = config.model_copy(
                    update={"project_root": args.source_project_root.resolve()}
                )
            package = build_kaggle_model_package(config, args.wheel)
            verified = verify_kaggle_model_package(package.root)
            _emit(
                status="PACKAGED",
                package_root=str(package.root),
                archive_sha256=verified["archive_sha256"],
                locked_test_read=False,
            )
            return 0
        if args.kaggle_model_command == "verify":
            verified = verify_kaggle_model_package(args.package)
            _emit(
                status="VERIFIED",
                archive_sha256=verified["archive_sha256"],
                reproduction_status=verified["reproduction_status"],
                locked_test_read=False,
            )
            return 0
        if args.kaggle_model_command == "reproduce":
            model_root = args.model_root.resolve()
            if not (model_root / "payload_manifest.json").is_file():
                model_root = model_root / "extracted"
            loaded_package = load_hierarchical_package(model_root)
            frame = pd.read_parquet(args.input)
            prediction = predict_hierarchical(loaded_package, frame)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            prediction.to_parquet(args.output, index=False)
            if args.expected is None:
                _emit(
                    status="PREDICTED",
                    rows=len(prediction),
                    output=str(args.output.resolve()),
                    locked_test_read=False,
                )
                return 0
            comparison = compare_expected(
                prediction, pd.read_parquet(args.expected)
            )
            _emit(
                status=comparison.status,
                compared_rows=comparison.compared_rows,
                first_mismatch_column=comparison.first_mismatch_column,
                first_mismatch_key=comparison.first_mismatch_key,
                output=str(args.output.resolve()),
                locked_test_read=False,
            )
            return 0 if comparison.status == "REPRODUCED" else 1
        raise AssertionError(
            f"unhandled Kaggle model command: {args.kaggle_model_command}"
        )

    if args.command == "availability-model":
        if args.availability_model_command == "package":
            package_root = build_sensor_availability_package(
                args.project_root.resolve(),
                series_id=args.series,
                output_root=args.output,
                max_rows_per_person=args.max_rows_per_person,
                wheel_path=args.wheel,
            )
            verified = verify_sensor_availability_package(package_root)
            _emit(
                status="PACKAGED",
                package_root=str(package_root),
                archive_sha256=verified["archive_sha256"],
                profiles=list(AVAILABILITY_PROFILES),
                locked_test_read=False,
                data_status="oracle/sanity",
                real_data_status="NOT VERIFIED",
            )
            return 0
        if args.availability_model_command == "verify":
            verified = verify_sensor_availability_package(args.package.resolve())
            _emit(
                status="VERIFIED",
                archive_sha256=verified["archive_sha256"],
                profiles=verified["profile_ids"],
                locked_test_read=False,
                data_status="oracle/sanity",
                real_data_status="NOT VERIFIED",
            )
            return 0
        raise AssertionError(
            f"unhandled availability model command: {args.availability_model_command}"
        )

    if args.command == "neon-adapter":
        if args.neon_adapter_command == "derive":
            mapped = pd.read_parquet(args.input)
            derived = derive_watch_features(
                mapped,
                clock_offset_ms=args.clock_offset_ms,
                physiological_lag_ms=args.physiological_lag_ms,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            derived.to_parquet(args.output, index=False)
            _emit(
                status="DERIVED_OBSERVED_ONLY",
                rows=len(derived),
                output=str(args.output.resolve()),
                model_ready=False,
                real_data_status="NOT VERIFIED",
            )
            return 0
        if args.neon_adapter_command == "fit":
            observed = pd.read_parquet(args.input)
            adapter = fit_personal_baseline_adapter(
                observed,
                person_id=args.person_id,
                warmup_seconds=args.warmup_seconds,
                weight_cap=args.weight_cap,
            )
            transformed = transform_with_personal_adapter(observed, adapter)
            manifest = personal_adapter_manifest(adapter, model_version=args.model_version)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            transformed_path = args.output.with_name(args.output.stem + "__transformed.parquet")
            transformed.to_parquet(transformed_path, index=False)
            _emit(
                status=adapter.status,
                manifest=str(args.output.resolve()),
                transformed=str(transformed_path.resolve()),
                personal_weight=adapter.personal_weight,
                model_weights_changed=False,
                promotable=False,
                real_data_status="NOT VERIFIED",
            )
            return 0
        raise AssertionError(
            f"unhandled Neon adapter command: {args.neon_adapter_command}"
        )

    if args.command == "factory":
        if args.factory_command == "run":
            factory_config = load_factory_config(args.config)
            receipt = run_synthetic_factory(factory_config, args.output_receipt)
            _emit(
                pipeline_run_id=receipt.pipeline_run_id,
                status=receipt.status,
                receipt=str(args.output_receipt.resolve()),
            )
            return 0
        if args.factory_command == "validate":
            receipt = validate_stage_receipt(args.receipt)
            _emit(
                pipeline_run_id=receipt.pipeline_run_id,
                stage_id=receipt.stage_id,
                status=receipt.status,
                artifact_uri=receipt.artifact_uri,
            )
            return 0
        raise AssertionError(f"unhandled factory command: {args.factory_command}")

    if args.command == "registry":
        if args.registry_command == "init":
            registry_config = load_training_registry_config(args.config)
            model_registry = ModelRegistry(registry_config.registry_path)
            model_registry.initialize()
            _emit(
                status="INITIALIZED",
                registry=str(registry_config.registry_path),
            )
            return 0
        if args.registry_command == "import-oracle":
            project = args.project_root.resolve()
            registry_path = (
                args.registry_path
                if args.registry_path.is_absolute()
                else project / args.registry_path
            )
            model_registry = ModelRegistry(registry_path)
            model_registry.initialize()
            release_id = import_oracle_bundle(
                model_registry,
                project / "artifacts" / args.experiment,
            )
            _emit(
                status="IMPORTED",
                release_id=release_id,
                registry=str(registry_path.resolve()),
            )
            return 0
        if args.registry_command == "promote":
            model_registry = ModelRegistry(args.registry_path)
            model_registry.initialize()
            model_registry.promote_release(
                args.release,
                audit_reason=args.audit_reason,
            )
            _emit(
                status="champion",
                release_id=args.release,
                registry=str(args.registry_path.resolve()),
            )
            return 0
        if args.registry_command == "run-stage":
            registry_config = load_training_registry_config(args.config)
            receipt = run_registry_stage(
                registry_config,
                stage=args.stage,
                run_id=args.run_id,
                input_receipt=args.input_receipt,
                output_receipt=args.output_receipt,
            )
            _emit(
                status=receipt.status,
                stage=receipt.stage_id,
                run_id=receipt.pipeline_run_id,
                receipt=str(args.output_receipt.resolve()),
                message_ko=receipt.message_ko,
            )
            return 0
        if args.registry_command == "run-all":
            registry_config = load_training_registry_config(args.config)
            registry_result: RegistryRunResult = run_registry_all(registry_config)
            _emit(
                status=registry_result.status,
                real_data_status="NOT VERIFIED",
                run_id=registry_result.run_id,
                series_id=registry_result.series_id,
                release_id=registry_result.release_id,
                final_receipt=str(registry_result.final_receipt.resolve()),
            )
            return 0
        if args.registry_command == "compare":
            kind_aliases = {
                "stage": "stage_model",
                "behavior": "behavior_model",
                "types": "standard_type",
                "labels": "label_set",
            }
            kind = kind_aliases.get(args.target, args.target)
            model_registry = ModelRegistry(args.registry_path)
            model_registry.initialize()
            _emit(target=args.target, versions=model_registry.list_versions(kind=kind))
            return 0
        if args.registry_command == "predict":
            model_registry = ModelRegistry(args.registry_path)
            model_registry.initialize()
            release = model_registry.release_details(args.release)
            dataset_version = str(release["dataset_version"])
            if args.dataset not in {
                dataset_version,
                dataset_version.removeprefix("dataset-"),
            } and args.dataset not in dataset_version:
                raise ValueError(
                    f"release dataset {dataset_version!r} does not match {args.dataset!r}"
                )
            predictions = model_registry.list_versions(kind="prediction")
            _emit(
                status="AVAILABLE" if predictions else "NOT_AVAILABLE",
                release_id=args.release,
                dataset_version=dataset_version,
                prediction_versions=predictions,
            )
            return 0
        if args.registry_command == "export-knime":
            project = args.project_root.resolve()
            config_path = project / "configs" / "training_registry.yaml"
            registry_config = load_training_registry_config(config_path)
            export = export_registry_knime_tables(
                registry_config,
                run_id=args.run_id,
            )
            _emit(
                status="EXPORTED",
                run_id=args.run_id,
                knime_export=str(export.resolve()),
            )
            return 0
        raise AssertionError(f"unhandled registry command: {args.registry_command}")

    if args.command == "materialize-synthetic":
        goal_config = load_goal15_config(args.config)
        registered = materialize_synthetic(goal_config)
        _emit(series_id=registered.series_id, registry=str(registered.registry_dir))
        return 0

    if args.command == "phase3":
        phase3_config = load_phase3_config(args.config)
        if args.phase3_command == "prepare":
            source = prepare_phase3_source(phase3_config)
            _emit(
                status="PREPARED",
                data_status="oracle/sanity",
                locked_test_read=False,
                source=str(source),
            )
            return 0
        if args.phase3_command == "train-validate":
            phase3_result = run_phase3_from_config(phase3_config)
            _emit(
                status="VALIDATED",
                data_status="oracle/sanity",
                real_accuracy_status="NOT VERIFIED",
                locked_test_read=False,
                manifest=str(phase3_result.manifest_json),
            )
            return 0
        if args.phase3_command == "report-input":
            phase3_manifest = phase3_config.artifact_root / "result" / "phase3_manifest.json"
            if not phase3_manifest.exists():
                raise FileNotFoundError("run 'phase3 train-validate' before report-input")
            _emit(status="READY", manifest=str(phase3_manifest))
            return 0
        raise AssertionError(f"unhandled Phase 3 command: {args.phase3_command}")

    if args.command == "prepare":
        project = args.project_root.resolve()
        prepared = prepare_series(
            project / "data" / "registry" / args.series,
            project / "data" / "prepared" / args.series,
        )
        _emit(series_id=args.series, prepared=str(prepared.root))
        return 0

    if args.command == "train":
        goal_config = load_goal15_config(args.experiment)
        bundle = train_prepared(
            goal_config.data_root / "prepared" / goal_config.series_id,
            goal_config.artifact_root / goal_config.experiment_id,
            random_state=goal_config.random_state,
            locked_test_audit_reason=goal_config.locked_test_audit_reason,
            uv_lock=Path(__file__).parents[2] / "uv.lock",
        )
        _emit(experiment_id=goal_config.experiment_id, bundle=str(bundle))
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
        goal_config = load_goal15_config(args.config)
        registered = materialize_synthetic(goal_config)
        prepared_root = goal_config.data_root / "prepared" / goal_config.series_id
        prepared = (
            _prepared_from_root(prepared_root)
            if prepared_root.exists()
            else prepare_series(registered.registry_dir, prepared_root)
        )
        bundle_root = goal_config.artifact_root / goal_config.experiment_id
        bundle = (
            bundle_root
            if bundle_root.exists()
            else train_prepared(
                prepared.root,
                bundle_root,
                random_state=goal_config.random_state,
                locked_test_audit_reason=goal_config.locked_test_audit_reason,
                uv_lock=Path(__file__).parents[2] / "uv.lock",
            )
        )
        export_root = (
            Path(__file__).parents[2]
            / "knime"
            / "exports"
            / goal_config.experiment_id
        )
        export = (
            export_root
            if export_root.exists()
            else export_knime_artifacts(bundle, prepared.root, export_root)
        )
        _emit(
            series_id=goal_config.series_id,
            experiment_id=goal_config.experiment_id,
            registry=str(registered.registry_dir),
            prepared=str(prepared.root),
            bundle=str(bundle),
            knime_export=str(export),
            status="oracle/sanity",
            real_data_status="NOT VERIFIED",
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")
