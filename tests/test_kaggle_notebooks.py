"""Static safety contracts for the unexecuted Kaggle notebook shells."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

KAGGLE_DIR = Path(__file__).parents[1] / "kaggle"
NOTEBOOKS = [
    "01_ml_data.ipynb",
    "02_ml_benchmark.ipynb",
    "03_dl_sequence_data.ipynb",
    "04_dl_tcn_benchmark.ipynb",
]
SHARED_CONSTANTS = (
    'SERIES_ID = "mvp3-oracle-v1"',
    'EXPECTED_SPLIT_COUNTS = {"train": 24, "validation": 6, "locked_test": 6}',
    'DATA_STATUS = "oracle/sanity"',
    'REAL_ACCURACY_STATUS = "NOT VERIFIED"',
    'DEVICE_SYNCHRONIZATION_STATUS = "NOT_AVAILABLE_TRUTH_ONLY"',
    "RUN_TRAINING = False",
    "RUN_LOCKED_TEST = False",
)
BEHAVIOR_CODES = (
    "ear_covering",
    "exit_attempt",
    "head_turn_away",
    "motion_freeze",
    "movement_reduction",
    "repetitive_body_movement",
    "repetitive_hand_movement",
    "repetitive_object_contact",
    "sustained_pressure_or_contact",
    "withdrawal_movement",
)


def load_notebooks() -> dict[Path, dict[str, Any]]:
    return {
        KAGGLE_DIR / filename: json.loads((KAGGLE_DIR / filename).read_text())
        for filename in NOTEBOOKS
    }


def notebook_source(notebook: dict[str, Any]) -> str:
    return "\n".join(
        normalize_cell_source(cell.get("source"))
        for cell in notebook["cells"]
    )


def normalize_cell_source(source: Any) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(source)
    return ""


def code_cell_source(notebook: dict[str, Any]) -> str:
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell.get("source")
        if isinstance(source, list):
            sources.append("\n".join(source))
        elif isinstance(source, str):
            sources.append(source)
    return "\n".join(sources)


def ml_data_namespace() -> dict[str, Any]:
    notebook = load_notebooks()[KAGGLE_DIR / "01_ml_data.ipynb"]
    safe_cells = []
    for cell in notebook["cells"]:
        source = normalize_cell_source(cell.get("source"))
        if cell["cell_type"] == "code" and "RUN_DATA_PREPARATION = False" not in source:
            safe_cells.append("\n".join(cell["source"]))
    namespace: dict[str, Any] = {}
    exec("\n".join(safe_cells), namespace)
    return namespace


def ml_benchmark_namespace() -> dict[str, Any]:
    notebook = load_notebooks()[KAGGLE_DIR / "02_ml_benchmark.ipynb"]
    namespace: dict[str, Any] = {}
    exec(code_cell_source(notebook), namespace)
    return namespace


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def write_manifest_fixture(root: Path) -> None:
    prepared_person = root / "prepared__people__person-1.parquet"
    prepared_person.write_text("prepared person")
    personal_baseline = root / "prepared__personal_baseline.parquet"
    personal_baseline.write_text("personal baseline")
    outcome_events = root / "outcomes__outcome_events.parquet"
    outcome_events.write_text("outcome events")
    outcome_stages = root / "outcomes__outcome_stages.parquet"
    outcome_stages.write_text("outcome stages")
    outcome_behaviors = root / "outcomes__outcome_behaviors.parquet"
    outcome_behaviors.write_text("outcome behaviors")
    registry_records = root / "registry__registry.jsonl"
    registry_records.write_text(
        json.dumps({"dataset_id": "person-1", "logical_hash": "a" * 64}) + "\n"
    )
    split = root / "registry__splits.parquet"
    split.write_text("split contents")
    (root / "prepared__manifest.json").write_text(
        json.dumps(
            {
                "series_id": "mvp3-oracle-v1",
                "people": [{"dataset_id": "person-1", "logical_hash": "a" * 64}],
                "personal_baseline_sha256": sha256_text("personal baseline"),
                "source_split_sha256": sha256_text("split contents"),
            }
        )
    )
    (root / "outcomes__manifest.json").write_text(
        json.dumps(
            {
                "series_id": "mvp3-oracle-v1",
                "files": {
                    "outcome_events.parquet": sha256_text("outcome events"),
                    "outcome_stages.parquet": sha256_text("outcome stages"),
                    "outcome_behaviors.parquet": sha256_text("outcome behaviors"),
                },
            }
        )
    )
    (root / "registry__manifest.json").write_text(
        json.dumps(
            {
                "series_id": "mvp3-oracle-v1",
                "records_sha256": sha256_text(
                    json.dumps({"dataset_id": "person-1", "logical_hash": "a" * 64})
                    + "\n"
                ),
            }
        )
    )


def complete_multitask_labels(labels: pd.DataFrame) -> pd.DataFrame:
    completed = labels.copy()
    if "stage_code" not in completed:
        completed["stage_code"] = completed["event_binary"].map(
            {0: "NO_EVENT", 1: "LOW"}
        )
    for behavior_code in BEHAVIOR_CODES:
        if behavior_code not in completed:
            completed[behavior_code] = 0
    return completed


def write_outcome_parquet_fixture(
    root: Path,
    stages: pd.DataFrame,
    behaviors: pd.DataFrame,
) -> None:
    pd.DataFrame(
        columns=["run_id", "person_id", "event_id", "start_time_ns", "end_time_ns"]
    ).to_parquet(root / "outcomes__outcome_events.parquet", index=False)
    stages.to_parquet(root / "outcomes__outcome_stages.parquet", index=False)
    behaviors.to_parquet(root / "outcomes__outcome_behaviors.parquet", index=False)
    (root / "outcomes__manifest.json").write_text(
        json.dumps(
            {
                "series_id": "mvp3-oracle-v1",
                "files": {
                    "outcome_events.parquet": hashlib.sha256(
                        (root / "outcomes__outcome_events.parquet").read_bytes()
                    ).hexdigest(),
                    "outcome_stages.parquet": hashlib.sha256(
                        (root / "outcomes__outcome_stages.parquet").read_bytes()
                    ).hexdigest(),
                    "outcome_behaviors.parquet": hashlib.sha256(
                        (root / "outcomes__outcome_behaviors.parquet").read_bytes()
                    ).hexdigest(),
                },
            }
        )
    )


def test_notebooks_have_required_structure() -> None:
    for path, notebook in load_notebooks().items():
        cell_types = {cell["cell_type"] for cell in notebook["cells"]}
        assert notebook["nbformat"] == 4, path
        assert "markdown" in cell_types, path
        assert "code" in cell_types, path
        assert notebook["metadata"]["kernelspec"]["name"] == "python3", path


def test_kaggle_directory_contains_exactly_the_four_contract_notebooks() -> None:
    assert sorted(path.name for path in KAGGLE_DIR.glob("*.ipynb")) == NOTEBOOKS


def test_notebooks_have_korean_title_english_purpose_and_oracle_warning() -> None:
    for path, notebook in load_notebooks().items():
        markdown = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        assert any("가" <= character <= "힣" for character in markdown), path
        assert "Purpose:" in markdown, path
        assert "oracle/sanity-only" in markdown, path


def test_notebooks_are_unexecuted() -> None:
    for notebook in load_notebooks().values():
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []


def test_notebooks_do_not_embed_secrets() -> None:
    forbidden = ("wandb.ai/authorize", "kaggle.json", "api_key=", "WANDB_API_KEY=")
    for path, notebook in load_notebooks().items():
        source = notebook_source(notebook)
        assert not any(token in source for token in forbidden), path


def test_notebook_source_includes_string_form_cell_source_for_secret_checks() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": "WANDB_API_KEY=must-not-be-embedded",
            }
        ]
    }
    source = notebook_source(notebook)
    assert "WANDB_API_KEY=" in source


def test_notebooks_declare_shared_safety_constants() -> None:
    for path, notebook in load_notebooks().items():
        source = notebook_source(notebook)
        for constant in SHARED_CONSTANTS:
            assert constant in source, path


def test_ml_data_contract() -> None:
    """The ML data notebook exposes the safe row-view preparation contract."""
    path = KAGGLE_DIR / "01_ml_data.ipynb"
    notebook = load_notebooks()[path]
    source = notebook_source(notebook)
    compile(code_cell_source(notebook), str(path), "exec")

    expected_definitions = (
        "resolve_kaggle_dataset_root",
        "load_split_registry",
        "validate_split_contract",
        "validate_manifest_hashes",
        "assert_no_truth_leakage",
        "build_ml_role_view",
        "write_ml_view_manifest",
    )
    for definition in expected_definitions:
        assert f"def {definition}(" in source

    assert 'ML_OUTPUT_ROOT = Path("/kaggle/working/goal15_ml_view")' in source
    assert '    "active_target_",' in source
    assert '    "hard_negative_id",' in source
    assert '    "hard_negative_type",' in source
    assert '    "artifact_schedule_id",' in source
    assert '    "participant_truth_baseline",' in source
    assert '    "event_intensity_truth",' in source
    assert "RUN_DATA_PREPARATION = False" in source
    assert "if RUN_DATA_PREPARATION:" in source


def test_validate_split_contract_rejects_person_overlap() -> None:
    namespace = ml_data_namespace()
    namespace["EXPECTED_SPLIT_COUNTS"] = {
        "train": 24,
        "validation": 6,
        "locked_test": 6,
    }
    split = pd.DataFrame(
        {
            "person_key": [
                *(f"P{index:02d}" for index in range(24)),
                "P00",
                *(f"P{index:02d}" for index in range(24, 29)),
                *(f"P{index:02d}" for index in range(29, 35)),
            ],
            "split_role": ["train"] * 24 + ["validation"] * 6 + ["locked_test"] * 6,
        }
    )

    with pytest.raises(ValueError, match="person leakage"):
        namespace["validate_split_contract"](split)


def test_build_ml_role_view_keeps_id_and_type_hard_negatives_and_caps_baselines() -> None:
    namespace = ml_data_namespace()
    namespace["EXPECTED_SPLIT_COUNTS"] = {
        "train": 1,
        "validation": 1,
        "locked_test": 1,
    }
    prepared = pd.DataFrame(
        {
            "person_key": ["train"] * 10,
            "run_id": ["run"] * 10,
            "person_id": ["train"] * 10,
            "canonical_time": list(range(10)),
            "feature": list(range(10)),
        }
    )
    labels = complete_multitask_labels(pd.DataFrame(
        {
                "run_id": ["run"] * 10,
                "person_id": ["train"] * 10,
                "canonical_time": list(range(10)),
                "event_binary": [1] + [0] * 9,
                "hard_negative_id": [None, "negative-id", *([None] * 8)],
                "hard_negative_type": [None, None, "artifact-only", *([None] * 7)],
        }
    ))
    split = pd.DataFrame({"person_key": ["train"], "split_role": ["train"]})

    view = namespace["build_ml_role_view"](prepared, labels, split, "train")

    assert {0, 1, 2}.issubset(set(view["canonical_time"]))
    assert len(view) == 6
    assert not any(
        column.startswith(("hard_negative_id", "hard_negative_type"))
        for column in view.columns
    )


@pytest.mark.parametrize("split_role", ["train", "validation", "locked_test"])
def test_build_ml_role_view_removes_denied_columns_for_every_role(split_role: str) -> None:
    namespace = ml_data_namespace()
    namespace["EXPECTED_SPLIT_COUNTS"] = {
        "train": 1,
        "validation": 1,
        "locked_test": 1,
    }
    prepared = pd.DataFrame(
        {
            "person_key": [split_role],
                "run_id": ["run"],
                "person_id": [split_role],
            "canonical_time": [0],
            "feature": [1.0],
        }
    )
    labels = complete_multitask_labels(pd.DataFrame(
        {
            "run_id": ["run"],
            "person_id": [split_role],
            "canonical_time": [0],
            "event_binary": [1],
            "event_intensity_truth": [0.9],
            "active_target_event": [1],
            "hard_negative_id": ["not-a-feature"],
        }
    ))
    split = pd.DataFrame({"person_key": [split_role], "split_role": [split_role]})

    view = namespace["build_ml_role_view"](prepared, labels, split, split_role)

    namespace["assert_no_truth_leakage"](view.columns)


def test_validate_manifest_hashes_rejects_missing_flat_file(tmp_path: Path) -> None:
    namespace = ml_data_namespace()
    write_manifest_fixture(tmp_path)
    (tmp_path / "outcomes__outcome_events.parquet").unlink()

    with pytest.raises((FileNotFoundError, ValueError), match=r"outcome_events|missing"):
        namespace["validate_manifest_hashes"](tmp_path)


def test_validate_manifest_hashes_rejects_hash_mismatch(tmp_path: Path) -> None:
    namespace = ml_data_namespace()
    write_manifest_fixture(tmp_path)
    (tmp_path / "outcomes__outcome_events.parquet").write_text("modified")

    with pytest.raises(ValueError, match="hash mismatch"):
        namespace["validate_manifest_hashes"](tmp_path)


@pytest.mark.parametrize(
    ("manifest_name", "hash_field"),
    [
        ("prepared__manifest.json", "personal_baseline_sha256"),
        ("prepared__manifest.json", "source_split_sha256"),
        ("registry__manifest.json", "records_sha256"),
    ],
)
def test_validate_manifest_hashes_rejects_missing_required_hash_declaration(
    tmp_path: Path, manifest_name: str, hash_field: str
) -> None:
    namespace = ml_data_namespace()
    write_manifest_fixture(tmp_path)
    manifest_path = tmp_path / manifest_name
    manifest = json.loads(manifest_path.read_text())
    manifest.pop(hash_field)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=f"missing required hash: {hash_field}"):
        namespace["validate_manifest_hashes"](tmp_path)


def test_outcome_labels_materialize_stage_and_behavior_columns(tmp_path: Path) -> None:
    namespace = ml_data_namespace()
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    stages = pd.DataFrame(
        {
            "run_id": ["run", "run", "run"],
            "person_id": ["P1", "P1", "P1"],
            "timestamp_utc": [
                timestamp,
                timestamp + pd.Timedelta(seconds=1),
                timestamp + pd.Timedelta(seconds=2),
            ],
            "event_id": ["target-1", None, "hard-negative-1"],
            "stage_code": ["LOW", "NO_EVENT", "NO_EVENT"],
        }
    )
    behaviors = pd.DataFrame(
        {
            "run_id": ["run", "run", "run"],
            "person_id": ["P1", "P1", "P1"],
            "event_id": ["target-1", "target-1", "hard-negative-1"],
            "behavior_code": ["ear_covering", "exit_attempt", "movement_reduction"],
            "label_value": [1, 1, 1],
        }
    )
    write_outcome_parquet_fixture(tmp_path, stages, behaviors)

    labels = namespace["_load_outcome_labels"](tmp_path)

    assert set(BEHAVIOR_CODES).issubset(labels.columns)
    target = labels.loc[labels["event_id"] == "target-1"].iloc[0]
    non_event = labels.loc[labels["event_id"].isna()].iloc[0]
    hard_negative = labels.loc[labels["event_id"] == "hard-negative-1"].iloc[0]
    assert target["ear_covering"] == 1
    assert target["exit_attempt"] == 1
    assert (non_event[list(BEHAVIOR_CODES)] == 0).all()
    assert hard_negative["movement_reduction"] == 1


def test_outcome_labels_reject_unknown_behavior_code(tmp_path: Path) -> None:
    namespace = ml_data_namespace()
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    stages = pd.DataFrame({
        "run_id": ["run"], "person_id": ["P1"], "timestamp_utc": [timestamp],
        "event_id": ["event"], "stage_code": ["LOW"],
    })
    behaviors = pd.DataFrame({
        "run_id": ["run"], "person_id": ["P1"], "event_id": ["event"],
        "behavior_code": ["unknown"], "label_value": [1],
    })
    write_outcome_parquet_fixture(tmp_path, stages, behaviors)

    with pytest.raises(ValueError, match="unknown behavior"):
        namespace["_load_outcome_labels"](tmp_path)


def test_outcome_labels_reject_duplicate_conflicting_behavior_labels(tmp_path: Path) -> None:
    namespace = ml_data_namespace()
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    stages = pd.DataFrame({
        "run_id": ["run"], "person_id": ["P1"], "timestamp_utc": [timestamp],
        "event_id": ["event"], "stage_code": ["LOW"],
    })
    behaviors = pd.DataFrame(
        {
            "run_id": ["run", "run"],
            "person_id": ["P1", "P1"],
            "event_id": ["event", "event"],
            "behavior_code": ["ear_covering", "ear_covering"],
            "label_value": [0, 1],
        }
    )
    write_outcome_parquet_fixture(tmp_path, stages, behaviors)

    with pytest.raises(ValueError, match="conflicting behavior"):
        namespace["_load_outcome_labels"](tmp_path)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("behavior_code", None),
        ("behavior_code", float("nan")),
        ("label_value", None),
        ("label_value", float("nan")),
        ("label_value", -1),
        ("label_value", 2),
        ("label_value", "positive"),
    ],
    ids=(
        "null-code",
        "nan-code",
        "null-value",
        "nan-value",
        "negative-value",
        "value-above-one",
        "nonnumeric-value",
    ),
)
def test_outcome_labels_reject_invalid_behavior_rows_before_pivot(
    tmp_path: Path,
    field: str,
    invalid_value: Any,
) -> None:
    namespace = ml_data_namespace()
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    stages = pd.DataFrame({
        "run_id": ["run"], "person_id": ["P1"], "timestamp_utc": [timestamp],
        "event_id": ["event"], "stage_code": ["LOW"],
    })
    behavior_row: dict[str, list[Any]] = {
        "run_id": ["run"],
        "person_id": ["P1"],
        "event_id": ["event"],
        "behavior_code": ["ear_covering"],
        "label_value": [1],
    }
    behavior_row[field] = [invalid_value]
    write_outcome_parquet_fixture(tmp_path, stages, pd.DataFrame(behavior_row))

    with pytest.raises(ValueError, match=field):
        namespace["_load_outcome_labels"](tmp_path)


@pytest.mark.parametrize("label_value", [0, 1])
def test_outcome_labels_accept_exact_binary_behavior_values(
    tmp_path: Path,
    label_value: int,
) -> None:
    namespace = ml_data_namespace()
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    stages = pd.DataFrame({
        "run_id": ["run"], "person_id": ["P1"], "timestamp_utc": [timestamp],
        "event_id": ["event"], "stage_code": ["LOW"],
    })
    behaviors = pd.DataFrame({
        "run_id": ["run"], "person_id": ["P1"], "event_id": ["event"],
        "behavior_code": ["ear_covering"], "label_value": [label_value],
    })
    write_outcome_parquet_fixture(tmp_path, stages, behaviors)

    labels = namespace["_load_outcome_labels"](tmp_path)

    assert labels.loc[0, "ear_covering"] == label_value


def test_write_ml_view_manifest_uses_bounded_metadata_without_reading_role_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = ml_data_namespace()
    view_paths = {
        role: tmp_path / f"{role}.parquet"
        for role in ("train", "validation", "locked_test")
    }
    for role, path in view_paths.items():
        path.write_bytes(f"bounded-{role}".encode())
    row_counts = {"train": 12, "validation": 5, "locked_test": 7}
    role_columns = {
        role: ["person_key", "canonical_time", "feature", "event_binary"]
        for role in view_paths
    }
    real_read_parquet = pd.read_parquet

    def reject_role_output_reads(path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if Path(path) in view_paths.values():
            raise AssertionError("manifest creation loaded a finished role output")
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", reject_role_output_reads)

    manifest_path = namespace["write_ml_view_manifest"](
        tmp_path,
        view_paths,
        "a" * 64,
        "b" * 64,
        row_counts,
        role_columns,
    )
    manifest = json.loads(manifest_path.read_text())

    for role, path in view_paths.items():
        assert manifest["files"][role]["row_count"] == row_counts[role]
        assert manifest["files"][role]["columns"] == role_columns[role]
        assert manifest["files"][role]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("required_file", ["outcome_stages.parquet", "outcome_behaviors.parquet"])
def test_outcome_manifest_requires_stage_and_behavior_hashes(
    tmp_path: Path, required_file: str
) -> None:
    namespace = ml_data_namespace()
    write_manifest_fixture(tmp_path)
    manifest_path = tmp_path / "outcomes__manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"].pop(required_file)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=f"missing required outcome hash: {required_file}"):
        namespace["validate_manifest_hashes"](tmp_path)


def test_outcome_manifest_rejects_stage_hash_mismatch(tmp_path: Path) -> None:
    namespace = ml_data_namespace()
    write_manifest_fixture(tmp_path)
    (tmp_path / "outcomes__outcome_stages.parquet").write_text("modified")

    with pytest.raises(ValueError, match="hash mismatch"):
        namespace["validate_manifest_hashes"](tmp_path)


def test_build_ml_role_view_preserves_event_boundary_semantics() -> None:
    namespace = ml_data_namespace()
    namespace["EXPECTED_SPLIT_COUNTS"] = {"train": 1, "validation": 1, "locked_test": 1}
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    prepared = pd.DataFrame({
        "person_key": ["P1", "P1"], "run_id": ["run", "run"], "person_id": ["P1", "P1"],
        "canonical_time": [timestamp, timestamp + pd.Timedelta(seconds=1)],
        "event_binary": [0, 1], "feature": [1.0, 2.0],
    })
    labels = complete_multitask_labels(pd.DataFrame({
        "run_id": ["run", "run"], "person_id": ["P1", "P1"],
        "canonical_time": [timestamp, timestamp + pd.Timedelta(seconds=1)],
        "stage_code": ["LOW", "HIGH"], "event_binary": [0, 1],
    }))
    split = pd.DataFrame({"person_key": ["P1"], "split_role": ["validation"]})

    view = namespace["build_ml_role_view"](
        prepared, labels.drop(columns="event_binary"), split, "validation"
    )

    assert view["event_binary"].tolist() == [0, 1]
    assert view["pattern_binary"].tolist() == [1, 1]


def test_outcome_labels_propagate_hard_negative_behaviors(tmp_path: Path) -> None:
    namespace = ml_data_namespace()
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    stages = pd.DataFrame({
        "run_id": ["run"], "person_id": ["P1"], "timestamp_utc": [timestamp],
        "event_id": [None], "stage_code": ["NO_EVENT"],
    })
    behaviors = pd.DataFrame({
        "run_id": ["run"], "person_id": ["P1"], "event_id": ["hard-1"],
        "behavior_code": ["movement_reduction"], "label_value": [1],
    })
    write_outcome_parquet_fixture(tmp_path, stages, behaviors)
    events = pd.DataFrame({
        "run_id": ["run"], "person_id": ["P1"], "event_id": ["hard-1"],
        "start_time_ns": [timestamp.value], "end_time_ns": [timestamp.value],
    })
    events.to_parquet(tmp_path / "outcomes__outcome_events.parquet", index=False)
    manifest = json.loads((tmp_path / "outcomes__manifest.json").read_text())
    manifest["files"]["outcome_events.parquet"] = hashlib.sha256(
        (tmp_path / "outcomes__outcome_events.parquet").read_bytes()
    ).hexdigest()
    (tmp_path / "outcomes__manifest.json").write_text(json.dumps(manifest))

    labels = namespace["_load_outcome_labels"](tmp_path)

    assert labels.loc[0, "movement_reduction"] == 1


def _write_ml_view_fixture(root: Path) -> None:
    namespace = ml_benchmark_namespace()
    behavior_columns = list(namespace["BEHAVIOR_CODES"])
    stage_codes = list(namespace["STAGE_CODES"])
    files: dict[str, dict[str, Any]] = {}
    role_counts = {"train": 24, "validation": 6, "locked_test": 6}
    person_offset = 0
    for split_role, person_count in role_counts.items():
        rows: list[dict[str, Any]] = []
        for local_index in range(person_count):
            index = person_offset + local_index
            is_pattern = index % 6 != 0
            row: dict[str, Any] = {
                "person_key": f"person-{index:02d}",
                "canonical_time": index,
                "feature_one": float(index % 5),
                "feature_two": float(index // 3),
                "pattern_binary": int(is_pattern),
                "event_binary": int(index % 5 != 0),
                "stage_code": (
                    stage_codes[index % len(stage_codes)] if is_pattern else "NO_EVENT"
                ),
            }
            row.update(
                {
                    column: int((index + offset) % 3 == 0)
                    for offset, column in enumerate(behavior_columns)
                }
            )
            rows.append(row)
        path = root / f"{split_role}.parquet"
        frame = pd.DataFrame(rows)
        frame.to_parquet(path, index=False)
        files[split_role] = {
            "path": path.name,
            "sha256": namespace["sha256_file"](path),
            "row_count": len(frame),
            "columns": list(frame.columns),
        }
        person_offset += person_count
    (root / "view_manifest.json").write_text(
        json.dumps(
            {
                "series_id": "mvp3-oracle-v1",
                "data_status": "oracle/sanity",
                "source_dataset_hash": "a" * 64,
                "split_hash": "b" * 64,
                "files": files,
            }
        )
    )


def _rewrite_ml_role_fixture(root: Path, split_role: str, frame: pd.DataFrame) -> None:
    namespace = ml_benchmark_namespace()
    path = root / f"{split_role}.parquet"
    frame.to_parquet(path, index=False)
    manifest_path = root / "view_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][split_role].update(
        {
            "sha256": namespace["sha256_file"](path),
            "row_count": len(frame),
            "columns": list(frame.columns),
        }
    )
    manifest_path.write_text(json.dumps(manifest))


def test_ml_benchmark_contract() -> None:
    """The ML notebook is CPU-only, compiled, and secrets are Kaggle-only."""
    path = KAGGLE_DIR / "02_ml_benchmark.ipynb"
    notebook = load_notebooks()[path]
    source = notebook_source(notebook)
    compile(code_cell_source(notebook), str(path), "exec")

    expected_definitions = (
        "load_ml_views",
        "verify_ml_view_manifest",
        "fit_logistic_candidate",
        "fit_hgb_candidate",
        "select_validation_threshold",
        "compute_common_metrics",
        "bootstrap_people_ci",
        "select_validation_champion",
        "login_wandb_from_kaggle_secret",
    )
    for definition in expected_definitions:
        assert f"def {definition}(" in source

    assert "CUDA_VISIBLE_DEVICES" not in source
    assert 'device = "cpu"' in source
    assert source.count('UserSecretsClient().get_secret("WANDB_API_KEY")') == 1
    assert source.count("WANDB_API_KEY") == 1
    assert "RUN_TRAINING = False" in source
    assert "RUN_LOCKED_TEST = False" in source
    assert 'PATTERN_TARGET = "pattern_binary"' in source
    assert 'ONSET_EVENT_TARGET = "event_binary"' in source


@pytest.mark.parametrize("missing_target", ["pattern_binary", "event_binary", "stage_code"])
def test_ml_view_manifest_requires_all_multitask_labels(
    tmp_path: Path, missing_target: str
) -> None:
    namespace = ml_benchmark_namespace()
    _write_ml_view_fixture(tmp_path)

    manifest = namespace["verify_ml_view_manifest"](tmp_path)
    views = namespace["load_ml_views"](tmp_path)

    assert set(manifest["files"]) == {"train", "validation", "locked_test"}
    assert set(views) == {"train", "validation", "locked_test"}
    assert set(views["validation"]["split_role"]) == {"validation"}

    broken = json.loads((tmp_path / "view_manifest.json").read_text())
    broken["files"]["train"]["columns"].remove(missing_target)
    (tmp_path / "view_manifest.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="required columns"):
        namespace["verify_ml_view_manifest"](tmp_path)


def test_load_ml_views_rejects_person_overlap(tmp_path: Path) -> None:
    namespace = ml_benchmark_namespace()
    _write_ml_view_fixture(tmp_path)
    train = pd.read_parquet(tmp_path / "train.parquet")
    validation = pd.read_parquet(tmp_path / "validation.parquet")
    validation.loc[validation.index[0], "person_key"] = train.loc[train.index[0], "person_key"]
    _rewrite_ml_role_fixture(tmp_path, "validation", validation)

    with pytest.raises(ValueError, match="person leakage"):
        namespace["load_ml_views"](tmp_path)


def test_load_ml_views_rejects_wrong_person_count(tmp_path: Path) -> None:
    namespace = ml_benchmark_namespace()
    _write_ml_view_fixture(tmp_path)
    train = pd.read_parquet(tmp_path / "train.parquet").iloc[:-1].copy()
    _rewrite_ml_role_fixture(tmp_path, "train", train)

    with pytest.raises(ValueError, match="person count mismatch"):
        namespace["load_ml_views"](tmp_path)


def test_numeric_feature_contract_handles_task2_shaped_frame_for_both_candidates() -> None:
    namespace = ml_benchmark_namespace()
    rows = 60
    frame = pd.DataFrame(
        {
            "person_id": [f"P{value % 8:02d}" for value in range(rows)],
            "person_key": [f"run/P{value % 8:02d}" for value in range(rows)],
            "run_id": ["run"] * rows,
            "dataset_id": ["dataset"] * rows,
            "canonical_time": pd.date_range("2026-01-01", periods=rows, freq="s", tz="UTC"),
            "split_role": ["train"] * rows,
            "context": ["focused_task"] * rows,
            "event_id": [f"event-{value // 6}" for value in range(rows)],
            "causal_z": [float(value % 5) for value in range(rows)],
            "causal_slope": [float(value // 5) for value in range(rows)],
            "validity_flag": [value % 2 == 0 for value in range(rows)],
            "pattern_binary": [int(value % 6 != 0) for value in range(rows)],
            "event_binary": [int(value % 5 != 0) for value in range(rows)],
            "stage_code": [
                namespace["STAGE_CODES"][value % len(namespace["STAGE_CODES"])]
                if value % 6 != 0
                else "NO_EVENT"
                for value in range(rows)
            ],
        }
    )
    for offset, behavior in enumerate(namespace["BEHAVIOR_CODES"]):
        frame[behavior] = [int((value + offset) % 3 == 0) for value in range(rows)]

    features = namespace["_infer_feature_columns"](frame)
    candidates = [
        namespace["fit_logistic_candidate"](frame, features),
        namespace["fit_hgb_candidate"](frame, features),
    ]

    assert features == ["causal_z", "causal_slope", "validity_flag"]
    assert all(candidate["pattern_model"].n_features_in_ == 3 for candidate in candidates)


@pytest.mark.parametrize(
    "invalid_feature",
    [
        pd.Series(["bad", "feature"], dtype="object"),
        pd.Series(["bad", "feature"], dtype="category"),
        pd.Series(pd.date_range("2026-01-01", periods=2, freq="s")),
    ],
    ids=("object", "category", "timestamp"),
)
def test_numeric_feature_contract_rejects_unclassified_nonnumeric_columns(
    invalid_feature: pd.Series,
) -> None:
    namespace = ml_benchmark_namespace()
    frame = pd.DataFrame({"unexpected_feature": invalid_feature})

    with pytest.raises(ValueError, match="non-numeric feature columns"):
        namespace["_infer_feature_columns"](frame)


def test_feature_matrix_rejects_nonfinite_numeric_values() -> None:
    namespace = ml_benchmark_namespace()
    frame = pd.DataFrame({"causal_z": [0.0, float("inf")]})

    with pytest.raises(ValueError, match="non-finite"):
        namespace["_feature_matrix"](frame, ["causal_z"])


def test_numeric_feature_contract_rejects_empty_feature_set() -> None:
    namespace = ml_benchmark_namespace()
    frame = pd.DataFrame(
        {
            "person_key": ["P1"],
            "canonical_time": [0],
            "pattern_binary": [0],
            "event_binary": [0],
            "stage_code": ["NO_EVENT"],
        }
    )
    for behavior in namespace["BEHAVIOR_CODES"]:
        frame[behavior] = 0

    with pytest.raises(ValueError, match="at least one feature"):
        namespace["_infer_feature_columns"](frame)


def test_ml_candidates_fit_event_stage_and_behavior_heads() -> None:
    namespace = ml_benchmark_namespace()
    frame = pd.DataFrame(
        {
            "feature_one": [float(value % 5) for value in range(60)],
            "feature_two": [float(value // 5) for value in range(60)],
            "pattern_binary": [int(value % 6 != 0) for value in range(60)],
            "event_binary": [0] * 60,
            "stage_code": [
                namespace["STAGE_CODES"][value % len(namespace["STAGE_CODES"])]
                if value % 6 != 0
                else "NO_EVENT"
                for value in range(60)
            ],
        }
    )
    for offset, behavior in enumerate(namespace["BEHAVIOR_CODES"]):
        frame[behavior] = [int((value + offset) % 3 == 0) for value in range(60)]

    for factory in (namespace["fit_logistic_candidate"], namespace["fit_hgb_candidate"]):
        candidate = factory(frame, ["feature_one", "feature_two"])
        assert set(candidate) == {
            "behavior_models",
            "model_name",
            "pattern_model",
            "stage_models",
        }
        assert set(candidate["stage_models"]) == set(namespace["STAGE_CODES"])
        assert set(candidate["behavior_models"]) == set(namespace["BEHAVIOR_CODES"])
        probability = candidate["pattern_model"].predict_proba(
            namespace["_feature_matrix"](frame, ["feature_one", "feature_two"])
        )[:, 1]
        assert probability.shape == (len(frame),)
        assert ((probability >= 0) & (probability <= 1)).all()


def test_ml_training_masks_keep_pattern_onset_and_behavior_semantics_distinct() -> None:
    namespace = ml_benchmark_namespace()
    frame = pd.DataFrame(
        {
            "row_kind": ["low_pre_onset", "onset_audit", "hard_negative", "baseline"],
            "pattern_binary": [1, 0, 0, 0],
            "event_binary": [0, 1, 0, 0],
            "stage_code": ["LOW", "NO_EVENT", "NO_EVENT", "NO_EVENT"],
        }
    )
    for behavior in namespace["BEHAVIOR_CODES"]:
        frame[behavior] = 0
    frame.loc[frame["row_kind"].eq("hard_negative"), "ear_covering"] = 1

    stage_rows = frame.loc[namespace["_select_stage_training_rows"](frame), "row_kind"]
    behavior_rows = frame.loc[namespace["_select_behavior_training_rows"](frame), "row_kind"]

    assert stage_rows.tolist() == ["low_pre_onset"]
    assert behavior_rows.tolist() == ["low_pre_onset", "hard_negative"]
    assert frame.loc[frame["row_kind"].eq("onset_audit"), "event_binary"].item() == 1
    assert frame.loc[frame["row_kind"].eq("onset_audit"), "stage_code"].item() == "NO_EVENT"


def test_prediction_rows_apply_target_specific_decision_masks() -> None:
    namespace = ml_benchmark_namespace()

    class ConstantProbabilityModel:
        classes_ = np.array([0, 1], dtype=np.int8)

        def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
            return np.tile(np.array([[0.4, 0.6]]), (len(matrix), 1))

    frame = pd.DataFrame(
        {
            "person_key": ["P1"] * 4,
            "canonical_time": [0, 1, 2, 3],
            "causal_z": [0.1, 0.2, 0.3, 0.4],
            "pattern_binary": [1, 0, 0, 0],
            "event_binary": [0, 1, 0, 0],
            "stage_code": ["LOW", "NO_EVENT", "NO_EVENT", "NO_EVENT"],
        }
    )
    for behavior in namespace["BEHAVIOR_CODES"]:
        frame[behavior] = 0
    frame.loc[2, "ear_covering"] = 1
    model = ConstantProbabilityModel()
    candidate = {
        "model_name": "constant",
        "pattern_model": model,
        "stage_models": {stage: model for stage in namespace["STAGE_CODES"]},
        "behavior_models": {behavior: model for behavior in namespace["BEHAVIOR_CODES"]},
    }

    predictions = namespace["_prediction_rows"](
        candidate,
        frame,
        ["causal_z"],
        split_role="validation",
        pattern_threshold=0.5,
    )

    assert len(predictions.loc[predictions["target"].eq("pattern_binary")]) == 4
    assert all(
        len(predictions.loc[predictions["target"].eq(f"stage::{stage}")]) == 1
        for stage in namespace["STAGE_CODES"]
    )
    assert all(
        len(predictions.loc[predictions["target"].eq(f"behavior::{behavior}")]) == 2
        for behavior in namespace["BEHAVIOR_CODES"]
    )


def test_event_segmentation_never_merges_people_at_boundaries() -> None:
    namespace = ml_benchmark_namespace()
    people = np.array(["P1", "P1", "P2", "P2"])
    canonical_time = np.array([0, 1, 0, 1])

    event_recall, false_alerts = namespace["_event_alert_summary"](
        np.array([0, 1, 1, 0], dtype=np.int8),
        np.array([0, 1, 0, 0], dtype=bool),
        people,
        canonical_time,
    )
    _, boundary_false_alerts = namespace["_event_alert_summary"](
        np.zeros(4, dtype=np.int8),
        np.array([0, 1, 1, 0], dtype=bool),
        people,
        canonical_time,
    )

    assert event_recall == 0.5
    assert false_alerts == 0.0
    assert boundary_false_alerts == 2.0


def test_event_segmentation_rejects_out_of_order_person_time() -> None:
    namespace = ml_benchmark_namespace()

    with pytest.raises(ValueError, match="not ordered"):
        namespace["_event_alert_summary"](
            np.array([0, 1], dtype=np.int8),
            np.array([0, 1], dtype=bool),
            np.array(["P1", "P1"]),
            np.array([1, 0]),
        )


def test_hgb_candidate_uses_balanced_class_weight() -> None:
    namespace = ml_benchmark_namespace()
    estimator = namespace["_make_hgb_estimator"]()

    assert estimator.class_weight == "balanced"


def test_validation_helpers_do_not_select_locked_test_metrics() -> None:
    namespace = ml_benchmark_namespace()
    truth = pd.Series([0, 0, 1, 1, 0, 0], dtype="int8")
    probability = pd.Series([0.1, 0.2, 0.9, 0.8, 0.7, 0.1], dtype="float64")
    people = np.array(["P1"] * 6)
    canonical_time = np.arange(6)
    threshold = namespace["select_validation_threshold"](
        truth.to_numpy(),
        probability.to_numpy(),
        people,
        canonical_time,
        duration_hours=6 / 3600,
    )
    metrics = namespace["compute_common_metrics"](
        truth.to_numpy(),
        probability.to_numpy(),
        threshold=threshold,
        duration_hours=6 / 3600,
        model_name="logistic_regression",
        split_role="validation",
        target="pattern_binary",
        person_keys=people,
        canonical_time=canonical_time,
    )
    assert set(metrics.columns) == set(namespace["METRIC_COLUMNS"])
    assert "aucpr" in set(metrics["metric"])

    def metric_row(model_name: str, split_role: str, metric: str, value: float) -> dict[str, Any]:
        return {
            "model_name": model_name,
            "split_role": split_role,
            "target": "pattern_binary",
            "metric": metric,
            "value": value,
        }

    candidates = pd.DataFrame(
        [
            metric_row("logistic_regression", "validation", "aucpr", 0.70),
            metric_row("logistic_regression", "validation", "event_recall", 0.80),
            metric_row("logistic_regression", "validation", "false_alerts_per_hour", 2.0),
            metric_row("logistic_regression", "validation", "ece", 0.2),
            metric_row("hist_gradient_boosting", "validation", "aucpr", 0.70),
            metric_row("hist_gradient_boosting", "validation", "event_recall", 0.80),
            metric_row("hist_gradient_boosting", "validation", "false_alerts_per_hour", 1.0),
            metric_row("hist_gradient_boosting", "validation", "ece", 0.1),
            metric_row("hist_gradient_boosting", "locked_test", "aucpr", 1.0),
        ]
    )
    assert namespace["select_validation_champion"](candidates) == "hist_gradient_boosting"


def test_people_bootstrap_returns_ordered_confidence_interval() -> None:
    namespace = ml_benchmark_namespace()
    frame = pd.DataFrame(
        {
            "person_key": ["a", "a", "b", "b", "c", "c"],
            "label": [0, 1, 0, 1, 0, 1],
            "probability": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
        }
    )

    interval = namespace["bootstrap_people_ci"](frame, iterations=50, random_state=7)

    assert 0.0 <= interval["lower"] <= interval["estimate"] <= interval["upper"] <= 1.0
