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
        if cell["cell_type"] == "code" and "if RUN_DATA_PREPARATION:" not in source:
            safe_cells.append("\n".join(cell["source"]))
    namespace: dict[str, Any] = {}
    exec("\n".join(safe_cells), namespace)
    return namespace


def ml_benchmark_namespace() -> dict[str, Any]:
    notebook = load_notebooks()[KAGGLE_DIR / "02_ml_benchmark.ipynb"]
    namespace: dict[str, Any] = {}
    exec(code_cell_source(notebook), namespace)
    return namespace


def dl_sequence_namespace() -> dict[str, Any]:
    """Load only definitions from the unexecuted DL sequence notebook."""
    notebook = load_notebooks()[KAGGLE_DIR / "03_dl_sequence_data.ipynb"]
    safe_cells = []
    for cell in notebook["cells"]:
        source = normalize_cell_source(cell.get("source"))
        if cell["cell_type"] == "code" and "RUN_DATA_PREPARATION = False" not in source:
            safe_cells.append("\n".join(cell["source"]))
    namespace: dict[str, Any] = {}
    exec("\n".join(safe_cells), namespace)
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
                "autonomic_arousal__robust_z": float(index % 5),
                "motor_activation__mean_5s": float(index // 3),
                "pattern_binary": int(is_pattern),
                "event_binary": int(index % 5 != 0),
                "hard_negative": int(not is_pattern),
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


@pytest.mark.parametrize(
    "missing_target",
    ["pattern_binary", "event_binary", "hard_negative", "stage_code"],
)
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


@pytest.mark.parametrize(
    ("split_role", "column"),
    [("validation", "future_label"), ("locked_test", "participant_index")],
)
def test_loaded_roles_reject_adversarial_numeric_columns(
    tmp_path: Path, split_role: str, column: str
) -> None:
    namespace = ml_benchmark_namespace()
    _write_ml_view_fixture(tmp_path)
    frame = pd.read_parquet(tmp_path / f"{split_role}.parquet")
    frame[column] = range(len(frame))
    _rewrite_ml_role_fixture(tmp_path, split_role, frame)

    with pytest.raises(ValueError, match="unapproved columns"):
        namespace["load_ml_views"](tmp_path)


@pytest.mark.parametrize("split_role", ["validation", "locked_test"])
def test_loaded_roles_require_the_train_feature_schema(
    tmp_path: Path, split_role: str
) -> None:
    namespace = ml_benchmark_namespace()
    _write_ml_view_fixture(tmp_path)
    frame = pd.read_parquet(tmp_path / f"{split_role}.parquet").drop(
        columns="motor_activation__mean_5s"
    )
    _rewrite_ml_role_fixture(tmp_path, split_role, frame)

    with pytest.raises(ValueError, match="feature schema mismatch"):
        namespace["load_ml_views"](tmp_path)


@pytest.mark.parametrize("split_role", ["validation", "locked_test"])
def test_loaded_roles_reject_wrong_approved_feature_dtype(
    tmp_path: Path, split_role: str
) -> None:
    namespace = ml_benchmark_namespace()
    _write_ml_view_fixture(tmp_path)
    frame = pd.read_parquet(tmp_path / f"{split_role}.parquet")
    frame["motor_activation__mean_5s"] = "not-numeric"
    _rewrite_ml_role_fixture(tmp_path, split_role, frame)

    with pytest.raises(ValueError, match="numeric or bool"):
        namespace["load_ml_views"](tmp_path)


@pytest.mark.parametrize("split_role", ["validation", "locked_test"])
def test_loaded_roles_reject_nonfinite_approved_features(
    tmp_path: Path, split_role: str
) -> None:
    namespace = ml_benchmark_namespace()
    _write_ml_view_fixture(tmp_path)
    frame = pd.read_parquet(tmp_path / f"{split_role}.parquet")
    frame.loc[frame.index[0], "autonomic_arousal__robust_z"] = float("inf")
    _rewrite_ml_role_fixture(tmp_path, split_role, frame)

    with pytest.raises(ValueError, match="non-finite"):
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
            "context__sleep": [value % 7 == 0 for value in range(rows)],
            "time_cos": [float(value % 24) for value in range(rows)],
            "motor_activation__mean_5s": [float(value // 5) for value in range(rows)],
            "autonomic_arousal__robust_z": [float(value % 5) for value in range(rows)],
            "is_awake": [value % 8 != 0 for value in range(rows)],
            "pattern_binary": [int(value % 6 != 0) for value in range(rows)],
            "event_binary": [int(value % 5 != 0) for value in range(rows)],
            "hard_negative": [int(value % 6 == 0) for value in range(rows)],
            "forecast_60s": [int(value % 9 == 0) for value in range(rows)],
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

    assert features == [
        "autonomic_arousal__robust_z",
        "motor_activation__mean_5s",
        "time_cos",
        "is_awake",
        "context__sleep",
    ]
    assert all(candidate["pattern_model"].n_features_in_ == 5 for candidate in candidates)


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

    with pytest.raises(ValueError, match="unapproved columns"):
        namespace["_infer_feature_columns"](frame)


@pytest.mark.parametrize("column", ["participant_index", "row_index", "future_label"])
def test_causal_feature_allowlist_rejects_numeric_adversarial_columns(column: str) -> None:
    namespace = ml_benchmark_namespace()
    frame = pd.DataFrame(
        {
            "autonomic_arousal__robust_z": [0.1, 0.2],
            column: [1, 2],
        }
    )

    with pytest.raises(ValueError, match="unapproved columns"):
        namespace["_infer_feature_columns"](frame)


def test_feature_matrix_rejects_nonfinite_numeric_values() -> None:
    namespace = ml_benchmark_namespace()
    frame = pd.DataFrame({"autonomic_arousal__robust_z": [0.0, float("inf")]})

    with pytest.raises(ValueError, match="non-finite"):
        namespace["_feature_matrix"](frame, ["autonomic_arousal__robust_z"])


def test_numeric_feature_contract_rejects_empty_feature_set() -> None:
    namespace = ml_benchmark_namespace()
    frame = pd.DataFrame(
        {
            "person_key": ["P1"],
            "canonical_time": [0],
            "pattern_binary": [0],
            "event_binary": [0],
            "hard_negative": [0],
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
            "autonomic_arousal__robust_z": [float(value % 5) for value in range(60)],
            "motor_activation__mean_5s": [float(value // 5) for value in range(60)],
            "pattern_binary": [int(value % 6 != 0) for value in range(60)],
            "event_binary": [0] * 60,
            "hard_negative": [int(value % 6 == 0) for value in range(60)],
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
        feature_columns = [
            "autonomic_arousal__robust_z",
            "motor_activation__mean_5s",
        ]
        candidate = factory(frame, feature_columns)
        assert set(candidate) == {
            "behavior_models",
            "model_name",
            "pattern_model",
            "stage_models",
        }
        assert set(candidate["stage_models"]) == set(namespace["STAGE_CODES"])
        assert set(candidate["behavior_models"]) == set(namespace["BEHAVIOR_CODES"])
        probability = candidate["pattern_model"].predict_proba(
            namespace["_feature_matrix"](frame, feature_columns)
        )[:, 1]
        assert probability.shape == (len(frame),)
        assert ((probability >= 0) & (probability <= 1)).all()


def test_ml_training_masks_keep_pattern_onset_and_behavior_semantics_distinct() -> None:
    namespace = ml_benchmark_namespace()
    frame = pd.DataFrame(
        {
            "row_kind": [
                "low_pre_onset",
                "onset_audit",
                "hard_negative_positive",
                "hard_negative_negative",
                "baseline",
            ],
            "pattern_binary": [1, 0, 0, 0, 0],
            "event_binary": [0, 1, 0, 0, 0],
            "hard_negative": [0, 0, 1, 1, 0],
            "stage_code": ["LOW", "NO_EVENT", "NO_EVENT", "NO_EVENT", "NO_EVENT"],
        }
    )
    for behavior in namespace["BEHAVIOR_CODES"]:
        frame[behavior] = 0
    frame.loc[frame["row_kind"].eq("hard_negative_positive"), "ear_covering"] = 1

    stage_rows = frame.loc[namespace["_select_stage_training_rows"](frame), "row_kind"]
    behavior_rows = frame.loc[namespace["_select_behavior_training_rows"](frame), "row_kind"]

    assert stage_rows.tolist() == ["low_pre_onset"]
    assert behavior_rows.tolist() == ["low_pre_onset", "hard_negative_positive"]
    assert frame.loc[frame["row_kind"].eq("onset_audit"), "event_binary"].item() == 1
    assert frame.loc[frame["row_kind"].eq("onset_audit"), "stage_code"].item() == "NO_EVENT"


def test_behavior_decision_mask_rejects_positive_ordinary_baseline() -> None:
    namespace = ml_benchmark_namespace()
    frame = pd.DataFrame(
        {
            "pattern_binary": [0],
            "hard_negative": [0],
        }
    )
    for behavior in namespace["BEHAVIOR_CODES"]:
        frame[behavior] = 0
    frame.loc[0, "ear_covering"] = 1

    with pytest.raises(ValueError, match="behavior-positive rows must be pattern or hard negative"):
        namespace["_select_behavior_training_rows"](frame)


@pytest.mark.parametrize("dtype", ["Int64", "boolean"])
def test_behavior_decision_mask_rejects_nullable_hard_negative(dtype: str) -> None:
    namespace = ml_benchmark_namespace()
    frame = pd.DataFrame(
        {
            "pattern_binary": [0],
            "hard_negative": pd.Series([pd.NA], dtype=dtype),
        }
    )
    for behavior in namespace["BEHAVIOR_CODES"]:
        frame[behavior] = 0
    frame.loc[0, "ear_covering"] = 1

    with pytest.raises(ValueError, match="hard_negative contains null"):
        namespace["_select_behavior_training_rows"](frame)


@pytest.mark.parametrize("dtype", ["Int64", "boolean"])
def test_behavior_decision_mask_returns_strict_bool_for_valid_nullable_dtype(
    dtype: str,
) -> None:
    namespace = ml_benchmark_namespace()
    frame = pd.DataFrame(
        {
            "pattern_binary": [1, 0, 0, 0],
            "hard_negative": pd.Series([0, 1, 1, 0], dtype=dtype),
        }
    )
    for behavior in namespace["BEHAVIOR_CODES"]:
        frame[behavior] = 0
    frame.loc[1, "ear_covering"] = 1

    mask = namespace["_select_behavior_training_rows"](frame)

    assert mask.dtype == bool
    assert mask.tolist() == [True, True, False, False]


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
            "autonomic_arousal__robust_z": [0.1, 0.2, 0.3, 0.4],
            "pattern_binary": [1, 0, 0, 0],
            "event_binary": [0, 1, 0, 0],
            "hard_negative": [0, 0, 1, 1],
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
        ["autonomic_arousal__robust_z"],
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


def test_dl_sequence_contract() -> None:
    """The DL data notebook indexes only bounded causal windows."""
    path = KAGGLE_DIR / "03_dl_sequence_data.ipynb"
    notebook = load_notebooks()[path]
    source = notebook_source(notebook)

    expected_definitions = (
        "validate_shared_dataset_identity",
        "fit_train_normalization",
        "make_causal_window_index",
        "assert_window_boundaries",
        "sample_training_windows",
        "write_sequence_manifest",
    )
    for definition in expected_definitions:
        assert f"def {definition}(" in source

    assert "SEQUENCE_LENGTHS_SECONDS = (300, 600)" in source
    assert 'SEQUENCE_OUTPUT_ROOT = Path("/kaggle/working/goal15_dl_sequences")' in source
    assert "window_end >= window_start" in source
    assert "person_key" in source
    assert 'split_role"].eq("train")' in source
    assert "RUN_DATA_PREPARATION = False" in source
    assert "if RUN_DATA_PREPARATION:" in source


def _sequence_source_frame() -> pd.DataFrame:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for second in range(8):
        rows.append(
            {
                "person_key": "P1",
                "run_id": "run-1",
                "dataset_id": "dataset-1",
                "session_id": "session-1",
                "day_key": "2026-01-01",
                "context": "focused_task",
                "canonical_time": start + pd.Timedelta(seconds=second),
                "split_role": "train",
                "missing_block": False,
                "feature_a": float(second),
                "pattern_binary": int(second == 3),
                "hard_negative": int(second == 4),
            }
        )
    return pd.DataFrame(rows)


def test_make_causal_window_index_rejects_boundary_and_missing_block_crossings() -> None:
    namespace = dl_sequence_namespace()
    namespace["SEQUENCE_LENGTHS_SECONDS"] = (3,)
    frame = _sequence_source_frame()
    frame.loc[2, "missing_block"] = True
    frame.loc[6, "session_id"] = "session-2"

    index = namespace["make_causal_window_index"](frame, length_seconds=3)

    namespace["assert_window_boundaries"](index)
    assert set(index["prediction_time"]) == {frame.loc[5, "canonical_time"]}
    assert (index["window_end"] == index["prediction_time"]).all()
    assert (
        index["window_start"]
        == index["prediction_time"] - pd.Timedelta(seconds=2)
    ).all()


def test_make_causal_window_index_honors_declared_date_boundaries() -> None:
    namespace = dl_sequence_namespace()
    namespace["SEQUENCE_LENGTHS_SECONDS"] = (3,)
    frame = _sequence_source_frame().drop(columns="day_key")
    frame["date"] = ["2026-01-01"] * 3 + ["2026-01-02"] * 5

    index = namespace["make_causal_window_index"](frame, length_seconds=3)

    assert set(index["prediction_time"]) == {
        frame.loc[2, "canonical_time"],
        frame.loc[5, "canonical_time"],
        frame.loc[6, "canonical_time"],
        frame.loc[7, "canonical_time"],
    }


def test_fit_train_normalization_uses_only_train_people() -> None:
    namespace = dl_sequence_namespace()
    frame = pd.DataFrame(
        {
            "person_key": ["train-a", "train-b", "validation-a", "validation-b"],
            "split_role": ["train", "train", "validation", "validation"],
            "feature_a": [1.0, 3.0, 100.0, 200.0],
        }
    )

    statistics = namespace["fit_train_normalization"](
        frame,
        feature_columns=["feature_a"],
        source_hash="a" * 64,
    )

    assert statistics["fit_split_role"] == "train"
    assert statistics["source_hash"] == "a" * 64
    assert statistics["features"]["feature_a"] == {"median": 2.0, "iqr": 1.0}


def test_sample_training_windows_is_deterministic_and_retains_required_windows() -> None:
    namespace = dl_sequence_namespace()
    namespace["SEQUENCE_LENGTHS_SECONDS"] = (3,)
    frame = _sequence_source_frame()
    frame["pattern_binary"] = 0
    frame["hard_negative"] = 0
    frame.loc[3, "pattern_binary"] = 1
    frame.loc[4, "hard_negative"] = 1
    index = namespace["make_causal_window_index"](frame, length_seconds=3)

    first = namespace["sample_training_windows"](index, baseline_multiplier=3)
    second = namespace["sample_training_windows"](index, baseline_multiplier=3)

    assert first.equals(second)
    assert {"positive_centered", "hard_negative", "matched_baseline"} == set(
        first["sample_type"]
    )
    assert len(first.loc[first["sample_type"] == "matched_baseline"]) == 3
    assert len(first.loc[first["sample_type"] == "positive_centered"]) == 1
    assert len(first.loc[first["sample_type"] == "hard_negative"]) == 1


def test_validate_shared_dataset_identity_rejects_changed_split_hash(tmp_path: Path) -> None:
    namespace = dl_sequence_namespace()
    source_hash, split_hash = _write_flat_identity_fixture(tmp_path)
    manifest = {
        "series_id": "mvp3-oracle-v1",
        "data_status": "oracle/sanity",
        "source_dataset_hash": source_hash,
        "split_hash": split_hash,
    }

    assert namespace["validate_shared_dataset_identity"](manifest, tmp_path) == (
        source_hash,
        split_hash,
    )
    manifest["split_hash"] = "wrong"
    with pytest.raises(ValueError, match="split hash"):
        namespace["validate_shared_dataset_identity"](manifest, tmp_path)


def test_write_sequence_manifest_hashes_indexes_and_train_statistics(tmp_path: Path) -> None:
    namespace = dl_sequence_namespace()
    index_paths = {
        "train_300": tmp_path / "train_300.parquet",
        "validation_300": tmp_path / "validation_300.parquet",
    }
    for name, path in index_paths.items():
        path.write_bytes(name.encode())
    statistics_path = tmp_path / "train_normalization.json"
    statistics_path.write_text('{"fit_split_role":"train"}\n')

    manifest_path = namespace["write_sequence_manifest"](
        tmp_path,
        index_paths=index_paths,
        normalization_path=statistics_path,
        source_dataset_hash="a" * 64,
        split_hash="b" * 64,
        row_counts={"train_300": 4, "validation_300": 6},
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["normalization"]["sha256"] == hashlib.sha256(
        statistics_path.read_bytes()
    ).hexdigest()
    assert manifest["files"]["train_300"]["row_count"] == 4
    assert manifest["files"]["validation_300"]["sha256"] == hashlib.sha256(
        index_paths["validation_300"].read_bytes()
    ).hexdigest()


def _write_flat_identity_fixture(root: Path) -> tuple[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    for name, payload in {
        "prepared__manifest.json": {"series_id": "mvp3-oracle-v1", "kind": "prepared"},
        "outcomes__manifest.json": {"series_id": "mvp3-oracle-v1", "kind": "outcomes"},
        "registry__manifest.json": {"series_id": "mvp3-oracle-v1", "kind": "registry"},
    }.items():
        (root / name).write_text(json.dumps(payload, sort_keys=True))
    split = pd.DataFrame(
        {
            "person_key": [
                *(f"train-{index:02d}" for index in range(24)),
                *(f"validation-{index:02d}" for index in range(6)),
                *(f"locked-{index:02d}" for index in range(6)),
            ],
            "split_role": ["train"] * 24 + ["validation"] * 6 + ["locked_test"] * 6,
        }
    )
    split_path = root / "registry__splits.parquet"
    split.to_parquet(split_path, index=False)
    manifest_hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "prepared__manifest.json",
            "outcomes__manifest.json",
            "registry__manifest.json",
        )
    }
    return (
        hashlib.sha256(json.dumps(manifest_hashes, sort_keys=True).encode()).hexdigest(),
        hashlib.sha256(split_path.read_bytes()).hexdigest(),
    )


def test_shared_identity_recomputes_original_flat_dataset_hashes(tmp_path: Path) -> None:
    namespace = dl_sequence_namespace()
    source_hash, split_hash = _write_flat_identity_fixture(tmp_path)
    view_manifest = {
        "series_id": "mvp3-oracle-v1",
        "data_status": "oracle/sanity",
        "source_dataset_hash": source_hash,
        "split_hash": split_hash,
    }

    assert namespace["validate_shared_dataset_identity"](view_manifest, tmp_path) == (
        source_hash,
        split_hash,
    )
    (tmp_path / "outcomes__manifest.json").write_text(
        json.dumps({"series_id": "mvp3-oracle-v1", "kind": "changed"}, sort_keys=True)
    )
    with pytest.raises(ValueError, match="source dataset hash"):
        namespace["validate_shared_dataset_identity"](view_manifest, tmp_path)


def _strict_sequence_frame(namespace: dict[str, Any]) -> pd.DataFrame:
    row: dict[str, Any] = {
        "person_key": "P1",
        "run_id": "run-1",
        "dataset_id": "dataset-1",
        "canonical_time": pd.Timestamp("2026-01-01T00:00:00Z"),
        "split_role": "train",
        "context": "focused_task",
        "pattern_binary": 0,
        "event_binary": 1,
        "hard_negative": 0,
        "stage_code": "NO_EVENT",
    }
    row.update({behavior: 0 for behavior in namespace["BEHAVIOR_CODES"]})
    row.update({feature: 0.0 for feature in namespace["ALLOWED_FEATURE_COLUMNS"]})
    return pd.DataFrame([row])


def test_sequence_role_contract_requires_full_ordered_causal_features() -> None:
    namespace = dl_sequence_namespace()
    frame = _strict_sequence_frame(namespace)

    assert namespace["validate_sequence_role_frame"](frame, "train") == list(
        namespace["ALLOWED_FEATURE_COLUMNS"]
    )
    with pytest.raises(ValueError, match="missing approved features"):
        namespace["validate_sequence_role_frame"](
            frame.drop(columns=namespace["ALLOWED_FEATURE_COLUMNS"][0]), "train"
        )
    frame["numeric_leak"] = 1.0
    with pytest.raises(ValueError, match="unapproved"):
        namespace["validate_sequence_role_frame"](frame, "train")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda frame, namespace: frame.assign(pattern_binary=1),
        lambda frame, namespace: frame.assign(stage_code="INVALID"),
        lambda frame, namespace: frame.assign(hard_negative=1, pattern_binary=1),
        lambda frame, namespace: frame.assign(ear_covering=1),
        lambda frame, namespace: frame.assign(event_binary=np.nan),
    ],
    ids=("stage-pattern-mismatch", "invalid-stage", "overlap", "ordinary-behavior", "null-binary"),
)
def test_sequence_role_contract_rejects_invalid_labels(
    mutate: Any,
) -> None:
    namespace = dl_sequence_namespace()
    frame = mutate(_strict_sequence_frame(namespace), namespace)

    with pytest.raises(ValueError):
        namespace["validate_sequence_role_frame"](frame, "train")


def test_matched_baseline_sampling_stays_in_person_run_context_stratum() -> None:
    namespace = dl_sequence_namespace()
    index = pd.DataFrame(
        {
            "person_key": ["P1", "P1", "P1", "P2", "P2", "P2"],
            "run_id": ["run"] * 6,
            "dataset_id": ["dataset"] * 6,
            "context": ["sleep", "sleep", "transition", "sleep", "sleep", "sleep"],
            "split_role": ["train"] * 6,
            "pattern_binary": [1, 0, 0, 0, 0, 0],
            "hard_negative": [0] * 6,
            "length_seconds": [300] * 6,
            "prediction_time": pd.date_range("2026-01-01", periods=6, freq="s", tz="UTC"),
            "window_id": [f"window-{index}" for index in range(6)],
        }
    )

    sampled = namespace["sample_training_windows"](index, baseline_multiplier=3)

    assert set(sampled["person_key"]) == {"P1"}
    assert set(sampled["context"]) == {"sleep"}
    assert len(sampled.loc[sampled["sample_type"] == "matched_baseline"]) == 1


def _write_strict_ml_views(
    root: Path,
    namespace: dict[str, Any],
    source_hash: str,
    split_hash: str,
    *,
    include_full_timeline: bool = True,
) -> None:
    files: dict[str, dict[str, Any]] = {}
    role_counts = {"train": 24, "validation": 6, "locked_test": 6}
    for split_role, count in role_counts.items():
        rows = []
        prefix = "locked" if split_role == "locked_test" else split_role
        for index in range(count):
            frame = _strict_sequence_frame(namespace)
            frame["person_key"] = f"{prefix}-{index:02d}"
            frame["split_role"] = split_role
            frame["canonical_time"] = pd.Timestamp("2026-01-01T00:00:00Z")
            rows.append(frame)
        role_frame = pd.concat(rows, ignore_index=True)
        path = root / f"{split_role}.parquet"
        role_frame.to_parquet(path, index=False, compression="zstd", row_group_size=1)
        files[split_role] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "row_count": len(role_frame),
            "columns": list(role_frame.columns),
        }
    full_files = {
        role: {
            **metadata,
            "feature_columns": list(namespace["ALLOWED_FEATURE_COLUMNS"]),
            "dataset_ids": ["dataset-1"],
            "view_kind": "full_causal_timeline",
            "sampled": False,
        }
        for role, metadata in files.items()
    }
    manifest = {
        "series_id": "mvp3-oracle-v1",
        "data_status": "oracle/sanity",
        "source_dataset_hash": source_hash,
        "split_hash": split_hash,
        "files": files,
    }
    if include_full_timeline:
        manifest["dl_timeline_files"] = full_files
    (root / "view_manifest.json").write_text(json.dumps(manifest))


def test_bounded_sequence_build_never_reads_complete_role_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = dl_sequence_namespace()
    flat_root = tmp_path / "flat"
    source_hash, split_hash = _write_flat_identity_fixture(flat_root)
    views_root = tmp_path / "views"
    views_root.mkdir()
    _write_strict_ml_views(views_root, namespace, source_hash, split_hash)
    real_read_parquet = pd.read_parquet

    def reject_complete_role_read(path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if Path(path).parent == views_root:
            raise AssertionError("complete ML role parquet read")
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", reject_complete_role_read)
    output_root = tmp_path / "output"
    manifest_path = namespace["build_all_sequence_indexes"](
        flat_dataset_root=flat_root,
        ml_view_root=views_root,
        output_root=output_root,
    )

    assert manifest_path.is_file()
    assert (output_root / "train_300.parquet").is_file()
    empty_index = namespace["pq"].read_table(output_root / "train_300.parquet")
    assert empty_index.num_rows == 0
    assert empty_index.schema.equals(namespace["sequence_index_arrow_schema"]())


def test_sequence_build_handles_empty_first_chunk_before_later_window(
    tmp_path: Path,
) -> None:
    namespace = dl_sequence_namespace()
    flat_root = tmp_path / "flat"
    source_hash, split_hash = _write_flat_identity_fixture(flat_root)
    views_root = tmp_path / "views"
    views_root.mkdir()
    _write_strict_ml_views(views_root, namespace, source_hash, split_hash)

    first = _strict_sequence_frame(namespace)
    first["person_key"] = "train-00"
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    first["canonical_time"] = start - pd.Timedelta(seconds=2)
    later_rows = []
    for second in range(300):
        row = _strict_sequence_frame(namespace).iloc[0].to_dict()
        row["person_key"] = "train-00"
        row["canonical_time"] = start + pd.Timedelta(seconds=second)
        row["pattern_binary"] = 1
        row["stage_code"] = "LOW"
        later_rows.append(row)
    membership_rows = []
    for index in range(1, 24):
        row = _strict_sequence_frame(namespace)
        row["person_key"] = f"train-{index:02d}"
        membership_rows.append(row)
    train = pd.concat([first, pd.DataFrame(later_rows), *membership_rows], ignore_index=True)
    train_path = views_root / "train.parquet"
    train.to_parquet(train_path, index=False, compression="zstd", row_group_size=1)
    view_manifest_path = views_root / "view_manifest.json"
    view_manifest = json.loads(view_manifest_path.read_text())
    view_manifest["dl_timeline_files"]["train"]["sha256"] = hashlib.sha256(
        train_path.read_bytes()
    ).hexdigest()
    view_manifest["dl_timeline_files"]["train"]["row_count"] = len(train)
    view_manifest_path.write_text(json.dumps(view_manifest))

    output_root = tmp_path / "output"
    namespace["build_all_sequence_indexes"](
        flat_dataset_root=flat_root,
        ml_view_root=views_root,
        output_root=output_root,
    )

    emitted = namespace["pq"].read_table(output_root / "train_300.parquet")
    assert emitted.num_rows == 1
    assert emitted.schema.equals(namespace["sequence_index_arrow_schema"]())


def test_tail_preserves_endpoint_before_later_missing_block_across_row_group_layout() -> None:
    namespace = dl_sequence_namespace()
    namespace["SEQUENCE_LENGTHS_SECONDS"] = (3,)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for second in range(5):
        row = _strict_sequence_frame(namespace).iloc[0].to_dict()
        row["canonical_time"] = start + pd.Timedelta(seconds=second)
        row["missing_block"] = second == 3
        rows.append(row)
    full = pd.DataFrame(rows)

    single_layout = namespace["make_causal_window_index"](full, length_seconds=3)
    split_layout = namespace["_index_current_chunk"](
        full.iloc[:2].copy(), full.iloc[2:].copy(), length_seconds=3
    )

    assert list(single_layout["prediction_time"]) == [start + pd.Timedelta(seconds=2)]
    assert split_layout["window_id"].tolist() == single_layout["window_id"].tolist()


def test_role_chunk_iterator_rejects_noncontiguous_identity_reappearance(tmp_path: Path) -> None:
    namespace = dl_sequence_namespace()
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    groups = []
    for person_key, second in (("P1", 0), ("P2", 0), ("P1", 1)):
        frame = _strict_sequence_frame(namespace)
        frame["person_key"] = person_key
        frame["canonical_time"] = start + pd.Timedelta(seconds=second)
        groups.append(frame)
    path = tmp_path / "train.parquet"
    schema = namespace["pa"].Table.from_pandas(groups[0], preserve_index=False).schema
    writer = namespace["pq"].ParquetWriter(path, schema, compression="zstd")
    try:
        for group in groups:
            table = namespace["pa"].Table.from_pandas(
                group, schema=schema, preserve_index=False
            )
            writer.write_table(table)
    finally:
        writer.close()

    with pytest.raises(ValueError, match="contiguous"):
        list(namespace["_iter_role_person_chunks"](path, "train"))


def _write_row_grouped_sequence_views(
    root: Path,
    namespace: dict[str, Any],
    source_hash: str,
    split_hash: str,
    train_groups: list[pd.DataFrame],
) -> None:
    _write_strict_ml_views(root, namespace, source_hash, split_hash)
    train_path = root / "train.parquet"
    schema = namespace["pa"].Table.from_pandas(train_groups[0], preserve_index=False).schema
    writer = namespace["pq"].ParquetWriter(train_path, schema, compression="zstd")
    try:
        for group in train_groups:
            table = namespace["pa"].Table.from_pandas(
                group, schema=schema, preserve_index=False
            )
            writer.write_table(table)
    finally:
        writer.close()
    manifest_path = root / "view_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dl_timeline_files"]["train"]["sha256"] = hashlib.sha256(
        train_path.read_bytes()
    ).hexdigest()
    manifest["dl_timeline_files"]["train"]["row_count"] = sum(
        len(group) for group in train_groups
    )
    manifest_path.write_text(json.dumps(manifest))


def test_sequence_build_is_row_group_layout_invariant_for_global_train_sampling(
    tmp_path: Path,
) -> None:
    namespace = dl_sequence_namespace()
    namespace["SEQUENCE_LENGTHS_SECONDS"] = (3,)
    flat_root = tmp_path / "flat"
    source_hash, split_hash = _write_flat_identity_fixture(flat_root)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    timeline_rows = []
    for second in range(17):
        row = _strict_sequence_frame(namespace).iloc[0].to_dict()
        row["person_key"] = "train-00"
        row["canonical_time"] = start + pd.Timedelta(seconds=second)
        if second in {3, 6, 9, 12}:
            row["pattern_binary"] = 1
            row["stage_code"] = "LOW"
        timeline_rows.append(row)
    timeline = pd.DataFrame(timeline_rows)
    membership_groups = []
    for index in range(1, 24):
        row = _strict_sequence_frame(namespace)
        row["person_key"] = f"train-{index:02d}"
        membership_groups.append(row)

    whole_root = tmp_path / "whole"
    split_root = tmp_path / "split"
    whole_root.mkdir()
    split_root.mkdir()
    _write_row_grouped_sequence_views(
        whole_root, namespace, source_hash, split_hash, [timeline, *membership_groups]
    )
    _write_row_grouped_sequence_views(
        split_root,
        namespace,
        source_hash,
        split_hash,
        [
            timeline.iloc[:4],
            timeline.iloc[4:7],
            timeline.iloc[7:10],
            timeline.iloc[10:13],
            timeline.iloc[13:],
            *membership_groups,
        ],
    )

    whole_output = tmp_path / "whole-output"
    split_output = tmp_path / "split-output"
    namespace["build_all_sequence_indexes"](
        flat_dataset_root=flat_root, ml_view_root=whole_root, output_root=whole_output
    )
    namespace["build_all_sequence_indexes"](
        flat_dataset_root=flat_root, ml_view_root=split_root, output_root=split_output
    )

    whole = namespace["pq"].read_table(whole_output / "train_3.parquet").to_pandas()
    split = namespace["pq"].read_table(split_output / "train_3.parquet").to_pandas()
    assert len(whole) == 15
    assert whole.sort_values("window_id").reset_index(drop=True).equals(
        split.sort_values("window_id").reset_index(drop=True)
    )


def test_verified_role_views_reject_cross_role_feature_type_mismatch(tmp_path: Path) -> None:
    namespace = dl_sequence_namespace()
    flat_root = tmp_path / "flat"
    source_hash, split_hash = _write_flat_identity_fixture(flat_root)
    views_root = tmp_path / "views"
    views_root.mkdir()
    _write_strict_ml_views(views_root, namespace, source_hash, split_hash)
    feature = namespace["ALLOWED_FEATURE_COLUMNS"][0]
    validation_path = views_root / "validation.parquet"
    validation = pd.read_parquet(validation_path)
    validation[feature] = validation[feature].astype("string")
    validation.to_parquet(validation_path, index=False, compression="zstd", row_group_size=1)
    manifest_path = views_root / "view_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dl_timeline_files"]["validation"]["sha256"] = hashlib.sha256(
        validation_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="feature schema mismatch"):
        namespace["_verified_role_views"](views_root, flat_root)


@pytest.mark.parametrize(
    "column,value",
    [
        ("person_key", 9),
        ("run_id", " run"),
        ("dataset_id", ""),
        ("context", "unapproved"),
    ],
)
def test_sequence_contract_rejects_noncanonical_identity_and_context_domain(
    column: str, value: Any
) -> None:
    namespace = dl_sequence_namespace()
    frame = _strict_sequence_frame(namespace)
    frame[column] = value

    with pytest.raises(ValueError, match=r"identity|context"):
        namespace["validate_sequence_role_frame"](frame, "train")


def test_make_causal_window_index_rejects_invalid_length_time_and_duplicate_id() -> None:
    namespace = dl_sequence_namespace()
    namespace["SEQUENCE_LENGTHS_SECONDS"] = (3,)
    frame = _sequence_source_frame()

    with pytest.raises(ValueError, match="length_seconds"):
        namespace["make_causal_window_index"](frame, length_seconds=4)
    duplicate = frame.copy()
    duplicate.loc[1, "canonical_time"] = duplicate.loc[0, "canonical_time"]
    with pytest.raises(ValueError, match="duplicate"):
        namespace["make_causal_window_index"](duplicate, length_seconds=3)
    naive = frame.copy()
    naive["canonical_time"] = naive["canonical_time"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="UTC"):
        namespace["make_causal_window_index"](naive, length_seconds=3)


def test_memmap_normalization_is_deterministic_and_removes_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = dl_sequence_namespace()
    feature = namespace["ALLOWED_FEATURE_COLUMNS"][0]
    path = tmp_path / "train.parquet"
    pd.DataFrame({feature: [1.0, 3.0, 5.0, 7.0]}).to_parquet(
        path, index=False, row_group_size=2
    )

    def reject_concatenate(*args: Any, **kwargs: Any) -> np.ndarray:
        raise AssertionError("normalization must not concatenate feature chunks")

    monkeypatch.setattr(np, "concatenate", reject_concatenate)
    first = namespace["fit_train_normalization_from_role_file"](
        path,
        feature_columns=[feature],
        source_hash="a" * 64,
        temporary_directory=tmp_path,
    )
    second = namespace["fit_train_normalization_from_role_file"](
        path,
        feature_columns=[feature],
        source_hash="a" * 64,
        temporary_directory=tmp_path,
    )

    assert first == second
    assert first["features"][feature] == {"median": 4.0, "iqr": 3.0}
    assert not list(tmp_path.glob("goal15-normalization-*.mmap"))


def test_row_group_tail_emits_one_600_second_endpoint_without_crossing_boundary() -> None:
    namespace = dl_sequence_namespace()
    namespace["SEQUENCE_LENGTHS_SECONDS"] = (600,)
    base = _strict_sequence_frame(namespace).iloc[0].to_dict()
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for second in range(600):
        row = dict(base)
        row["canonical_time"] = start + pd.Timedelta(seconds=second)
        rows.append(row)
    first = pd.DataFrame(rows[:300])
    second = pd.DataFrame(rows[300:])

    tail = namespace["_combine_contiguous_tail"](pd.DataFrame(), first, max_length=600)
    index = namespace["_index_current_chunk"](tail, second, length_seconds=600)

    assert len(index) == 1
    assert index.loc[0, "window_start"] == start
    assert index.loc[0, "window_end"] == start + pd.Timedelta(seconds=599)
    second.loc[0, "canonical_time"] = start + pd.Timedelta(seconds=601)
    broken = namespace["_index_current_chunk"](tail, second, length_seconds=600)
    assert broken.empty


def test_full_dl_timeline_keeps_audit_columns_and_injects_dataset_id() -> None:
    """Removing the full-view builder or selecting audit columns as features must fail."""
    namespace = ml_data_namespace()
    namespace["EXPECTED_SPLIT_COUNTS"] = {
        "train": 1,
        "validation": 1,
        "locked_test": 1,
    }
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    prepared = pd.DataFrame(
        {
            "person_key": ["P1", "P1"],
            "run_id": ["run-1", "run-1"],
            "person_id": ["P1", "P1"],
            "canonical_time": [timestamp, timestamp + pd.Timedelta(seconds=1)],
            "context": ["focused_task", "focused_task"],
            "event_binary": [0, 1],
            "forecast_60s": [1, 0],
            "phase": ["pre_early", "onset"],
            "autonomic_arousal__robust_z": [0.0, 1.0],
        }
    )
    labels = complete_multitask_labels(
        pd.DataFrame(
            {
                "run_id": ["run-1", "run-1"],
                "person_id": ["P1", "P1"],
                "canonical_time": prepared["canonical_time"],
                "event_binary": [0, 1],
                "hard_negative": [1, 0],
            }
        )
    )
    split = pd.DataFrame({"person_key": ["P1"], "split_role": ["train"]})

    view = namespace["build_full_dl_role_view"](
        prepared,
        labels,
        split,
        "train",
        dataset_id="prepared-person-1",
    )

    assert len(view) == 2
    assert view["dataset_id"].tolist() == ["prepared-person-1"] * 2
    assert view["split_role"].tolist() == ["train"] * 2
    assert view["phase"].tolist() == ["pre_early", "onset"]
    assert view["forecast_60s"].tolist() == [1, 0]
    assert {"phase", "forecast_60s", "event_binary"}.isdisjoint(
        namespace["DL_CAUSAL_FEATURE_COLUMNS"]
    )


def test_prepared_people_are_streamed_in_sequence_identity_order() -> None:
    """Sorting by opaque dataset filename must not make DL identities reappear."""
    namespace = ml_data_namespace()
    manifest = {
        "people": [
            {
                "dataset_id": "001-opaque",
                "person_key": "run-z/P2",
                "run_id": "run-z",
                "person_id": "P2",
            },
            {
                "dataset_id": "999-opaque",
                "person_key": "run-a/P1",
                "run_id": "run-a",
                "person_id": "P1",
            },
        ]
    }

    ordered = namespace["ordered_prepared_entries"](manifest)

    assert [entry["dataset_id"] for entry in ordered] == ["999-opaque", "001-opaque"]


def test_dl_sequence_consumes_only_manifest_full_timeline_entries(tmp_path: Path) -> None:
    """Pointing the DL builder at sampled ML files must fail closed."""
    namespace = dl_sequence_namespace()
    flat_root = tmp_path / "flat"
    source_hash, split_hash = _write_flat_identity_fixture(flat_root)
    views_root = tmp_path / "views"
    views_root.mkdir()
    _write_strict_ml_views(
        views_root,
        namespace,
        source_hash,
        split_hash,
        include_full_timeline=False,
    )

    with pytest.raises(ValueError, match="full causal timeline"):
        namespace["_verified_role_views"](views_root, flat_root)


def test_dl_sequence_allows_audit_columns_but_never_selects_them() -> None:
    """Adding phase/forecast audit data must not change the model feature list."""
    namespace = dl_sequence_namespace()
    frame = _strict_sequence_frame(namespace)
    frame["phase"] = "pre_early"
    frame["forecast_60s"] = 1

    selected = namespace["validate_sequence_role_frame"](frame, "train")

    assert selected == list(namespace["ALLOWED_FEATURE_COLUMNS"])
    assert {"phase", "forecast_60s", "event_binary"}.isdisjoint(selected)


def test_sequence_temp_spools_are_removed_when_role_validation_fails(
    tmp_path: Path,
) -> None:
    """Any exception after raw spool creation must remove Parquet and SQLite temporaries."""
    namespace = dl_sequence_namespace()
    flat_root = tmp_path / "flat"
    source_hash, split_hash = _write_flat_identity_fixture(flat_root)
    views_root = tmp_path / "views"
    views_root.mkdir()
    _write_strict_ml_views(views_root, namespace, source_hash, split_hash)
    manifest_path = views_root / "view_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dl_timeline_files"] = {}
    for role, metadata in manifest["files"].items():
        manifest["dl_timeline_files"][role] = {
            **metadata,
            "view_kind": "full_causal_timeline",
            "sampled": False,
            "feature_columns": list(namespace["ALLOWED_FEATURE_COLUMNS"]),
            "dataset_ids": ["dataset-1"],
        }
    manifest_path.write_text(json.dumps(manifest))
    output_root = tmp_path / "output"

    original_make_index = namespace["make_causal_window_index"]
    calls = 0

    def fail_during_window_write(
        frame: pd.DataFrame, *, length_seconds: int
    ) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected window-write failure")
        return original_make_index(frame, length_seconds=length_seconds)

    namespace["make_causal_window_index"] = fail_during_window_write
    with pytest.raises(RuntimeError, match="injected"):
        namespace["build_all_sequence_indexes"](
            flat_dataset_root=flat_root,
            ml_view_root=views_root,
            output_root=output_root,
        )

    assert not list(output_root.glob("goal15-train*.parquet"))
    assert not list(output_root.glob("goal15-train-sampling-*.sqlite"))


def _write_dual_view_fixture(
    root: Path,
    namespace: dict[str, Any],
    source_hash: str,
    split_hash: str,
    train_groups: list[pd.DataFrame],
) -> None:
    _write_strict_ml_views(root, namespace, source_hash, split_hash)
    manifest_path = root / "view_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    full_files: dict[str, dict[str, Any]] = {}
    for role in ("validation", "locked_test"):
        source_path = root / manifest["files"][role]["path"]
        frame = pd.read_parquet(source_path)
        path = root / f"dl_timeline_{role}.parquet"
        frame.to_parquet(path, index=False, compression="zstd", row_group_size=1)
        full_files[role] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "row_count": len(frame),
            "columns": list(frame.columns),
            "feature_columns": list(namespace["ALLOWED_FEATURE_COLUMNS"]),
            "dataset_ids": ["dataset-1"],
            "view_kind": "full_causal_timeline",
            "sampled": False,
        }
    train_path = root / "dl_timeline_train.parquet"
    schema = namespace["pa"].Table.from_pandas(train_groups[0], preserve_index=False).schema
    writer = namespace["pq"].ParquetWriter(train_path, schema, compression="zstd")
    try:
        for group in train_groups:
            writer.write_table(
                namespace["pa"].Table.from_pandas(
                    group, schema=schema, preserve_index=False
                )
            )
    finally:
        writer.close()
    full_files["train"] = {
        "path": train_path.name,
        "sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "row_count": sum(len(group) for group in train_groups),
        "columns": list(schema.names),
        "feature_columns": list(namespace["ALLOWED_FEATURE_COLUMNS"]),
        "dataset_ids": ["dataset-1"],
        "view_kind": "full_causal_timeline",
        "sampled": False,
    }
    manifest["files"]["train"]["view_kind"] = "sampled_ml_rows"
    manifest["files"]["train"]["sampled"] = True
    manifest["dl_timeline_files"] = full_files
    manifest_path.write_text(json.dumps(manifest))


def test_dual_views_produce_layout_invariant_600s_windows_and_baselines(
    tmp_path: Path,
) -> None:
    """The DL path must use the full view and ignore sampled ML row layout."""
    namespace = dl_sequence_namespace()
    namespace["SEQUENCE_LENGTHS_SECONDS"] = (600,)
    flat_root = tmp_path / "flat"
    source_hash, split_hash = _write_flat_identity_fixture(flat_root)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    timeline_rows = []
    for second in range(604):
        row = _strict_sequence_frame(namespace).iloc[0].to_dict()
        row["person_key"] = "train-00"
        row["canonical_time"] = start + pd.Timedelta(seconds=second)
        row["event_binary"] = int(second == 603)
        row["hard_negative"] = int(second == 600)
        row["pattern_binary"] = int(second == 603)
        row["stage_code"] = "LOW" if second == 603 else "NO_EVENT"
        row["phase"] = "onset" if second == 603 else "none"
        row["forecast_60s"] = int(543 <= second < 603)
        timeline_rows.append(row)
    timeline = pd.DataFrame(timeline_rows)
    membership = []
    for index in range(1, 24):
        row = _strict_sequence_frame(namespace)
        row["person_key"] = f"train-{index:02d}"
        row["phase"] = "none"
        row["forecast_60s"] = 0
        membership.append(row)

    whole_root = tmp_path / "whole"
    split_root = tmp_path / "split"
    whole_root.mkdir()
    split_root.mkdir()
    _write_dual_view_fixture(
        whole_root,
        namespace,
        source_hash,
        split_hash,
        [timeline, *membership],
    )
    _write_dual_view_fixture(
        split_root,
        namespace,
        source_hash,
        split_hash,
        [
            timeline.iloc[:211],
            timeline.iloc[211:487],
            timeline.iloc[487:],
            *membership,
        ],
    )

    whole_output = tmp_path / "whole-output"
    split_output = tmp_path / "split-output"
    namespace["build_all_sequence_indexes"](
        flat_dataset_root=flat_root,
        ml_view_root=whole_root,
        output_root=whole_output,
    )
    namespace["build_all_sequence_indexes"](
        flat_dataset_root=flat_root,
        ml_view_root=split_root,
        output_root=split_output,
    )

    whole = namespace["pq"].read_table(whole_output / "train_600.parquet").to_pandas()
    split = namespace["pq"].read_table(split_output / "train_600.parquet").to_pandas()
    assert len(whole) == 5
    assert set(whole["sample_type"]) == {
        "positive_centered",
        "hard_negative",
        "matched_baseline",
    }
    assert whole.sort_values("window_id").reset_index(drop=True).equals(
        split.sort_values("window_id").reset_index(drop=True)
    )
