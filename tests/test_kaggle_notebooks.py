"""Static safety contracts for the unexecuted Kaggle notebook shells."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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


def test_dl_timeline_manifest_carries_exact_physical_source_groups(
    tmp_path: Path,
) -> None:
    """The DL manifest must preserve person/run/dataset/split row-count proof."""
    namespace = ml_data_namespace()
    view_path = tmp_path / "train.parquet"
    view_path.write_bytes(b"sampled ML rows")
    source_group = {
        "person_key": "train-00",
        "run_id": "run-1",
        "dataset_id": "source-train-00",
        "split_role": "train",
        "row_count": 1,
    }
    timeline = pd.DataFrame(
        {
            "person_key": ["train-00"],
            "run_id": ["run-1"],
            "dataset_id": ["source-train-00"],
            "split_role": ["train"],
            **{feature: [0.0] for feature in namespace["DL_CAUSAL_FEATURE_COLUMNS"]},
        }
    )
    timeline_path = tmp_path / "dl_timeline_train.parquet"
    timeline.to_parquet(timeline_path, index=False)
    source_inventory = [
        {
            **source_group,
            "person_id": "train-00",
            "path": "prepared__people__source-train-00.parquet",
            "sha256": "c" * 64,
            "schema_fingerprint": "d" * 64,
        }
    ]
    inventory_hash = hashlib.sha256(
        json.dumps(source_inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    manifest_path = namespace["write_ml_view_manifest"](
        tmp_path,
        {"train": view_path},
        "a" * 64,
        "b" * 64,
        {"train": 1},
        {"train": ["person_key"]},
        {"train": timeline_path},
        {"train": 1},
        {"train": list(timeline.columns)},
        {"train": {"source-train-00"}},
        source_inventory,
        inventory_hash,
    )

    metadata = json.loads(manifest_path.read_text())["dl_timeline_files"]["train"]
    assert metadata["source_groups"] == [source_group]


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
    frame.loc[6:, "session_id"] = "session-2"

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
        "source_content_inventory": _FLAT_SOURCE_INVENTORIES[source_hash][0],
        "source_content_inventory_hash": _FLAT_SOURCE_INVENTORIES[source_hash][1],
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


_FLAT_SOURCE_INVENTORIES: dict[str, tuple[list[dict[str, Any]], str]] = {}


def _write_flat_identity_fixture(
    root: Path,
    *,
    person_row_counts: dict[str, int] | None = None,
) -> tuple[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    person_row_counts = person_row_counts or {}
    people: list[dict[str, str]] = []
    person_keys = [
        *(f"train-{index:02d}" for index in range(24)),
        *(f"validation-{index:02d}" for index in range(6)),
        *(f"locked-{index:02d}" for index in range(6)),
    ]
    for person_key in person_keys:
        dataset_id = f"source-{person_key}"
        row_count = person_row_counts.get(person_key, 1)
        pd.DataFrame(
            {
                "person_key": [person_key] * row_count,
                "run_id": ["run-1"] * row_count,
                "person_id": [person_key] * row_count,
                "source_value": np.arange(row_count, dtype=np.float64),
            }
        ).to_parquet(root / f"prepared__people__{dataset_id}.parquet", index=False)
        people.append(
            {
                "dataset_id": dataset_id,
                "person_key": person_key,
                "run_id": "run-1",
                "person_id": person_key,
            }
        )
    for name, payload in {
        "prepared__manifest.json": {
            "series_id": "mvp3-oracle-v1",
            "kind": "prepared",
            "people": people,
        },
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
    inventory: list[dict[str, Any]] = []
    for person in sorted(people, key=lambda item: item["dataset_id"]):
        path = root / f"prepared__people__{person['dataset_id']}.parquet"
        parquet = pq.ParquetFile(path)
        schema_hash = hashlib.sha256(
            parquet.schema_arrow.remove_metadata().serialize().to_pybytes()
        ).hexdigest()
        inventory.append(
            {
                **person,
                "split_role": (
                    "train"
                    if person["person_key"].startswith("train-")
                    else "validation"
                    if person["person_key"].startswith("validation-")
                    else "locked_test"
                ),
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "row_count": parquet.metadata.num_rows,
                "schema_fingerprint": schema_hash,
            }
        )
    inventory_hash = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    source_hash = hashlib.sha256(
        json.dumps(
            {
                "manifest_hashes": dict(sorted(manifest_hashes.items())),
                "prepared_source_inventory_hash": inventory_hash,
                "split_sha256": split_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    _FLAT_SOURCE_INVENTORIES[source_hash] = (inventory, inventory_hash)
    return source_hash, split_hash


def test_shared_identity_recomputes_original_flat_dataset_hashes(tmp_path: Path) -> None:
    namespace = dl_sequence_namespace()
    source_hash, split_hash = _write_flat_identity_fixture(tmp_path)
    view_manifest = {
        "series_id": "mvp3-oracle-v1",
        "data_status": "oracle/sanity",
        "source_dataset_hash": source_hash,
        "split_hash": split_hash,
        "source_content_inventory": _FLAT_SOURCE_INVENTORIES[source_hash][0],
        "source_content_inventory_hash": _FLAT_SOURCE_INVENTORIES[source_hash][1],
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
            person_key = f"{prefix}-{index:02d}"
            frame["person_key"] = person_key
            frame["dataset_id"] = f"source-{person_key}"
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
    full_files = {}
    inventory = _FLAT_SOURCE_INVENTORIES[source_hash][0]
    for role, metadata in files.items():
        frame = pd.read_parquet(root / metadata["path"], columns=["dataset_id"])
        source_groups = sorted(
            [
                {
                    "person_key": item["person_key"],
                    "run_id": item["run_id"],
                    "dataset_id": item["dataset_id"],
                    "split_role": item["split_role"],
                    "row_count": item["row_count"],
                }
                for item in inventory
                if item["split_role"] == role
            ],
            key=lambda item: (
                item["person_key"],
                item["run_id"],
                item["dataset_id"],
            ),
        )
        full_files[role] = {
            **metadata,
            "feature_columns": list(namespace["ALLOWED_FEATURE_COLUMNS"]),
            "dataset_ids": sorted(frame["dataset_id"].unique()),
            "view_kind": "full_causal_timeline",
            "sampled": False,
            "source_groups": source_groups,
        }
    manifest = {
        "series_id": "mvp3-oracle-v1",
        "data_status": "oracle/sanity",
        "source_dataset_hash": source_hash,
        "split_hash": split_hash,
        "files": files,
        "source_content_inventory": _FLAT_SOURCE_INVENTORIES[source_hash][0],
        "source_content_inventory_hash": _FLAT_SOURCE_INVENTORIES[source_hash][1],
    }
    if include_full_timeline:
        manifest["dl_timeline_files"] = full_files
    (root / "view_manifest.json").write_text(json.dumps(manifest))


def _bind_fixture_dataset_ids(frame: pd.DataFrame) -> pd.DataFrame:
    bound = frame.copy()
    bound["dataset_id"] = "source-" + bound["person_key"].astype(str)
    return bound


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
    source_hash, split_hash = _write_flat_identity_fixture(
        flat_root, person_row_counts={"train-00": 301}
    )
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
    train = _bind_fixture_dataset_ids(
        pd.concat([first, pd.DataFrame(later_rows), *membership_rows], ignore_index=True)
    )
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
    train_groups = [_bind_fixture_dataset_ids(group) for group in train_groups]
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
    manifest["dl_timeline_files"]["train"]["dataset_ids"] = sorted(
        pd.concat(train_groups, ignore_index=True)["dataset_id"].unique()
    )
    manifest_path.write_text(json.dumps(manifest))


def test_sequence_build_is_row_group_layout_invariant_for_global_train_sampling(
    tmp_path: Path,
) -> None:
    namespace = dl_sequence_namespace()
    namespace["SEQUENCE_LENGTHS_SECONDS"] = (3,)
    flat_root = tmp_path / "flat"
    source_hash, split_hash = _write_flat_identity_fixture(
        flat_root, person_row_counts={"train-00": 17}
    )
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
    second["canonical_time"] = second["canonical_time"] + pd.Timedelta(seconds=301)
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
    namespace["EXPECTED_SPLIT_COUNTS"] = {"train": 2}
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


def _prepared_manifest_preflight_fixture(
    root: Path, mutation: str
) -> tuple[list[dict[str, str]], pd.DataFrame]:
    role_people = {
        "train": [f"train-{index:02d}" for index in range(24)],
        "validation": [f"validation-{index:02d}" for index in range(6)],
        "locked_test": [f"locked-{index:02d}" for index in range(6)],
    }
    split = pd.DataFrame(
        [
            {"person_key": person_key, "split_role": role}
            for role, people in role_people.items()
            for person_key in people
        ]
    )
    entries = [
        {
            "person_key": person_key,
            "run_id": f"run-{person_key}",
            "person_id": f"person-{person_key}",
            "dataset_id": f"source-{person_key}",
        }
        for person_key in split["person_key"]
    ]
    if mutation == "duplicate_entry":
        entries.append(dict(entries[0]))
    elif mutation == "same_person_different_dataset":
        entries[1]["person_key"] = entries[0]["person_key"]
        entries[1]["run_id"] = entries[0]["run_id"]
        entries[1]["person_id"] = entries[0]["person_id"]
    elif mutation == "different_person_same_dataset":
        entries[1]["dataset_id"] = entries[0]["dataset_id"]
    elif mutation == "missing_person":
        entries.pop()
    elif mutation == "extra_person":
        entries.append(
            {
                "person_key": "extra-00",
                "run_id": "run-extra-00",
                "person_id": "person-extra-00",
                "dataset_id": "source-extra-00",
            }
        )
    elif mutation in {"resolved_path_alias", "valid"}:
        pass
    elif mutation == "missing_extra_person":
        entries[-1] = {
            "person_key": "extra-00",
            "run_id": "run-extra-00",
            "person_id": "person-extra-00",
            "dataset_id": "source-extra-00",
        }
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    written: set[str] = set()
    for entry in entries:
        if entry["dataset_id"] in written:
            continue
        written.add(entry["dataset_id"])
        pd.DataFrame(
            {
                "person_key": [entry["person_key"]],
                "run_id": [entry["run_id"]],
                "person_id": [entry["person_id"]],
                "canonical_time": [pd.Timestamp("2026-01-01T00:00:00Z")],
            }
        ).to_parquet(
            root / f"prepared__people__{entry['dataset_id']}.parquet",
            index=False,
        )
    if mutation == "resolved_path_alias":
        alias = root / f"prepared__people__{entries[1]['dataset_id']}.parquet"
        alias.unlink()
        alias.symlink_to(f"prepared__people__{entries[0]['dataset_id']}.parquet")
    pd.DataFrame(split).to_parquet(root / "registry__splits.parquet", index=False)
    (root / "prepared__manifest.json").write_text(json.dumps({"people": entries}))
    (root / "outcomes__manifest.json").write_text("{}")
    (root / "registry__manifest.json").write_text("{}")
    return entries, split


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_entry", r"duplicate prepared manifest (entry|tuple)"),
        (
            "same_person_different_dataset",
            r"duplicate prepared manifest person_key",
        ),
        (
            "different_person_same_dataset",
            r"duplicate prepared manifest (dataset_id|physical source)",
        ),
        ("missing_person", r"prepared manifest must declare exactly 36 entries"),
        ("extra_person", r"prepared manifest must declare exactly 36 entries"),
        (
            "missing_extra_person",
            r"prepared manifest people must exactly match split registry",
        ),
        (
            "resolved_path_alias",
            r"duplicate prepared manifest physical source path",
        ),
    ],
)
@pytest.mark.parametrize("entrypoint", ["ml", "dl"])
def test_prepared_manifest_preflight_fails_before_any_writer_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
    entrypoint: str,
) -> None:
    """Both standalone notebooks must reject the same manifest before outputs."""
    namespace = ml_data_namespace() if entrypoint == "ml" else dl_sequence_namespace()
    _, split = _prepared_manifest_preflight_fixture(tmp_path, mutation)
    output_root = tmp_path / "output"
    writer_calls = 0

    def reject_writer(*args: Any, **kwargs: Any) -> Any:
        nonlocal writer_calls
        writer_calls += 1
        raise AssertionError("writer creation occurred before prepared preflight")

    monkeypatch.setitem(namespace["pq"].__dict__, "ParquetWriter", reject_writer)
    if entrypoint == "ml":
        monkeypatch.setitem(
            namespace,
            "validate_manifest_hashes",
            lambda dataset_root: {
                "prepared__manifest.json": hashlib.sha256(
                    (dataset_root / "prepared__manifest.json").read_bytes()
                ).hexdigest(),
                "outcomes__manifest.json": "b" * 64,
                "registry__manifest.json": "c" * 64,
            },
        )
        monkeypatch.setitem(namespace, "load_split_registry", lambda dataset_root: split)
        monkeypatch.setitem(
            namespace,
            "_load_outcome_labels",
            lambda dataset_root, run_id, person_id: pd.DataFrame(),
        )
        monkeypatch.setitem(
            namespace,
            "build_ml_role_view",
            lambda prepared, labels, split, split_role: pd.DataFrame({"value": [1]}),
        )
        build_function = namespace["build_all_ml_views"]
        build_kwargs = {
            "flat_dataset_root": tmp_path,
            "output_root": output_root,
        }
    else:
        views_root = tmp_path / "views"
        views_root.mkdir()
        (views_root / "view_manifest.json").write_text(
            json.dumps(
                {
                    "series_id": "mvp3-oracle-v1",
                    "data_status": "oracle/sanity",
                    "source_dataset_hash": "a" * 64,
                    "split_hash": "b" * 64,
                    "source_content_inventory": [],
                    "source_content_inventory_hash": "c" * 64,
                }
            )
        )
        build_function = namespace["build_all_sequence_indexes"]
        build_kwargs = {
            "flat_dataset_root": tmp_path,
            "ml_view_root": views_root,
            "output_root": output_root,
        }

    with pytest.raises(ValueError, match=message):
        build_function(**build_kwargs)

    assert writer_calls == 0
    assert not output_root.exists()


@pytest.mark.parametrize("entrypoint", ["ml", "dl"])
@pytest.mark.parametrize("mutation", ["symlink_swap", "stat_change"])
def test_canonical_source_preflight_rejects_postcheck_path_drift_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    mutation: str,
) -> None:
    """Canonical source records must fail closed if path identity later drifts."""
    source_hash, split_hash = _write_flat_identity_fixture(tmp_path)
    namespace = ml_data_namespace() if entrypoint == "ml" else dl_sequence_namespace()
    output_root = tmp_path / "output"
    writer_calls = 0

    def reject_writer(*args: Any, **kwargs: Any) -> Any:
        nonlocal writer_calls
        writer_calls += 1
        raise AssertionError("writer received bytes from a changed prepared source")

    monkeypatch.setitem(namespace["pq"].__dict__, "ParquetWriter", reject_writer)
    if entrypoint == "ml":
        helper_name = "_preflight_prepared_sources"
        real_preflight = namespace[helper_name]
        manifest_hashes = {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in (
                "prepared__manifest.json",
                "outcomes__manifest.json",
                "registry__manifest.json",
            )
        }
        monkeypatch.setitem(
            namespace,
            "validate_manifest_hashes",
            lambda dataset_root: manifest_hashes,
        )
        build_function = namespace["build_all_ml_views"]
        build_kwargs = {
            "flat_dataset_root": tmp_path,
            "output_root": output_root,
        }
        source_index = 3
    else:
        helper_name = "_preflight_flat_dataset_identity"
        real_preflight = namespace[helper_name]
        views_root = tmp_path / "views"
        views_root.mkdir()
        inventory, inventory_hash = _FLAT_SOURCE_INVENTORIES[source_hash]
        (views_root / "view_manifest.json").write_text(
            json.dumps(
                {
                    "series_id": "mvp3-oracle-v1",
                    "data_status": "oracle/sanity",
                    "source_dataset_hash": source_hash,
                    "split_hash": split_hash,
                    "source_content_inventory": inventory,
                    "source_content_inventory_hash": inventory_hash,
                }
            )
        )
        build_function = namespace["build_all_sequence_indexes"]
        build_kwargs = {
            "flat_dataset_root": tmp_path,
            "ml_view_root": views_root,
            "output_root": output_root,
        }
        source_index = 2

    def mutate_after_preflight(*args: Any, **kwargs: Any) -> Any:
        result = real_preflight(*args, **kwargs)
        sources = result[source_index]
        source = sources[0]
        if mutation == "symlink_swap":
            source.declared_path.unlink()
            source.declared_path.symlink_to(sources[1].canonical_path)
        else:
            source.canonical_path.write_bytes(
                source.canonical_path.read_bytes() + b"changed-after-preflight"
            )
        return result

    monkeypatch.setitem(namespace, helper_name, mutate_after_preflight)

    with pytest.raises(ValueError, match="changed after preflight"):
        build_function(**build_kwargs)

    assert writer_calls == 0
    assert not output_root.exists() or not list(output_root.iterdir())


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
    train_groups = [_bind_fixture_dataset_ids(group) for group in train_groups]
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
            "dataset_ids": sorted(frame["dataset_id"].unique()),
            "view_kind": "full_causal_timeline",
            "sampled": False,
            "source_groups": manifest["dl_timeline_files"][role][
                "source_groups"
            ],
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
        "dataset_ids": sorted(
            pd.concat(train_groups, ignore_index=True)["dataset_id"].unique()
        ),
        "view_kind": "full_causal_timeline",
        "sampled": False,
        "source_groups": sorted(
            [
                {
                    "person_key": item["person_key"],
                    "run_id": item["run_id"],
                    "dataset_id": item["dataset_id"],
                    "split_role": item["split_role"],
                    "row_count": item["row_count"],
                }
                for item in manifest["source_content_inventory"]
                if item["split_role"] == "train"
            ],
            key=lambda item: (
                item["person_key"],
                item["run_id"],
                item["dataset_id"],
            ),
        ),
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
    source_hash, split_hash = _write_flat_identity_fixture(
        flat_root, person_row_counts={"train-00": 604}
    )
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


def _write_physical_prepared_fixture(root: Path) -> list[dict[str, str]]:
    people: list[dict[str, str]] = []
    for index, person_key in enumerate(("train-00", "validation-00", "locked-00")):
        dataset_id = f"source-{index}"
        path = root / f"prepared__people__{dataset_id}.parquet"
        pd.DataFrame(
            {
                "person_key": [person_key],
                "run_id": [f"run-{index}"],
                "person_id": [f"P{index}"],
                "value": [float(index)],
            }
        ).to_parquet(path, index=False)
        people.append(
            {
                "person_key": person_key,
                "run_id": f"run-{index}",
                "person_id": f"P{index}",
                "dataset_id": dataset_id,
            }
        )
    return people


def test_source_dataset_identity_binds_physical_prepared_parquet_bytes(
    tmp_path: Path,
) -> None:
    """Changing prepared bytes must change the source identity without trusting logical_hash."""
    namespace = ml_data_namespace()
    namespace["EXPECTED_SPLIT_COUNTS"] = {
        "train": 1,
        "validation": 1,
        "locked_test": 1,
    }
    people = _write_physical_prepared_fixture(tmp_path)
    manifest_hashes = {
        "prepared__manifest.json": "a" * 64,
        "outcomes__manifest.json": "b" * 64,
        "registry__manifest.json": "c" * 64,
    }
    split_path = tmp_path / "registry__splits.parquet"
    pd.DataFrame(
        {
            "person_key": ["train-00", "validation-00", "locked-00"],
            "split_role": ["train", "validation", "locked_test"],
        }
    ).to_parquet(split_path, index=False)

    first_hash, first_inventory, first_inventory_hash = namespace[
        "derive_source_dataset_identity"
    ](tmp_path, manifest_hashes, people)
    changed_path = tmp_path / "prepared__people__source-0.parquet"
    changed = pd.read_parquet(changed_path)
    changed["value"] = 99.0
    changed.to_parquet(changed_path, index=False)
    second_hash, second_inventory, second_inventory_hash = namespace[
        "derive_source_dataset_identity"
    ](tmp_path, manifest_hashes, people)

    assert first_hash != second_hash
    assert first_inventory_hash != second_inventory_hash
    assert first_inventory[0]["sha256"] != second_inventory[0]["sha256"]
    assert first_inventory[0]["row_count"] == second_inventory[0]["row_count"] == 1
    assert first_inventory[0]["schema_fingerprint"] == second_inventory[0][
        "schema_fingerprint"
    ]


def test_source_dataset_identity_changes_for_row_count_and_schema_mutation(
    tmp_path: Path,
) -> None:
    """Prepared row-count and schema changes cannot retain a prior source identity."""
    namespace = ml_data_namespace()
    namespace["EXPECTED_SPLIT_COUNTS"] = {
        "train": 1,
        "validation": 1,
        "locked_test": 1,
    }
    people = _write_physical_prepared_fixture(tmp_path)
    manifest_hashes = {
        "prepared__manifest.json": "a" * 64,
        "outcomes__manifest.json": "b" * 64,
        "registry__manifest.json": "c" * 64,
    }
    pd.DataFrame(
        {
            "person_key": ["train-00", "validation-00", "locked-00"],
            "split_role": ["train", "validation", "locked_test"],
        }
    ).to_parquet(tmp_path / "registry__splits.parquet", index=False)
    original_hash, original_inventory, _ = namespace["derive_source_dataset_identity"](
        tmp_path, manifest_hashes, people
    )
    path = tmp_path / "prepared__people__source-0.parquet"
    changed = pd.read_parquet(path)
    changed = pd.concat([changed, changed], ignore_index=True)
    changed["new_column"] = 1
    changed.to_parquet(path, index=False)

    changed_hash, changed_inventory, _ = namespace["derive_source_dataset_identity"](
        tmp_path, manifest_hashes, people
    )

    assert changed_hash != original_hash
    assert changed_inventory[0]["row_count"] == 2
    assert changed_inventory[0]["row_count"] != original_inventory[0]["row_count"]
    assert changed_inventory[0]["schema_fingerprint"] != original_inventory[0][
        "schema_fingerprint"
    ]


def test_source_inventory_rejects_manifest_run_id_that_differs_from_physical_rows(
    tmp_path: Path,
) -> None:
    """A logical registry run_id cannot override physical prepared row identity."""
    namespace = ml_data_namespace()
    namespace["EXPECTED_SPLIT_COUNTS"] = {
        "train": 1,
        "validation": 1,
        "locked_test": 1,
    }
    people = _write_physical_prepared_fixture(tmp_path)
    pd.DataFrame(
        {
            "person_key": ["train-00", "validation-00", "locked-00"],
            "split_role": ["train", "validation", "locked_test"],
        }
    ).to_parquet(tmp_path / "registry__splits.parquet", index=False)
    path = tmp_path / "prepared__people__source-0.parquet"
    changed = pd.read_parquet(path)
    changed["run_id"] = "physical-other-run"
    changed.to_parquet(path, index=False)

    with pytest.raises(ValueError, match=r"physical|run_id|identity"):
        namespace["derive_source_dataset_identity"](
            tmp_path,
            {
                "prepared__manifest.json": "a" * 64,
                "outcomes__manifest.json": "b" * 64,
                "registry__manifest.json": "c" * 64,
            },
            people,
        )


def test_verified_full_timeline_rejects_physical_row_count_mismatch(
    tmp_path: Path,
) -> None:
    """Manifest flags cannot hide a truncated full timeline."""
    namespace = dl_sequence_namespace()
    flat_root = tmp_path / "flat"
    source_hash, split_hash = _write_flat_identity_fixture(flat_root)
    views_root = tmp_path / "views"
    views_root.mkdir()
    _write_strict_ml_views(views_root, namespace, source_hash, split_hash)
    manifest_path = views_root / "view_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dl_timeline_files"]["train"]["row_count"] += 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="row count"):
        namespace["_verified_role_views"](views_root, flat_root)


def test_verified_full_timeline_rejects_source_binding_swap(
    tmp_path: Path,
) -> None:
    """Preserving row counts and ID sets cannot hide person-to-source swaps."""
    namespace = dl_sequence_namespace()
    flat_root = tmp_path / "flat"
    source_hash, split_hash = _write_flat_identity_fixture(flat_root)
    views_root = tmp_path / "views"
    views_root.mkdir()
    _write_strict_ml_views(views_root, namespace, source_hash, split_hash)
    manifest_path = views_root / "view_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    train_path = views_root / manifest["dl_timeline_files"]["train"]["path"]
    train = pd.read_parquet(train_path)
    first_id, second_id = train.loc[0, "dataset_id"], train.loc[1, "dataset_id"]
    train.loc[0, "dataset_id"] = second_id
    train.loc[1, "dataset_id"] = first_id
    train.to_parquet(train_path, index=False, row_group_size=1)
    manifest["dl_timeline_files"]["train"]["sha256"] = hashlib.sha256(
        train_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="binding"):
        namespace["_verified_role_views"](views_root, flat_root)


@pytest.mark.parametrize(
    ("column", "value"),
    [("run_id", "forged-run"), ("split_role", "validation")],
)
def test_verified_full_timeline_rejects_run_or_split_mutation(
    tmp_path: Path, column: str, value: str
) -> None:
    """Updating only timeline bytes/manifest cannot forge run or split provenance."""
    namespace = dl_sequence_namespace()
    flat_root = tmp_path / "flat"
    source_hash, split_hash = _write_flat_identity_fixture(flat_root)
    views_root = tmp_path / "views"
    views_root.mkdir()
    _write_strict_ml_views(views_root, namespace, source_hash, split_hash)
    manifest_path = views_root / "view_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    train_path = views_root / manifest["dl_timeline_files"]["train"]["path"]
    train = pd.read_parquet(train_path)
    train.loc[0, column] = value
    train.to_parquet(train_path, index=False, row_group_size=1)
    manifest["dl_timeline_files"]["train"]["sha256"] = hashlib.sha256(
        train_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=r"binding|split"):
        namespace["_verified_role_views"](views_root, flat_root)


def test_make_causal_window_index_is_vectorized_and_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A causal index must not perform a Series.diff scan once per endpoint."""
    namespace = dl_sequence_namespace()
    namespace["SEQUENCE_LENGTHS_SECONDS"] = (3,)
    frame = _sequence_source_frame()
    frame.loc[2, "missing_block"] = True
    group_keys = namespace["_window_group_keys"](frame)
    expected_times: set[pd.Timestamp] = set()
    run_length = 0
    previous: pd.Series | None = None
    for _, row in frame.iterrows():
        missing = bool(row["missing_block"])
        contiguous = (
            previous is not None
            and not bool(previous["missing_block"])
            and all(row[key] == previous[key] for key in group_keys)
            and row["canonical_time"] - previous["canonical_time"]
            == pd.Timedelta(seconds=1)
        )
        run_length = 0 if missing else run_length + 1 if contiguous else 1
        if run_length >= 3:
            expected_times.add(row["canonical_time"])
        previous = row

    def reject_series_diff(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("endpoint window scans must not call Series.diff")

    monkeypatch.setattr(pd.Series, "diff", reject_series_diff)
    index = namespace["make_causal_window_index"](frame, length_seconds=3)

    assert set(index["prediction_time"]) == expected_times


def test_make_causal_window_index_has_no_sorting_calls() -> None:
    """Reintroducing O(N log N) sorting inside causal indexing must fail."""
    notebook = load_notebooks()[KAGGLE_DIR / "03_dl_sequence_data.ipynb"]
    tree = ast.parse(code_cell_source(notebook))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "make_causal_window_index"
    )
    forbidden = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "sort_values",
            "argsort",
        }:
            forbidden.append(node.func.attr)
        if isinstance(node.func, ast.Name) and node.func.id == "sorted":
            forbidden.append(node.func.id)
    assert forbidden == []


def test_make_causal_window_index_rejects_unsorted_input() -> None:
    """Input order is part of the bounded streaming contract and cannot be repaired."""
    namespace = dl_sequence_namespace()
    namespace["SEQUENCE_LENGTHS_SECONDS"] = (3,)
    frame = _sequence_source_frame()
    frame.loc[[0, 1], "canonical_time"] = frame.loc[
        [1, 0], "canonical_time"
    ].to_numpy()

    with pytest.raises(ValueError, match=r"ordered|monotonic"):
        namespace["make_causal_window_index"](frame, length_seconds=3)


def test_train_sampler_removes_sqlite_when_connect_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A database path created before sqlite connect must still be removed."""
    namespace = dl_sequence_namespace()

    def fail_connect(path: Any) -> Any:
        raise RuntimeError("injected connect failure")

    monkeypatch.setattr(namespace["sqlite3"], "connect", fail_connect)
    with pytest.raises(RuntimeError, match="connect"):
        namespace["_sample_train_index_file"](
            tmp_path / "raw.parquet",
            tmp_path / "output.parquet",
            temporary_directory=tmp_path,
        )

    assert not list(tmp_path.glob("goal15-train-sampling-*.sqlite"))
    assert not (tmp_path / "output.parquet").exists()


def test_train_sampler_cleanup_continues_after_connection_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Connection close failure must not prevent sqlite temp unlink."""
    namespace = dl_sequence_namespace()
    empty = namespace["_sequence_index_table"](pd.DataFrame())
    raw_path = tmp_path / "raw.parquet"
    namespace["pq"].write_table(empty, raw_path)
    real_connect = namespace["sqlite3"].connect

    class CloseFailingConnection:
        def __init__(self, path: Path) -> None:
            self._connection = real_connect(path)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

        def close(self) -> None:
            self._connection.close()
            raise RuntimeError("injected connection close failure")

    monkeypatch.setattr(
        namespace["sqlite3"], "connect", lambda path: CloseFailingConnection(path)
    )
    with pytest.raises(RuntimeError, match="connection close"):
        namespace["_sample_train_index_file"](
            raw_path,
            tmp_path / "output.parquet",
            temporary_directory=tmp_path,
        )

    assert not list(tmp_path.glob("goal15-train-sampling-*.sqlite"))
    assert not (tmp_path / "output.parquet").exists()


@pytest.mark.parametrize("failure_stage", ["create", "write"])
def test_train_sampler_cleanup_handles_writer_creation_and_write_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    """Writer setup/write failures must close later resources and remove partial output."""
    namespace = dl_sequence_namespace()
    empty = namespace["_sequence_index_table"](pd.DataFrame())
    raw_path = tmp_path / "raw.parquet"
    namespace["pq"].write_table(empty, raw_path)
    real_writer = namespace["pq"].ParquetWriter

    class StageFailingWriter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if failure_stage == "create":
                raise RuntimeError("injected writer create failure")
            self._writer = real_writer(*args, **kwargs)

        def write_table(self, table: Any) -> None:
            if failure_stage == "write":
                raise RuntimeError("injected writer write failure")
            self._writer.write_table(table)

        def close(self) -> None:
            self._writer.close()

    monkeypatch.setattr(namespace["pq"], "ParquetWriter", StageFailingWriter)
    with pytest.raises(RuntimeError, match=failure_stage):
        namespace["_sample_train_index_file"](
            raw_path,
            tmp_path / "output.parquet",
            temporary_directory=tmp_path,
        )

    assert not list(tmp_path.glob("goal15-train-sampling-*.sqlite"))
    assert not (tmp_path / "output.parquet").exists()


def test_sequence_build_unlinks_raw_spools_after_outer_writer_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build-level writer close failure must not strand raw train spools."""
    namespace = dl_sequence_namespace()
    flat_root = tmp_path / "flat"
    source_hash, split_hash = _write_flat_identity_fixture(flat_root)
    views_root = tmp_path / "views"
    views_root.mkdir()
    _write_strict_ml_views(views_root, namespace, source_hash, split_hash)
    output_root = tmp_path / "output"
    real_writer = namespace["pq"].ParquetWriter

    class CloseFailingWriter:
        failed = False

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._writer = real_writer(*args, **kwargs)

        def write_table(self, table: Any) -> None:
            self._writer.write_table(table)

        def close(self) -> None:
            self._writer.close()
            if not CloseFailingWriter.failed:
                CloseFailingWriter.failed = True
                raise RuntimeError("injected build writer close failure")

    monkeypatch.setattr(namespace["pq"], "ParquetWriter", CloseFailingWriter)
    with pytest.raises(RuntimeError, match="build writer close"):
        namespace["build_all_sequence_indexes"](
            flat_dataset_root=flat_root,
            ml_view_root=views_root,
            output_root=output_root,
        )

    assert not list(output_root.glob("goal15-train_*.parquet"))


def test_train_sampler_cleanup_continues_after_writer_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writer close failure must not prevent database close or temp unlink."""
    namespace = dl_sequence_namespace()
    empty = namespace["_sequence_index_table"](pd.DataFrame())
    raw_path = tmp_path / "raw.parquet"
    namespace["pq"].write_table(empty, raw_path)
    real_writer = namespace["pq"].ParquetWriter

    class CloseFailingWriter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._writer = real_writer(*args, **kwargs)

        def write_table(self, table: Any) -> None:
            self._writer.write_table(table)

        def close(self) -> None:
            self._writer.close()
            raise RuntimeError("injected writer close failure")

    monkeypatch.setattr(namespace["pq"], "ParquetWriter", CloseFailingWriter)
    with pytest.raises(RuntimeError, match="writer close"):
        namespace["_sample_train_index_file"](
            raw_path,
            tmp_path / "output.parquet",
            temporary_directory=tmp_path,
        )

    assert not list(tmp_path.glob("goal15-train-sampling-*.sqlite"))
    assert not (tmp_path / "output.parquet").exists()


def _dl_tcn_ast() -> tuple[str, ast.Module]:
    path = KAGGLE_DIR / "04_dl_tcn_benchmark.ipynb"
    notebook = load_notebooks()[path]
    source = code_cell_source(notebook)
    return source, ast.parse(source, filename=str(path))


def _named_definition(tree: ast.Module, name: str) -> ast.FunctionDef | ast.ClassDef:
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    return definitions[name]


def _called_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        function = child.func
        if isinstance(function, ast.Name):
            names.append(function.id)
        elif isinstance(function, ast.Attribute):
            names.append(function.attr)
    return names


def test_dl_tcn_contract() -> None:
    """Removing any required safe TCN entrypoint breaks the notebook contract."""
    source, tree = _dl_tcn_ast()
    expected_definitions = (
        "verify_sequence_inputs",
        "require_exactly_two_cuda_devices",
        "setup_ddp",
        "cleanup_ddp",
        "CausalConvBlock",
        "Goal15TCN",
        "Goal15SequenceDataset",
        "masked_multitask_loss",
        "train_one_epoch",
        "evaluate_common_schema",
        "select_validation_threshold",
        "compute_metrics_from_predictions",
        "login_wandb_from_kaggle_secret",
        "select_validation_champion",
        "run_dl_training",
    )
    for definition in expected_definitions:
        assert _named_definition(tree, definition)

    assert "RUN_TRAINING = False" in source
    assert "RUN_LOCKED_TEST = False" in source
    assert 'PATTERN_TARGET = "pattern_binary"' in source
    assert 'ONSET_EVENT_TARGET = "event_binary"' in source
    assert 'SEQUENCE_OUTPUT_ROOT = Path("/kaggle/working/goal15_dl_sequences")' in source
    assert source.count('UserSecretsClient().get_secret("WANDB_API_KEY")') == 1
    assert source.count("WANDB_API_KEY") == 1


def test_dl_tcn_verifies_sequence_hashes_schema_and_split_membership() -> None:
    """A sampled, changed, or cross-split sequence input must fail before training."""
    source, tree = _dl_tcn_ast()
    verify = _named_definition(tree, "verify_sequence_inputs")
    verify_source = ast.get_source_segment(source, verify) or ""
    declared_verify_source = "\n".join(
        ast.get_source_segment(source, node) or ""
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "verify_sequence_inputs"
    )
    physical_identity_source = ast.get_source_segment(
        source, _named_definition(tree, "recompute_physical_dataset_identity")
    ) or ""
    verify_contract_source = verify_source + declared_verify_source

    assert "sha256_file" in declared_verify_source + physical_identity_source
    assert "source_dataset_hash" in verify_contract_source
    assert "split_hash" in verify_contract_source
    assert "normalization" in verify_contract_source
    assert "ALLOWED_FEATURE_COLUMNS" in verify_contract_source
    assert "EXPECTED_SPLIT_COUNTS" in verify_contract_source
    assert "person leakage" in verify_contract_source
    assert "sampled" in verify_contract_source
    assert "full_causal_timeline" in verify_contract_source
    assert "row_count" in verify_contract_source
    assert "schema" in verify_contract_source
    index_verifier = ast.get_source_segment(
        source, _named_definition(tree, "_verify_sequence_index")
    ) or ""
    assert "expected_length" in index_verifier
    assert "eq(expected_length)" in index_verifier
    assert "pattern and hard_negative cannot overlap" in index_verifier
    assert "behavior-positive ordinary baseline" in index_verifier
    assert "deterministic window_id mismatch" in index_verifier
    assert "timezone-aware UTC" in index_verifier
    assert "expected_feature_types" in verify_contract_source


def test_dl_tcn_architecture_is_causal_masked_and_within_parameter_budget() -> None:
    """Symmetric padding, unmasked pooling, or a model outside 0.5-5M is rejected."""
    source, tree = _dl_tcn_ast()
    causal = _named_definition(tree, "CausalConvBlock")
    model = _named_definition(tree, "Goal15TCN")
    budget = _named_definition(tree, "assert_parameter_budget")
    causal_source = ast.get_source_segment(source, causal) or ""
    model_source = ast.get_source_segment(source, model) or ""
    budget_source = ast.get_source_segment(source, budget) or ""

    assert "F.pad" in causal_source
    assert "(self.left_padding, 0)" in causal_source
    assert "padding=0" in causal_source
    assert "self.event_head = nn.Linear(hidden_size, 1)" in model_source
    assert "self.stage_head = nn.Linear(hidden_size, 5)" in model_source
    assert "self.behavior_head = nn.Linear(hidden_size, 10)" in model_source
    assert "mask" in model_source
    assert "sum" in model_source
    assert "500_000" in budget_source
    assert "5_000_000" in budget_source


def test_dl_tcn_loss_uses_pattern_and_conditional_stage_behavior_masks() -> None:
    """The onset-only audit label cannot replace the pattern target or decision masks."""
    source, tree = _dl_tcn_ast()
    loss = _named_definition(tree, "masked_multitask_loss")
    loss_source = ast.get_source_segment(source, loss) or ""

    assert "PATTERN_TARGET" in loss_source
    assert "ONSET_EVENT_TARGET" not in loss_source
    assert "stage_mask" in loss_source
    assert "behavior_mask" in loss_source
    assert "hard_negative" in loss_source
    assert "behavior_positive" in loss_source
    assert "binary_cross_entropy_with_logits" in loss_source
    assert "cross_entropy" in loss_source


def test_dl_tcn_dataset_is_lazy_bounded_and_train_normalized() -> None:
    """Dataset construction cannot load full timelines or fit normalization on eval roles."""
    source, tree = _dl_tcn_ast()
    dataset = _named_definition(tree, "Goal15SequenceDataset")
    dataset_source = ast.get_source_segment(source, dataset) or ""

    assert "pq.ParquetFile" in dataset_source
    assert "read_row_group" in dataset_source
    assert "pd.read_parquet" not in dataset_source
    assert "fit_split_role" in dataset_source
    assert "'train'" in dataset_source
    assert "ALLOWED_FEATURE_COLUMNS" in dataset_source
    assert "window_start" in dataset_source
    assert "window_end" in dataset_source


def test_dl_tcn_gpu_gate_precedes_ddp_wandb_and_data_construction() -> None:
    """A non-T4x2 runtime must fail before DDP, W&B, loaders, or training exist."""
    source, tree = _dl_tcn_ast()
    gate = _named_definition(tree, "require_exactly_two_cuda_devices")
    gate_source = ast.get_source_segment(source, gate) or ""
    runner = _named_definition(tree, "run_dl_training")
    calls = _called_names(runner)

    assert "device_count" in gate_source
    assert "!= 2" in gate_source
    assert "raise RuntimeError" in gate_source
    gate_index = calls.index("require_exactly_two_cuda_devices")
    for later_call in (
        "setup_ddp",
        "login_wandb_from_kaggle_secret",
        "Goal15SequenceDataset",
        "DataLoader",
        "train_one_epoch",
    ):
        assert gate_index < calls.index(later_call)


def test_dl_tcn_uses_torchrun_ddp_amp_and_deterministic_sampler() -> None:
    """Dropping one-process-per-GPU DDP or epoch-aware sampling is unsafe."""
    source, tree = _dl_tcn_ast()
    setup_source = ast.get_source_segment(
        source, _named_definition(tree, "setup_ddp")
    ) or ""
    train_source = ast.get_source_segment(
        source, _named_definition(tree, "train_one_epoch")
    ) or ""
    runner_source = ast.get_source_segment(
        source, _named_definition(tree, "run_dl_training")
    ) or ""

    assert all(name in setup_source for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "nccl"))
    assert "DistributedDataParallel" in runner_source
    assert "PersonBlockBatchSampler" in runner_source
    assert "DistributedSampler(" not in runner_source
    assert "set_epoch" in train_source
    assert "autocast" in train_source
    assert "GradScaler" in runner_source
    assert "clip_grad_norm_" in train_source
    assert "rank == 0" in runner_source


def test_dl_tcn_training_entrypoint_is_only_under_explicit_guard() -> None:
    """Importing or opening the notebook must never start training."""
    _, tree = _dl_tcn_ast()
    top_level_runner_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "run_dl_training"
    ]
    assert not top_level_runner_calls

    guarded = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "run_dl_training"
            for statement in node.body
            for call in ast.walk(statement)
        )
    ]
    assert len(guarded) == 1
    assert "RUN_TRAINING" in ast.unparse(guarded[0].test)


def test_dl_tcn_common_outputs_and_selection_are_validation_only() -> None:
    """Locked-test rows cannot choose the DL champion or enter ML/DL comparison."""
    source, tree = _dl_tcn_ast()
    evaluation = ast.get_source_segment(
        source, _named_definition(tree, "evaluate_common_schema")
    ) or ""
    binary_metrics = ast.get_source_segment(
        source, _named_definition(tree, "_binary_metric_rows")
    ) or ""
    selection = ast.get_source_segment(
        source, _named_definition(tree, "select_validation_champion")
    ) or ""
    comparison = ast.get_source_segment(
        source, _named_definition(tree, "write_validation_model_comparison")
    ) or ""
    aggregate_metrics = "\n".join(
        ast.get_source_segment(source, node) or ""
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "compute_metrics_from_predictions"
    )
    runner = ast.get_source_segment(
        source, _named_definition(tree, "run_dl_training")
    ) or ""

    assert "PREDICTION_COLUMNS" in evaluation
    assert "METRIC_COLUMNS" in aggregate_metrics
    assert all(metric in binary_metrics for metric in (
        "aucpr",
        "auroc",
        "event_recall",
        "false_alerts_per_hour",
        "brier_score",
        "ece",
        "person_macro",
    ))
    assert "validation" in selection
    assert "locked_test" in selection
    assert all(metric in selection for metric in (
        "aucpr",
        "event_recall",
        "false_alerts_per_hour",
        "ece",
    ))
    assert "validation" in comparison
    assert "locked_test" in comparison
    assert "keys = ['target', 'metric']" in comparison
    assert "validate='one_to_one'" in comparison
    assert all(metric in aggregate_metrics for metric in (
        "stage_macro_f1",
        "stage_balanced_accuracy",
        "behavior_micro_aucpr",
        "behavior_macro_aucpr",
    ))
    aggregate_shards = ast.get_source_segment(
        source, _named_definition(tree, "aggregate_prediction_shards")
    ) or ""
    validate_person_shard = ast.get_source_segment(
        source, _named_definition(tree, "_validate_person_prediction_rows")
    ) or ""
    assert "select_clean_validation_threshold" in aggregate_shards
    assert (
        "drop_duplicates" in runner
        or "prediction window coverage is duplicate" in validate_person_shard
    )
    assert (
        "compute_metrics_from_predictions" in runner
        or "aggregate_prediction_shards" in runner
    )


def test_dl_tcn_streams_person_sharded_predictions_without_dataframe_collectives() -> None:
    """Evaluation must stream person shards and use bounded threshold scans."""
    source, tree = _dl_tcn_ast()
    for definition in (
        "PersonShardEvalSampler",
        "evaluate_to_prediction_shard",
        "write_prediction_shard_manifest",
        "aggregate_prediction_shards",
        "StreamingMetricAccumulator",
    ):
        assert _named_definition(tree, definition)
    threshold_source = ast.get_source_segment(
        source, _named_definition(tree, "select_validation_threshold")
    ) or ""
    shard_source = ast.get_source_segment(
        source, _named_definition(tree, "evaluate_to_prediction_shard")
    ) or ""
    sampler_source = ast.get_source_segment(
        source, _named_definition(tree, "PersonShardEvalSampler")
    ) or ""
    aggregate_source = ast.get_source_segment(
        source, _named_definition(tree, "aggregate_prediction_shards")
    ) or ""
    runner_source = ast.get_source_segment(
        source, _named_definition(tree, "run_dl_training")
    ) or ""

    assert "THRESHOLD_GRID_SIZE" in threshold_source
    assert "np.unique" not in threshold_source
    assert "ParquetWriter" in shard_source
    assert "self.ranges" in sampler_source
    assert "self.indices.extend" not in sampler_source
    assert "seen_people" in aggregate_source
    assert "seen_windows" not in aggregate_source
    assert "all_gather_object" not in source
    assert "_gather_frames" not in source
    assert "PersonShardEvalSampler" in runner_source
    assert "aggregate_prediction_shards" in runner_source


def test_dl_tcn_ddp_cleanup_and_telemetry_fail_without_collective_hangs() -> None:
    """Cleanup cannot enter a barrier and optional telemetry cannot abort DDP."""
    source, tree = _dl_tcn_ast()
    setup = ast.get_source_segment(source, _named_definition(tree, "setup_ddp")) or ""
    cleanup = ast.get_source_segment(source, _named_definition(tree, "cleanup_ddp")) or ""
    telemetry = ast.get_source_segment(
        source, _named_definition(tree, "safe_wandb_call")
    ) or ""

    assert "timeout=" in setup
    assert "TORCH_NCCL_ASYNC_ERROR_HANDLING" in setup
    assert "destroy_process_group" in cleanup
    assert "barrier" not in cleanup
    assert "except Exception" in telemetry
    assert "return None" in telemetry


def test_dl_tcn_recomputes_physical_identity_and_endpoint_labels() -> None:
    """Rehashed declarations cannot hide changed source data or a shifted endpoint label."""
    source, tree = _dl_tcn_ast()
    for definition in (
        "discover_attached_goal15_inputs",
        "recompute_physical_dataset_identity",
        "verify_train_normalization_statistics",
    ):
        assert _named_definition(tree, definition)
    verify_source = ast.get_source_segment(
        source, _named_definition(tree, "verify_sequence_inputs")
    ) or ""
    dataset_source = ast.get_source_segment(
        source, _named_definition(tree, "Goal15SequenceDataset")
    ) or ""

    assert "recompute_physical_dataset_identity" in verify_source
    assert "source_content_inventory" in verify_source
    assert "EXPECTED_SPLIT_COUNTS" in verify_source
    assert "endpoint label mismatch" in dataset_source
    assert "endpoint context mismatch" in dataset_source
    assert "prediction_time" in dataset_source
    assert "forecast_60s" in dataset_source
    assert "phase" in dataset_source


def test_dl_tcn_normalization_is_per_time_and_has_padding_invariance_check() -> None:
    """Any normalization that aggregates across time can leak right padding into prefix logits."""
    source, tree = _dl_tcn_ast()
    assert "GroupNorm" not in source
    assert _named_definition(tree, "ChannelLayerNorm1d")
    invariant = ast.get_source_segment(
        source, _named_definition(tree, "assert_right_padding_invariance")
    ) or ""

    assert "torch.allclose" in invariant
    assert "event_logits" in invariant
    assert "stage_logits" in invariant
    assert "behavior_logits" in invariant
    assert "right padding" in invariant


def test_dl_tcn_has_guarded_self_contained_torchrun_launcher() -> None:
    """Kaggle import stays idle; workers rediscover physical hashes."""
    source, tree = _dl_tcn_ast()
    for definition in (
        "build_torchrun_worker_source",
        "write_guarded_torchrun_worker",
        "launch_dual_t4_torchrun",
    ):
        assert _named_definition(tree, definition)
    launcher = ast.get_source_segment(
        source, _named_definition(tree, "launch_dual_t4_torchrun")
    ) or ""
    worker = ast.get_source_segment(
        source, _named_definition(tree, "build_torchrun_worker_source")
    ) or ""

    assert "require_exactly_two_cuda_devices" in launcher
    assert "verify_sequence_inputs" in launcher
    assert "python" in launcher
    assert "torch.distributed.run" in launcher
    assert "--standalone" in launcher
    assert "--nproc_per_node=2" in launcher
    assert "/kaggle/working" in launcher
    assert "GOAL15_EXPECTED_SOURCE_HASH" in worker
    assert "GOAL15_EXPECTED_SPLIT_HASH" in worker
    assert "'RUN_LOCKED_TEST'" in worker
    assert "'_verify_declared_sequence_inputs'" in worker
    assert "if RUN_TRAINING" in source


def test_dl_tcn_complete_metrics_stress_and_comparison_contract() -> None:
    """Validation reports uncertainty, label detail, stress, and safe comparison."""
    source, tree = _dl_tcn_ast()
    for definition in (
        "bootstrap_people_ci",
        "compute_forecast_lead_times",
        "compute_stage_confusion_rows",
        "deterministic_stress_batch",
        "evaluate_noise_stress_to_shards",
        "compute_noise_degradation",
        "verify_metric_identity",
    ):
        assert _named_definition(tree, definition)
    metric_source = ast.get_source_segment(
        source, _named_definition(tree, "compute_metrics_from_predictions")
    ) or ""
    comparison = ast.get_source_segment(
        source, _named_definition(tree, "write_validation_model_comparison")
    ) or ""

    aggregate = ast.get_source_segment(
        source, _named_definition(tree, "aggregate_prediction_shards")
    ) or ""
    bootstrap = ast.get_source_segment(
        source, _named_definition(tree, "bootstrap_person_metrics")
    ) or ""
    metric_owner = metric_source + aggregate + bootstrap
    assert "bootstrap_person_metrics" in metric_owner
    assert "forecast_lead" in metric_owner
    assert "global_stage_recall" in metric_owner
    assert "confusion_frames" in metric_owner
    assert "behavior_micro_auroc" in metric_owner
    assert "behavior_micro_f1" in metric_owner
    assert "global_positive_support" in metric_owner
    assert "source_dataset_hash" in comparison
    assert "file_sha256" in comparison
    assert "one_to_one" in comparison
    assert "RUN_VALIDATION_COMPARISON" in source


def test_dl_tcn_train_only_weights_and_global_ddp_conditional_means() -> None:
    """Conditional losses and training telemetry must be global means with train-only support."""
    source, tree = _dl_tcn_ast()
    for definition in (
        "derive_train_loss_weights",
        "write_train_loss_support",
        "global_valid_count",
        "reduce_training_statistics",
    ):
        assert _named_definition(tree, definition)
    loss_source = ast.get_source_segment(
        source, _named_definition(tree, "masked_multitask_loss")
    ) or ""
    train_source = ast.get_source_segment(
        source, _named_definition(tree, "train_one_epoch")
    ) or ""
    runner_source = ast.get_source_segment(
        source, _named_definition(tree, "run_dl_training")
    ) or ""

    assert "all_reduce" in ast.get_source_segment(
        source, _named_definition(tree, "global_valid_count")
    )
    assert "world_size" in loss_source
    assert "global_stage_count" in loss_source
    assert "global_behavior_count" in loss_source
    assert "pos_weight" in loss_source
    assert "reduce_training_statistics" in train_source
    assert "event_local_sum" in loss_source
    assert "stage_local_count" in loss_source
    assert "valid_counts" in train_source
    assert runner_source.count("set_epoch") == 0
    assert "SEQUENCE_LENGTH_CANDIDATE = 600" in source


def test_round3_metric_rows_have_one_owner_and_unique_export_grain() -> None:
    """Streaming aggregation alone owns bootstrap and exported metric keys."""
    source, tree = _dl_tcn_ast()
    metric_source = ast.get_source_segment(
        source, _named_definition(tree, "compute_metrics_from_predictions")
    ) or ""
    aggregate_source = ast.get_source_segment(
        source, _named_definition(tree, "aggregate_prediction_shards")
    ) or ""
    unique_source = ast.get_source_segment(
        source, _named_definition(tree, "assert_unique_metric_rows")
    ) or ""

    assert "bootstrap_people_ci" not in metric_source
    assert "window_id" in aggregate_source
    assert "audit_event_binary" in aggregate_source
    assert "compute_stage_confusion_rows(person)" in aggregate_source
    assert "stress_condition" in unique_source
    assert "duplicated" in unique_source
    assert "assert_unique_metric_rows" in aggregate_source


def test_round3_threshold_modes_are_clean_only_and_hash_bound() -> None:
    """Stress and locked evaluation can only consume a persisted clean threshold."""
    source, tree = _dl_tcn_ast()
    selector = ast.get_source_segment(
        source, _named_definition(tree, "select_clean_validation_threshold")
    ) or ""
    fixed = ast.get_source_segment(
        source, _named_definition(tree, "load_fixed_threshold")
    ) or ""
    aggregate = ast.get_source_segment(
        source, _named_definition(tree, "aggregate_prediction_shards")
    ) or ""
    stress = ast.get_source_segment(
        source, _named_definition(tree, "evaluate_noise_stress_to_shards")
    ) or ""
    runner = ast.get_source_segment(
        source, _named_definition(tree, "run_dl_training")
    ) or ""

    assert "select_clean_validation" in selector
    assert "event_level_f1" in selector
    assert "event_recall" in selector
    assert "false_alerts_per_hour" in selector
    assert "THRESHOLD_GRID_SIZE" in selector
    assert "fixed_threshold" in fixed
    assert "sha256_file" in fixed
    assert "threshold_mode" in aggregate
    assert "select_clean_validation" in aggregate
    assert "fixed_threshold" in aggregate
    assert "threshold_artifact" in stress
    assert "threshold_artifact" in runner


def test_round3_global_and_macro_metrics_use_bounded_sufficient_statistics() -> None:
    """Nonlinear global metrics are not means of per-person scalar metrics."""
    source, tree = _dl_tcn_ast()
    accumulator = ast.get_source_segment(
        source, _named_definition(tree, "StreamingMetricAccumulator")
    ) or ""
    aggregate = ast.get_source_segment(
        source, _named_definition(tree, "aggregate_prediction_shards")
    ) or ""

    for token in (
        "brier_sum",
        "probability_sum",
        "global_aucpr",
        "global_auroc",
        "global_row_f1",
        "global_ece",
    ):
        assert token in accumulator
    assert "person_macro_f1" in aggregate
    assert "stage_macro_f1" in aggregate
    assert "confusion_frames" in aggregate
    assert "metric_frames" not in aggregate


def test_round3_row_group_cache_is_bounded_and_reuses_adjacent_reads() -> None:
    """Adjacent windows reuse one row group and cache eviction is bounded."""
    source, tree = _dl_tcn_ast()
    cache_node = _named_definition(tree, "BoundedRowGroupCache")
    cache_source = ast.get_source_segment(source, cache_node) or ""
    namespace: dict[str, Any] = {
        "OrderedDict": __import__("collections").OrderedDict,
        "Any": Any,
        "pd": pd,
    }
    exec(cache_source, namespace)
    cache = namespace["BoundedRowGroupCache"](max_groups=2, max_bytes=10_000)
    reads: list[int] = []

    def load(group: int) -> pd.DataFrame:
        reads.append(group)
        return pd.DataFrame({"value": np.arange(16) + group})

    for _ in range(20):
        cache.get(0, lambda: load(0))
    cache.get(1, lambda: load(1))
    cache.get(2, lambda: load(2))
    cache.get(0, lambda: load(0))

    assert reads.count(0) == 2
    assert reads.count(1) == 1
    assert reads.count(2) == 1
    assert cache.current_groups <= 2
    assert cache.current_bytes <= 10_000


def test_round3_dataset_uses_interval_index_and_person_local_block_batches() -> None:
    """Training IO stays row-group local without a global random window sampler."""
    source, tree = _dl_tcn_ast()
    dataset = ast.get_source_segment(
        source, _named_definition(tree, "Goal15SequenceDataset")
    ) or ""
    interval = ast.get_source_segment(
        source, _named_definition(tree, "RowGroupIntervalIndex")
    ) or ""
    sampler = ast.get_source_segment(
        source, _named_definition(tree, "PersonBlockBatchSampler")
    ) or ""
    runner = ast.get_source_segment(
        source, _named_definition(tree, "run_dl_training")
    ) or ""

    assert "metadata.row_group" in interval
    assert "BoundedRowGroupCache" in dataset
    assert "read_row_group" in dataset
    assert "TIMELINE_READ_COLUMNS" in dataset
    assert "canonical_time contiguous" in sampler
    assert "set_epoch" in sampler
    assert "rank" in sampler
    assert "batch_sampler=train_batch_sampler" in runner
    assert "DistributedSampler(" not in runner

    class FakeSamplerBase:
        @classmethod
        def __class_getitem__(cls, item: Any) -> type[FakeSamplerBase]:
            _ = item
            return cls

    sampler_namespace: dict[str, Any] = {
        "Sampler": FakeSamplerBase,
        "Goal15SequenceDataset": Any,
        "hashlib": hashlib,
        "np": np,
    }
    exec(sampler, sampler_namespace)

    class FakeDataset:
        def person_ranges(self) -> dict[str, list[range]]:
            return {"person-a": [range(0, 5)], "person-b": [range(5, 9)]}

    sampler_class = sampler_namespace["PersonBlockBatchSampler"]
    rank_zero = sampler_class(
        FakeDataset(), batch_size=4, rank=0, num_replicas=2, seed=17
    )
    rank_one = sampler_class(
        FakeDataset(), batch_size=4, rank=1, num_replicas=2, seed=17
    )
    batches_zero = list(rank_zero)
    batches_one = list(rank_one)
    indices_zero = {index for batch in batches_zero for index in batch}
    indices_one = {index for batch in batches_one for index in batch}

    assert len(batches_zero) == len(rank_zero)
    assert len(batches_one) == len(rank_one)
    assert len(batches_zero) == len(batches_one)
    assert not indices_zero & indices_one
    assert indices_zero | indices_one == set(range(9))
    assert all(
        batch == list(range(batch[0], batch[-1] + 1))
        for batch in [*batches_zero, *batches_one]
    )


def test_round3_ml_champion_artifact_is_identity_safe_and_one_to_one() -> None:
    """ML exports one validation champion on the common grid for DL comparison."""
    ml_source = code_cell_source(
        load_notebooks()[KAGGLE_DIR / "02_ml_benchmark.ipynb"]
    )
    dl_source, dl_tree = _dl_tcn_ast()
    discovery = ast.get_source_segment(
        dl_source, _named_definition(dl_tree, "discover_ml_champion_artifact")
    ) or ""
    comparison = ast.get_source_segment(
        dl_source, _named_definition(dl_tree, "write_validation_model_comparison")
    ) or ""

    assert "COMMON_THRESHOLD_GRID_SIZE" in ml_source
    assert "write_validation_champion_artifact" in ml_source
    assert "validation_champion_metrics.parquet" in ml_source
    for token in (
        "source_dataset_hash",
        "split_hash",
        "label_schema_hash",
        "feature_schema_hash",
        "threshold",
        "model_name",
        "target",
        "file_sha256",
    ):
        assert token in ml_source
        assert token in discovery or token in comparison
    assert "sha256_file" in discovery
    assert "validate='one_to_one'" in comparison
    assert "locked_test" in comparison


def test_round3_physical_audit_is_fail_closed_and_checks_exact_person_sets() -> None:
    """Mutable manifests cannot replace physical source and exact split audits."""
    source, tree = _dl_tcn_ast()
    verify = ast.get_source_segment(
        source, _named_definition(tree, "verify_sequence_inputs")
    ) or ""
    inventory = ast.get_source_segment(
        source, _named_definition(tree, "verify_declared_physical_inventory")
    ) or ""
    people = ast.get_source_segment(
        source, _named_definition(tree, "verify_exact_sequence_person_sets")
    ) or ""

    assert "RECOMPUTE_NORMALIZATION = True" in source
    for token in ("outcome", "registry", "personal_baseline", "prepared"):
        assert token in inventory
    assert "sha256_file" in inventory
    assert "schema" in inventory
    assert "row_count" in inventory
    assert "verify_declared_physical_inventory" in verify
    assert "train_300" in people
    assert "validation_600" in people
    assert "locked_test_600" in people
    assert "EXPECTED_SPLIT_COUNTS" in people
    assert "verify_exact_sequence_person_sets" in verify


def test_round3_person_shard_semantics_match_expected_index_rows() -> None:
    """Each person shard exactly matches index identities and conditional heads."""
    source, tree = _dl_tcn_ast()
    semantic = ast.get_source_segment(
        source, _named_definition(tree, "validate_person_shard_semantics")
    ) or ""
    aggregate = ast.get_source_segment(
        source, _named_definition(tree, "aggregate_prediction_shards")
    ) or ""

    assert "expected_person_windows" in semantic
    assert "one pattern row per expected window" in semantic
    assert "set(STAGE_CODES)" in semantic
    assert "set(BEHAVIOR_CODES)" in semantic
    assert "decision_mask" in semantic
    for token in (
        "dataset_id",
        "run_id",
        "person_key",
        "window_id",
        "canonical_time",
        "split_role",
    ):
        assert token in semantic
    assert "validate_person_shard_semantics" in aggregate


def test_round3_bootstrap_and_sequence_policy_are_explicit() -> None:
    """Bootstrap recomputes person-grouped metrics; the candidate is fixed at 600s."""
    source, tree = _dl_tcn_ast()
    bootstrap = ast.get_source_segment(
        source, _named_definition(tree, "bootstrap_person_metrics")
    ) or ""
    aggregate = ast.get_source_segment(
        source, _named_definition(tree, "aggregate_prediction_shards")
    ) or ""

    for token in (
        "global_aucpr_ci_lower",
        "global_event_recall_ci_lower",
        "global_false_alerts_per_hour_ci_upper",
        "person_macro_f1_ci_lower",
    ):
        assert token in bootstrap or token in aggregate
    assert "person_sufficient_statistics" in bootstrap
    assert "SEQUENCE_LENGTH_CANDIDATE = 600" in source
    assert "SEQUENCE_LENGTH_POLICY" in source
    assert "future comparison" in source


def test_round4_window_major_iterators_bound_carry_and_stream_expected_rows(
    tmp_path: Path,
) -> None:
    """Prediction and expected-index streams retain at most one window group."""
    source, tree = _dl_tcn_ast()
    iterator_node = _named_definition(tree, "iter_window_major_prediction_groups")
    iterator_source = ast.get_source_segment(source, iterator_node) or ""
    expected_source = ast.get_source_segment(
        source, _named_definition(tree, "iter_expected_sequence_windows")
    ) or ""
    dataset_source = ast.get_source_segment(
        source, _named_definition(tree, "Goal15SequenceDataset")
    ) or ""

    assert "pd.concat" not in iterator_source
    assert "max_window_rows" in iterator_source
    assert "iter_batches" in iterator_source
    assert "iter_batches" in expected_source
    assert "expected_windows_for_person" not in dataset_source
    assert "pd.Series" not in expected_source

    rows = []
    for window in range(30):
        for target in ("pattern_binary", "stage::LOW", "stage::MEDIUM"):
            rows.append(
                {
                    "person_key": "person-a",
                    "run_id": "run-a",
                    "dataset_id": "dataset-a",
                    "canonical_time": pd.Timestamp(
                        "2026-01-01", tz="UTC"
                    ) + pd.Timedelta(seconds=window),
                    "window_id": f"window-{window:03d}",
                    "target": target,
                }
            )
    path = tmp_path / "prediction-shard.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=4)
    namespace = {"Path": Path, "pq": pq, "pd": pd}
    exec(iterator_source, namespace)
    groups = list(
        namespace["iter_window_major_prediction_groups"](
            path, batch_rows=5, max_window_rows=16
        )
    )
    assert len(groups) == 30
    assert all(group["window_id"].nunique() == 1 for group in groups)
    assert all(len(group) == 3 for group in groups)


def test_round4_streaming_event_grid_is_chunk_boundary_invariant() -> None:
    """Event maxima and alert-run overlap survive arbitrary chunk boundaries."""
    source, tree = _dl_tcn_ast()
    event_source = ast.get_source_segment(
        source, _named_definition(tree, "StreamingEventGridState")
    ) or ""
    assert "alert_truth_overlap" in event_source
    assert "truth_event_max_bucket" in event_source
    assert "update_chunk" in event_source
    namespace = {"np": np, "Any": Any}
    exec(event_source, namespace)
    state_class = namespace["StreamingEventGridState"]
    truth = np.array([0, 1, 1, 0, 0, 1, 0, 0], dtype=np.int8)
    probability = np.array([0.8, 0.2, 0.9, 0.8, 0.1, 0.7, 0.8, 0.0])
    whole = state_class(bins=11)
    whole.update_chunk(truth, probability)
    whole_result = whole.finalize_person(duration_hours=8 / 3600)
    chunked = state_class(bins=11)
    chunked.update_chunk(truth[:3], probability[:3])
    chunked.update_chunk(truth[3:6], probability[3:6])
    chunked.update_chunk(truth[6:], probability[6:])
    chunked_result = chunked.finalize_person(duration_hours=8 / 3600)
    for key in ("detected", "false_alerts"):
        np.testing.assert_array_equal(whole_result[key], chunked_result[key])
    assert whole_result["truth_events"] == chunked_result["truth_events"]


def test_round4_ddp_writes_done_markers_before_rank0_cpu_postprocess() -> None:
    """No CPU aggregation or threshold communication occurs inside the process group."""
    source, tree = _dl_tcn_ast()
    runner = ast.get_source_segment(
        source, _named_definition(tree, "run_dl_training")
    ) or ""
    postprocess = ast.get_source_segment(
        source, _named_definition(tree, "run_rank0_postprocess")
    ) or ""
    marker = ast.get_source_segment(
        source, _named_definition(tree, "write_rank_done_marker")
    ) or ""

    assert "write_rank_done_marker" in runner
    assert runner.index("cleanup_ddp()") < runner.index("run_rank0_postprocess")
    assert "dist.broadcast" not in runner
    assert "gather_object" not in runner
    assert "_collect_small_manifests" not in runner
    assert "rank != 0" in runner
    assert "rank_done" in marker
    assert "select_clean_validation" in postprocess


def test_round4_ml_champion_exports_all_targets_and_endpoint_coverage() -> None:
    """ML/DL comparison is recomputed on exact DL-600 validation endpoints."""
    ml_source = code_cell_source(
        load_notebooks()[KAGGLE_DIR / "02_ml_benchmark.ipynb"]
    )
    dl_source, dl_tree = _dl_tcn_ast()
    comparison = ast.get_source_segment(
        dl_source, _named_definition(
            dl_tree, "compare_ml_dl_on_dl600_endpoints"
        )
    ) or ""

    assert "validation_champion_predictions.parquet" in ml_source
    assert "prediction_sha256" in ml_source
    assert "endpoint_coverage_hash" in ml_source
    assert "target_model_id" in ml_source
    assert "STAGE_CODES" in comparison
    assert "BEHAVIOR_CODES" in comparison
    assert "endpoint_coverage_hash" in comparison
    assert "iter_window_major_prediction_groups" in comparison
    assert "validation" in comparison
    assert "locked_test" in comparison


def test_round4_behavior_metrics_have_per_code_macro_and_micro_support() -> None:
    """Behavior reporting preserves all ten codes before macro aggregation."""
    source, tree = _dl_tcn_ast()
    metrics = ast.get_source_segment(
        source, _named_definition(tree, "finalize_streaming_metrics")
    ) or ""
    for token in (
        "behavior_code_aucpr",
        "behavior_code_auroc",
        "behavior_code_f1",
        "behavior_positive_support",
        "behavior_macro_aucpr",
        "behavior_macro_auroc",
        "behavior_macro_f1",
        "behavior_micro_aucpr",
        "behavior_micro_auroc",
        "behavior_micro_f1",
    ):
        assert token in metrics
    assert "for code in BEHAVIOR_CODES" in metrics
    assert "assert_unique_metric_rows" in metrics


def test_round4_stress_shift_and_latent_dropout_are_semantically_bounded() -> None:
    """Time shift never wraps and latent dropout cannot touch time/context columns."""
    source, tree = _dl_tcn_ast()
    stress = ast.get_source_segment(
        source, _named_definition(tree, "deterministic_stress_batch")
    ) or ""
    indices_source = ast.get_source_segment(
        source, _named_definition(tree, "latent_factor_feature_indices")
    ) or ""

    assert "torch.roll" not in stress
    assert "invalid_edge" in stress
    assert "mask" in stress
    assert "CAUSAL_FACTORS" in stress
    assert "latent_factor_feature_indices" in stress
    namespace = {"Sequence": list}
    exec(indices_source, namespace)
    columns = [
        "autonomic_arousal__robust_z",
        "autonomic_arousal__mean_5s",
        "autonomic_arousal__std_60s",
        "autonomic_arousal__slope_300s",
        "motor_activation__robust_z",
        "time_sin",
        "context__sleep",
    ]
    selected = namespace["latent_factor_feature_indices"](
        "autonomic_arousal", columns
    )
    assert selected == [0, 1, 2, 3]


def test_round4_common_grid_auroc_includes_exact_origin_for_p_equals_one() -> None:
    """An all-p==1 tie has AUROC 0.5, including the missing ROC origin."""
    ml_source = code_cell_source(
        load_notebooks()[KAGGLE_DIR / "02_ml_benchmark.ipynb"]
    )
    ml_tree = ast.parse(ml_source)
    ml_hist = ast.get_source_segment(
        ml_source, _named_definition(ml_tree, "_common_grid_histogram")
    ) or ""
    ml_metric = ast.get_source_segment(
        ml_source, _named_definition(ml_tree, "_common_grid_binary_metrics")
    ) or ""
    dl_source, dl_tree = _dl_tcn_ast()
    dl_metric = ast.get_source_segment(
        dl_source, _named_definition(dl_tree, "_histogram_binary_metrics")
    ) or ""
    namespace = {"np": np, "Any": Any, "COMMON_THRESHOLD_GRID_SIZE": 101}
    exec(ml_hist + "\n" + ml_metric, namespace)
    truth = np.array([0, 1], dtype=np.int8)
    probability = np.array([1.0, 1.0])
    ml_result = namespace["_common_grid_binary_metrics"](
        truth, probability, threshold=0.5
    )
    assert ml_result["global_auroc"] == pytest.approx(0.5)
    dl_namespace = {"np": np}
    exec(dl_metric, dl_namespace)
    positive = np.zeros(101)
    negative = np.zeros(101)
    probability_sum = np.zeros(101)
    positive[-1] = negative[-1] = 1
    probability_sum[-1] = 2
    dl_result = dl_namespace["_histogram_binary_metrics"](
        positive, negative, probability_sum, 1.0, 50
    )
    assert dl_result["global_auroc"] == pytest.approx(0.5)


def test_round4_sampler_honors_block_windows_and_batch_locality() -> None:
    """Configured blocks stay contiguous while only block order changes by epoch."""
    source, tree = _dl_tcn_ast()
    sampler = ast.get_source_segment(
        source, _named_definition(tree, "PersonBlockBatchSampler")
    ) or ""
    assert "block_windows" in sampler
    assert "range(person_range.start, person_range.stop, block_windows)" in sampler
    assert "batches_per_block" in sampler
    assert "shuffle(order)" in sampler
    assert "yield list(block" not in sampler


def test_round4_postprocess_writes_stress_degradation_and_unique_support() -> None:
    """Rank0 postprocess exports clean/stress degradation with fixed threshold identity."""
    source, tree = _dl_tcn_ast()
    postprocess = ast.get_source_segment(
        source, _named_definition(tree, "run_rank0_postprocess")
    ) or ""
    assert "compute_noise_degradation" in postprocess
    assert "degradation.to_parquet" in postprocess
    assert "threshold_artifact_hash" in postprocess
    assert "assert_unique_metric_rows" in postprocess
    assert "fixed_threshold" in postprocess
