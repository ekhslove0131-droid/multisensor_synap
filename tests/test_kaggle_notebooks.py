"""Static safety contracts for the unexecuted Kaggle notebook shells."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def write_manifest_fixture(root: Path) -> None:
    prepared_person = root / "prepared__people__person-1.parquet"
    prepared_person.write_text("prepared person")
    personal_baseline = root / "prepared__personal_baseline.parquet"
    personal_baseline.write_text("personal baseline")
    outcome_events = root / "outcomes__outcome_events.parquet"
    outcome_events.write_text("outcome events")
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
                "files": {"outcome_events.parquet": sha256_text("outcome events")},
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
            "canonical_time": list(range(10)),
            "feature": list(range(10)),
        }
    )
    labels = pd.DataFrame(
        {
            "person_key": ["train", "train", "train"],
            "canonical_time": [0, 1, 2],
            "event_binary": [1, 0, 0],
            "hard_negative_id": [None, "negative-id", None],
            "hard_negative_type": [None, None, "artifact-only"],
        }
    )
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
            "canonical_time": [0],
            "feature": [1.0],
        }
    )
    labels = pd.DataFrame(
        {
            "person_key": [split_role],
            "canonical_time": [0],
            "event_binary": [1],
            "event_intensity_truth": [0.9],
            "active_target_event": [1],
            "hard_negative_id": ["not-a-feature"],
        }
    )
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
