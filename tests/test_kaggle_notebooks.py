"""Static safety contracts for the unexecuted Kaggle notebook shells."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
