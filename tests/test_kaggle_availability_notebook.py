from __future__ import annotations

import json
from pathlib import Path

import nbformat


def test_sensor_availability_notebook_is_non_training_and_valid() -> None:
    path = Path(__file__).parents[1] / "kaggle/08_sensor_availability_reproduction.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    assert "RUN_TRAINING = False" in source
    assert "RUN_LOCKED_TEST = False" in source
    assert "verify_sensor_availability_package" in source
    assert "AVAILABILITY_PROFILES" in source


def test_sensor_availability_kernel_metadata_points_to_private_model() -> None:
    path = (
        Path(__file__).parents[1]
        / "kaggle/08_sensor_availability_reproduction.kernel-metadata.json"
    )
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["model_sources"] == [
        "bjcoding/multisensor-goal15-availability/ScikitLearn/oracle-sanity-v1/3"
    ]
