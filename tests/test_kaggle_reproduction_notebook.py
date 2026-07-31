import json
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from multisensor_ml.kaggle_model_package import build_kaggle_model_package
from scripts.build_kaggle_reproduction_notebook import build_notebook


def test_reproduction_notebook_is_cpu_only_and_non_training() -> None:
    notebook = nbformat.read(
        "kaggle/07_hierarchical_model_reproduction.ipynb", as_version=4
    )
    source = "\n".join(cell.source for cell in notebook.cells)
    metadata = json.loads(
        Path(
            "kaggle/07_hierarchical_model_reproduction.kernel-metadata.json"
        ).read_text()
    )

    assert "RUN_TRAINING = False" in source
    assert "RUN_LOCKED_TEST = False" in source
    assert "USE_GPU = False" in source
    assert "REPRODUCED" in source
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False


def test_reproduction_notebook_executes_top_to_bottom(
    package_config, built_wheel: Path, tmp_path: Path, monkeypatch
) -> None:
    package = build_kaggle_model_package(package_config, built_wheel)
    notebook = build_notebook(
        model_root_override=package.root,
        dataset_root_override=package_config.project_root
        / "data/prepared/mvp3-oracle-v1",
    )
    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")
    executed = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute()
    output_text = "\n".join(
        output.get("text", "")
        for cell in executed.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )

    assert '"status": "REPRODUCED"' in output_text
