from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import skops.io as sio
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from multisensor_ml.kaggle_model_contracts import (
    BEHAVIOR_CODES,
    KaggleModelPackageConfig,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _dump(path: Path, model: object) -> list[str]:
    sio.dump(model, path)
    return sorted(sio.get_untrusted_types(file=path))


@pytest.fixture
def hierarchical_source(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    prepared = root / "data/prepared/mvp3-oracle-v1"
    registry = root / "artifacts/registry/mvp3-oracle-v1"
    types = registry / "types"
    stage = registry / "stage-model"
    behavior = registry / "behavior-model"
    for path in (prepared, types, stage, behavior):
        path.mkdir(parents=True)

    x_binary = np.asarray([[-2.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    y_binary = np.asarray([0, 0, 1, 1])
    logistic = LogisticRegression(random_state=7).fit(x_binary, y_binary)
    hgb = HistGradientBoostingClassifier(random_state=7, min_samples_leaf=1).fit(
        x_binary, y_binary
    )
    x_stage = np.asarray([[float(index), 1.0] for index in range(10)])
    y_stage = np.asarray([index % 5 for index in range(10)])
    stage_hgb = HistGradientBoostingClassifier(
        random_state=7, min_samples_leaf=1
    ).fit(x_stage, y_stage)
    stage_logistic = LogisticRegression(random_state=7).fit(x_stage, y_stage)

    _write_json(prepared / "global_baseline.json", [{"context": "wake_rest"}])
    pd.DataFrame({"person_key": ["validation-person"], "n_eff": [1800]}).to_parquet(
        prepared / "personal_baseline.parquet", index=False
    )
    sample = pd.DataFrame(
        {
            "person_key": ["validation-person"] * 100,
            "session_id": ["session-1"] * 100,
            "timestamp_utc": pd.date_range("2026-01-01", periods=100, freq="s", tz="UTC"),
            "f1": np.linspace(-2.0, 2.0, 100),
        }
    )
    sample.to_parquet(prepared / "validation-person.parquet", index=False)
    _write_json(
        prepared / "manifest.json",
        {
            "prepared_schema": "goal1.5/prepared/v1",
            "status": "oracle/sanity",
            "real_data_status": "NOT VERIFIED",
            "feature_names": ["f1"],
            "people": [
                {
                    "person_key": "validation-person",
                    "split_role": "validation",
                    "path": "validation-person.parquet",
                    "dataset_id": "fixture-validation",
                }
            ],
        },
    )

    type_unknown = _dump(types / "standard_type_model.skops", logistic)
    _write_json(types / "standard_type_preprocessing.json", {"features": ["f1"]})
    pd.DataFrame(
        {"person_key": ["validation-person"], "STD-A": [1.0]}
    ).to_parquet(types / "person_standard_types.parquet", index=False)
    pd.DataFrame(
        {"person_key": ["validation-person"], "status": ["IN_DISTRIBUTION"]}
    ).to_parquet(types / "ood_status.parquet", index=False)
    _write_json(
        types / "manifest.json",
        {
            "schema_version": "goal1.5/standard-types/v1",
            "status": "oracle/sanity",
            "real_data_status": "NOT VERIFIED",
            "selected_k": 1,
            "type_names": ["STD-A"],
            "models": [
                {
                    "path": "standard_type_model.skops",
                    "unknown_types": type_unknown,
                }
            ],
        },
    )

    stage_entries = []
    for head, models in {
        "event": {"hist_gradient_boosting": hgb, "logistic_regression": logistic},
        "stage": {
            "hist_gradient_boosting": stage_hgb,
            "logistic_regression": stage_logistic,
        },
    }.items():
        for name, model in models.items():
            filename = f"{head}__{name}.skops"
            stage_entries.append(
                {
                    "head": head,
                    "model_name": name,
                    "path": filename,
                    "unknown_types": _dump(stage / filename, model),
                }
            )
    _write_json(
        stage / "feature_schema.json",
        {"features": ["f1", "STD-A"], "forbidden": ["hidden_archetype"]},
    )
    _write_json(
        stage / "manifest.json",
        {
            "schema_version": "goal1.5/hierarchical-stage-model/v1",
            "status": "oracle/sanity",
            "real_data_status": "NOT VERIFIED",
            "selected_event_model": "hist_gradient_boosting",
            "selected_stage_model": "hist_gradient_boosting",
            "event_threshold": 0.5,
            "models": stage_entries,
        },
    )

    behavior_entries = []
    selected = {}
    for code in BEHAVIOR_CODES:
        selected[code] = "logistic_regression"
        for name, model in {
            "hist_gradient_boosting": hgb,
            "logistic_regression": logistic,
        }.items():
            filename = f"behavior__{code}__{name}.skops"
            behavior_entries.append(
                {
                    "behavior_code": code,
                    "model_name": name,
                    "path": filename,
                    "selected": name == selected[code],
                    "unknown_types": _dump(behavior / filename, model),
                }
            )
    _write_json(
        behavior / "feature_schema.json",
        {"features": ["f1", "STD-A", "event_oof_probability"]},
    )
    _write_json(
        behavior / "manifest.json",
        {
            "schema_version": "goal1.5/behavior-model/v1",
            "status": "oracle/sanity",
            "real_data_status": "NOT VERIFIED",
            "selected_models": selected,
            "support_status": {code: "SUPPORTED" for code in BEHAVIOR_CODES},
            "models": behavior_entries,
        },
    )
    (root / "uv.lock").write_text("fixture-lock\n")
    return root


@pytest.fixture
def package_config(hierarchical_source: Path, tmp_path: Path) -> KaggleModelPackageConfig:
    return KaggleModelPackageConfig(
        schema_version="goal1.5/kaggle-model-package-config/v1",
        project_root=hierarchical_source,
        series_id="mvp3-oracle-v1",
        output_root=tmp_path / "package",
        owner_slug="bjcoding",
        model_slug="multisensor-goal15-hierarchical",
        framework="scikitLearn",
        variation_slug="oracle-sanity-v1",
        source_dataset_handle="bjcoding/multisensor-goal15-oracle-mvp3",
        source_dataset_version=1,
        license_name="Apache 2.0",
        notebook_slug="multisensor-goal15-hierarchical-reproduction",
        is_private=True,
        run_training=False,
        run_locked_test=False,
        use_gpu=False,
        sample_split_role="validation",
        sample_rows=60,
    )


@pytest.fixture
def built_wheel(tmp_path: Path) -> Path:
    wheel = tmp_path / "multisensor_ml-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"fixture-wheel")
    return wheel
