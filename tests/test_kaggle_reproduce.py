from pathlib import Path

import pandas as pd
import pytest

from multisensor_ml.kaggle_model_contracts import BEHAVIOR_CODES
from multisensor_ml.kaggle_model_package import collect_hierarchical_payload
from multisensor_ml.kaggle_reproduce import (
    compare_expected,
    load_hierarchical_package,
    predict_hierarchical,
)


@pytest.fixture
def loaded_package(package_config, built_wheel: Path, tmp_path: Path):
    root = tmp_path / "payload"
    collect_hierarchical_payload(package_config, root, built_wheel)
    return load_hierarchical_package(root)


@pytest.fixture
def sample_frame(package_config) -> pd.DataFrame:
    return pd.read_parquet(
        package_config.project_root
        / "data/prepared/mvp3-oracle-v1/validation-person.parquet"
    )


def test_predict_hierarchical_outputs_full_korean_contract(
    loaded_package, sample_frame: pd.DataFrame
) -> None:
    result = predict_hierarchical(loaded_package, sample_frame)

    assert {
        "event_probability",
        "predicted_stage",
        "std_type",
        "std_a_membership",
        "판정",
        "단계",
        "분포상태",
        *BEHAVIOR_CODES,
    }.issubset(result.columns)
    assert set(result["predicted_stage"]).issubset(
        {
            "NO_EVENT",
            "LOW",
            "MEDIUM",
            "HIGH",
            "DECREASING",
            "RECOVERY",
            "NOT_DECISIONABLE",
        }
    )


def test_predict_hierarchical_rejects_truth_feature(
    loaded_package, sample_frame: pd.DataFrame
) -> None:
    contaminated = sample_frame.assign(hidden_archetype="forbidden")
    with pytest.raises(ValueError, match="forbidden model input"):
        predict_hierarchical(loaded_package, contaminated)


def test_compare_expected_reports_first_probability_mismatch(
    loaded_package, sample_frame: pd.DataFrame
) -> None:
    actual = predict_hierarchical(loaded_package, sample_frame)
    expected = actual.copy()
    expected.loc[0, "event_probability"] += 0.01

    result = compare_expected(actual, expected)

    assert result.status == "FAILED"
    assert result.first_mismatch_column == "event_probability"


def test_compare_expected_accepts_exact_contract(
    loaded_package, sample_frame: pd.DataFrame
) -> None:
    actual = predict_hierarchical(loaded_package, sample_frame)
    result = compare_expected(actual, actual.copy())

    assert result.status == "REPRODUCED"
    assert result.compared_rows == len(actual)


def test_compare_expected_accepts_cross_platform_probability_roundoff(
    loaded_package, sample_frame: pd.DataFrame
) -> None:
    actual = predict_hierarchical(loaded_package, sample_frame)
    expected = actual.copy()
    expected.loc[0, "ear_covering"] += 5e-9

    result = compare_expected(actual, expected)

    assert result.status == "REPRODUCED"
