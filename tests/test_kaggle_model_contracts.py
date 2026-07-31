from pathlib import Path

import pytest

from multisensor_ml.kaggle_model_contracts import (
    BEHAVIOR_CODES,
    load_kaggle_model_package_config,
)


def test_fixed_reproduction_config_is_private_and_non_training() -> None:
    config = load_kaggle_model_package_config(
        Path("configs/kaggle_hierarchical_reproduction.yaml")
    )

    assert config.series_id == "mvp3-oracle-v1"
    assert config.framework == "scikitLearn"
    assert config.variation_slug == "oracle-sanity-v1"
    assert config.is_private is True
    assert config.run_training is False
    assert config.run_locked_test is False
    assert config.use_gpu is False
    assert config.sample_split_role == "validation"
    assert len(BEHAVIOR_CODES) == 10


def test_locked_test_cannot_be_selected_for_sample(tmp_path: Path) -> None:
    payload = Path("configs/kaggle_hierarchical_reproduction.yaml").read_text()
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        payload.replace(
            "sample_split_role: validation", "sample_split_role: locked_test"
        )
    )

    with pytest.raises(ValueError, match="locked_test"):
        load_kaggle_model_package_config(invalid)
