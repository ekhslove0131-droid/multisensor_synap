from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from multisensor_ml.cloud_run_onnx import PredictRequest, create_app
from multisensor_ml.dual_sensor_onnx import SENSOR_VARIANTS


def test_cloud_run_app_requires_both_final_variants(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="galaxy_watch"):
        create_app(tmp_path)


def _endpoint(app: object, path: str):
    return next(route.endpoint for route in app.routes if getattr(route, "path", None) == path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_final_bundles_are_hash_closed_and_routes_predict() -> None:
    root = Path(__file__).parents[1] / "services" / "onnx_api" / "models" / "goal15-final-v1"
    app = create_app(root)
    health = _endpoint(app, "/healthz")()
    assert health["status"] == "READY"
    assert tuple(health["variants"]) == SENSOR_VARIANTS
    assert health["locked_test_read"] is False
    predict = _endpoint(app, "/v1/predict")
    for variant in SENSOR_VARIANTS:
        variant_root = root / variant
        manifest = json.loads((variant_root / "manifest.json").read_text(encoding="utf-8"))
        assert _sha256(variant_root / "model.onnx") == manifest["model_sha256"]
        response = predict(
            PredictRequest(
                variant=variant,
                features={name: 0.0 for name in manifest["feature_names"]},
                request_id="test",
            )
        )
        assert response["status"] == "PREDICTED"
        assert 0.0 <= response["probability"] <= 1.0
        assert response["locked_test_read"] is False
