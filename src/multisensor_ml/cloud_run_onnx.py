"""Small fail-closed FastAPI surface for the two final ONNX sensor views."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from multisensor_ml.dual_sensor_onnx import (
    SENSOR_VARIANTS,
    OnnxBundle,
    SensorVariant,
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant: SensorVariant
    features: dict[str, float]
    request_id: str | None = Field(default=None, max_length=128)


def _default_model_root() -> Path:
    configured = os.environ.get("MODEL_ROOT")
    if configured:
        return Path(configured)
    return Path("services/onnx_api/models/goal15-final-v1")


def create_app(model_root: Path | None = None) -> Any:
    """Create an app only when both variant bundles pass manifest validation."""

    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as error:  # pragma: no cover - optional extra guard
        raise RuntimeError("install the cloud-run extra to run the API") from error
    root = (model_root or _default_model_root()).resolve()
    bundles = {
        variant: OnnxBundle.load(root / variant) for variant in SENSOR_VARIANTS
    }
    app = FastAPI(title="Multisensor Goal 1.5 ONNX API", version="1.0.0")

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {
            "status": "READY",
            "variants": list(bundles),
            "data_status": "oracle/sanity",
            "real_data_status": "NOT VERIFIED",
            "locked_test_read": False,
        }

    @app.post("/v1/predict")
    def predict(request: PredictRequest) -> dict[str, object]:
        bundle = bundles[request.variant]
        if not all(math.isfinite(value) for value in request.features.values()):
            raise HTTPException(status_code=422, detail="features must be finite numbers")
        try:
            result = bundle.predict_one(request.features)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        result["request_id"] = request.request_id
        return result

    return app


try:
    app = create_app()
except (FileNotFoundError, ValueError, RuntimeError):
    # Keep `uvicorn multisensor_ml.cloud_run_onnx:app` importable before the
    # model volume is mounted; healthz returns a clear 503 instead of a fake
    # prediction. The deployed image contains both bundles and uses create_app.
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Multisensor Goal 1.5 ONNX API", version="1.0.0")

    @app.get("/healthz")
    def unavailable_healthz() -> dict[str, object]:
        raise HTTPException(status_code=503, detail="ONNX model bundles are not mounted")
