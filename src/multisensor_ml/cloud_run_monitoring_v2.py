"""Cloud Run FastAPI adapter for the v2 standard-plus-pattern bundles."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from multisensor_ml.monitoring_v2 import V2_VARIANTS
from multisensor_ml.monitoring_v2_onnx import MonitoringV2Bundle


class MonitoringV2PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant: str
    features: dict[str, float]
    request_id: str | None = Field(default=None, max_length=128)


def _default_model_root() -> Path:
    configured = os.environ.get("MODEL_ROOT_V2")
    if configured:
        return Path(configured)
    return Path("services/onnx_api/models/goal15-monitor-v2")


def create_monitoring_v2_app(model_root: Path | None = None) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install the cloud-run extra") from error
    root = (model_root or _default_model_root()).resolve()
    bundles = {
        variant: MonitoringV2Bundle.load(root / variant) for variant in V2_VARIANTS
    }
    app = FastAPI(title="Multisensor Monitoring v2 ONNX API", version="2.0.0")

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {
            "status": "READY",
            "variants": list(bundles),
            "outputs": ["standard_forecast_30m", "pattern_probabilities"],
            "data_status": "oracle/sanity",
            "real_data_status": "NOT VERIFIED",
            "locked_test_read": False,
        }

    @app.post("/v2/predict")
    def predict(request: MonitoringV2PredictRequest) -> dict[str, object]:
        if request.variant not in bundles:
            raise HTTPException(status_code=422, detail="unsupported sensor variant")
        if not all(math.isfinite(value) for value in request.features.values()):
            raise HTTPException(status_code=422, detail="features must be finite numbers")
        try:
            result = bundles[request.variant].predict_one(request.features)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        result["request_id"] = request.request_id
        return result

    return app


try:
    app = create_monitoring_v2_app()
except (FileNotFoundError, ValueError, RuntimeError):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Multisensor Monitoring v2 ONNX API", version="2.0.0")

    @app.get("/healthz")
    def unavailable_healthz() -> dict[str, object]:
        raise HTTPException(status_code=503, detail="monitoring-v2 bundles are not mounted")
