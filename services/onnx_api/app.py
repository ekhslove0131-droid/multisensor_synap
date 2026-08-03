"""Cloud Run entrypoint for the final two-variant ONNX API."""

from multisensor_ml.cloud_run_onnx import app

__all__ = ["app"]
