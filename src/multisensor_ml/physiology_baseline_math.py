"""Pure TRAIN-only personal center/scale and VALIDATION residual summaries."""

from collections.abc import Sequence
from typing import Any

import numpy as np


def fit_center_scale(
    train_values: Sequence[Sequence[float]],
    validation_values: Sequence[Sequence[float]],
    feature_names: Sequence[str],
) -> tuple[list[dict[str, Any]], str]:
    train = np.asarray(train_values, dtype=float)
    validation = np.asarray(validation_values, dtype=float)
    width = len(feature_names)
    if (
        train.ndim != 2
        or validation.ndim != 2
        or train.shape[1] != width
        or validation.shape[1] != width
        or len(train) == 0
        or len(validation) == 0
        or not np.isfinite(train).all()
        or not np.isfinite(validation).all()
    ):
        raise ValueError("BASELINE_INPUT_INVALID")
    center = np.median(train, axis=0)
    mad_scale = 1.4826 * np.median(np.abs(train - center), axis=0)
    q25, q75 = np.quantile(train, [0.25, 0.75], axis=0, method="linear")
    iqr_scale = (q75 - q25) / 1.349
    scale = np.where(mad_scale > 0, mad_scale, iqr_scale)
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("BASELINE_SCALE_ZERO")
    residuals = (validation - center) / scale
    if not np.isfinite(residuals).all():
        raise ValueError("RESIDUAL_NONFINITE")
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(feature_names):
        absolute = np.abs(residuals[:, index])
        rows.append(
            {
                "feature_name": name,
                "train_count": len(train),
                "validation_count": len(validation),
                "center": float(center[index]),
                "scale": float(scale[index]),
                "scale_source": "MAD_1.4826" if mad_scale[index] > 0 else "IQR_1.349",
                "validation_residuals": residuals[:, index].astype(float).tolist(),
                "validation_mean_residual": float(np.mean(residuals[:, index])),
                "validation_mean_absolute_residual": float(np.mean(absolute)),
                "validation_median_absolute_residual": float(np.median(absolute)),
                "validation_p90_absolute_residual": float(
                    np.quantile(absolute, 0.9, method="linear")
                ),
            }
        )
    return rows, "TRAIN_MEDIAN_MAD_IQR_V1"
