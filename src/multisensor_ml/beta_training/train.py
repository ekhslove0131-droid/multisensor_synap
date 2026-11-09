"""Pre-registered CPU classifiers for SYNTHETIC_BETA M2 heads."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "LOKY_MAX_CPU_COUNT"):
    os.environ.setdefault(_variable, "1")

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


@dataclass(frozen=True, slots=True)
class BetaCandidate:
    algorithm: str
    source_set: str
    head: str
    estimator: Any
    fit_call_count: int
    input_width: int


def fit_candidate_models(train_x: np.ndarray, train_y: np.ndarray,
                         source_set: str, head: str) -> tuple[BetaCandidate, ...]:
    x = np.asarray(train_x, dtype=np.float32)
    y = np.asarray(train_y, dtype=np.int64)
    if (x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(x) < 2
            or len(np.unique(y)) != 2 or not np.isfinite(x).all()):
        raise ValueError("beta_train_input_invalid")
    estimators = (
        ("LR", LogisticRegression(C=1.0, class_weight="balanced",
                                   solver="lbfgs", max_iter=500, random_state=4242)),
        ("HGB", HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                                learning_rate=0.1, early_stopping=False,
                                                random_state=4242)),
    )
    candidates = []
    for algorithm, classifier in estimators:
        estimator = Pipeline([("scale", StandardScaler()), ("classifier", classifier)])
        estimator.fit(x, y)
        candidates.append(BetaCandidate(algorithm, source_set, head, estimator, 1, x.shape[1]))
    return tuple(candidates)
