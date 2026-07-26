from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


class ProbabilityClassifier(Protocol):
    def predict_proba(self, features: object) -> NDArray[np.float64]: ...


def select_training_rows(
    frame: pd.DataFrame,
    *,
    target: str,
    baseline_ratio: int = 3,
) -> NDArray[np.bool_]:
    """Keep all positive/hard-negative rows plus deterministic matched baselines."""

    positive = frame[target].astype(bool).to_numpy()
    hard_negative = frame["hard_negative"].astype(bool).to_numpy()
    selected = positive | hard_negative
    baseline_candidates = np.flatnonzero(~selected)
    baseline_limit = min(len(baseline_candidates), int(positive.sum()) * baseline_ratio)
    person_keys = frame["person_key"].astype(str).tolist()
    timestamps = frame["timestamp_utc"].astype(str).tolist()
    contexts = frame["context"].astype(str).tolist()
    ordered = sorted(
        baseline_candidates,
        key=lambda index: hashlib.sha256(
            f"{person_keys[index]}:{timestamps[index]}:{contexts[index]}".encode()
        ).digest(),
    )
    selected[np.asarray(ordered[:baseline_limit], dtype=np.int64)] = True
    return selected


def fit_candidate_models(
    features: NDArray[np.float32],
    target: NDArray[np.int8],
    *,
    random_state: int = 20260725,
) -> Mapping[str, ProbabilityClassifier]:
    """Fit the two fixed Goal 1.5 classical candidates."""

    logistic = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )
    histogram = HistGradientBoostingClassifier(
        class_weight="balanced",
        learning_rate=0.08,
        max_iter=160,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=random_state,
    )
    logistic.fit(features, target)
    histogram.fit(features, target)
    return {
        "logistic_regression": cast(ProbabilityClassifier, logistic),
        "hist_gradient_boosting": cast(ProbabilityClassifier, histogram),
    }
