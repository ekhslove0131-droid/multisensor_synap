from __future__ import annotations

import numpy as np

from multisensor_ml.beta_training.train import fit_candidate_models


def test_lr_and_hgb_fit_attempts_are_real_and_train_only() -> None:
    rng = np.random.default_rng(42)
    train_x = rng.normal(size=(120, 4)).astype(np.float32)
    train_y = (train_x[:, 0] + train_x[:, 1] > 0).astype(np.int64)
    candidates = fit_candidate_models(train_x, train_y, "watch", "current")
    assert {candidate.algorithm for candidate in candidates} == {"LR", "HGB"}
    assert all(candidate.fit_call_count == 1 for candidate in candidates)
    assert all(candidate.input_width == 4 for candidate in candidates)
    classifiers = {candidate.algorithm: candidate.estimator.named_steps["classifier"]
                   for candidate in candidates}
    assert classifiers["LR"].class_weight == "balanced"
    assert classifiers["HGB"].class_weight is None
