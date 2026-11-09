"""One-use locked-test ledger and metric helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def claim_locked_test(path: Path, bundle_sha256: str,
                      dataset_sha256: str) -> dict[str, object]:
    if path.exists():
        raise ValueError("LOCKED_TEST_ALREADY_USED")
    body = {"schema_version": "synthetic_beta_locked_use_v1",
            "status": "LOCKED_TEST_CLAIMED", "bundle_sha256": bundle_sha256,
            "dataset_sha256": dataset_sha256, "use_count": 1}
    content = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    receipt = {**body, "content_sha256": hashlib.sha256(content).hexdigest()}
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return receipt


def locked_claim_path(dataset_manifest: Path) -> Path:
    resolved = dataset_manifest.resolve()
    if resolved.name != "manifest.json":
        raise ValueError("locked_dataset_manifest_path_invalid")
    return resolved.parent / ".locked-test-use.json"


def window_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float,
) -> dict[str, object]:
    y = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=float)
    if (y.ndim != 1 or probability.shape != y.shape or len(np.unique(y)) != 2
            or not np.isfinite(probability).all() or not 0 < threshold < 1):
        raise ValueError("window_metrics_invalid")
    predicted = probability >= threshold
    return {
        "class_counts": {"negative": int(np.sum(y == 0)), "positive": int(np.sum(y == 1))},
        "auprc": float(average_precision_score(y, probability)),
        "auroc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "ece_10": _ece(y, probability, bins=10),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "tp": int(np.sum(predicted & (y == 1))),
        "fp": int(np.sum(predicted & (y == 0))),
        "fn": int(np.sum(~predicted & (y == 1))),
        "tn": int(np.sum(~predicted & (y == 0))),
    }


def _ece(labels: np.ndarray, probability: np.ndarray, *, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    value = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = ((probability >= lower) & (probability < upper)
                if index < bins - 1 else (probability >= lower) & (probability <= upper))
        if np.any(mask):
            value += float(np.sum(mask) / total
                           * abs(np.mean(labels[mask]) - np.mean(probability[mask])))
    return value
