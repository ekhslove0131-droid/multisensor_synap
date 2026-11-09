"""Single-use locked synthetic evaluation for a frozen beta bundle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import cast

import numpy as np
from synapbridge_beta_runtime.codec import canonical_bytes
from synapbridge_beta_runtime.model_types import SourceStandardModel
from synapbridge_beta_runtime.types import local_path

from .bundle import predict_onnx
from .cohort import load_cohort
from .evaluate import claim_locked_test, locked_claim_path, window_metrics
from .evaluation_contract import validate_bundle_contract, validate_selected_frozen_inputs
from .samples import SampleSet, assemble_samples, calibrate_locked
from .selection import _alarm_matches, _alarm_starts, valid_negative_exposure_hours


def evaluate_beta(bundle_path: Path, dataset_path: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError("beta_evaluation_output_exists")
    bundle_bytes = bundle_path.read_bytes()
    bundle = json.loads(bundle_bytes)
    validate_bundle_contract(bundle)
    if bundle.get("status") == "NO_QUALIFIED_BETA_MODEL":
        output.mkdir(parents=True)
        body: dict[str, object] = {
            "schema_version": "synthetic_beta_evaluation_v1",
            "realm": "SYNTHETIC_BETA", "status": "NO_QUALIFIED_BETA_MODEL",
            "actual_real_performance": "NOT VERIFIED",
            "locked_use": {"status": "NOT_ACCESSED", "use_count": 0},
            "results": [],
        }
        failure_receipt = {
            **body, "content_sha256": hashlib.sha256(canonical_bytes(body)).hexdigest()
        }
        _write(output / "evaluation.json", canonical_bytes(failure_receipt))
        return failure_receipt
    resolved_dataset, dataset_sha = validate_selected_frozen_inputs(
        bundle, dataset_path,
    )
    standards = _load_standards(bundle["standard_refs"])
    output.mkdir(parents=True)
    claim = claim_locked_test(
        locked_claim_path(dataset_path),
        hashlib.sha256(bundle_bytes).hexdigest(),
        dataset_sha,
    )
    cohort = load_cohort(resolved_dataset, ("LOCKED_TEST",))
    references = calibrate_locked(cohort, standards)
    results = []
    all_pass = len(bundle["pattern_models"]) == 14
    for model in bundle["pattern_models"]:
        source_set = tuple(str(model["source_set"]).split("+"))
        head = str(model["head"])
        samples = assemble_samples(cohort, references, source_set, head, "LOCKED_TEST")
        if samples.schema_sha != model["feature_schema_sha256"]:
            raise ValueError("locked_feature_schema_mismatch")
        probabilities = predict_onnx(local_path(model["model_uri"]), samples.x)
        window = window_metrics(samples.y, probabilities, float(model["threshold"]))
        event = _event_metrics(samples, probabilities, float(model["threshold"]), head)
        sensitivity = cast(float, event["sensitivity"])
        far = cast(float, event["far_per_hour"])
        lead = event["median_lead_seconds"]
        gate = (sensitivity >= (0.70 if head == "current" else 0.60)
                and far <= 1.0
                and (head != "onset60"
                     or (lead is not None and cast(float, lead) >= 10.0)))
        all_pass = all_pass and gate
        results.append({"source_set": model["source_set"], "head": head,
                        "model_sha256": model["model_sha256"], "window": window,
                        "event": event, "synthetic_gate_pass": gate})
    body = {
        "schema_version": "synthetic_beta_evaluation_v1", "realm": "SYNTHETIC_BETA",
        "status": ("SYNTHETIC_PERFORMANCE_PASSED" if all_pass
                   else "SYNTHETIC_PERFORMANCE_NOT_PASSED"),
        "actual_real_performance": "NOT VERIFIED", "locked_use": claim,
        "locked_subject_count": len(references),
        "locked_personal_calibration_count": len(references) * 3,
        "total_personal_calibration_count": int(bundle["personal_calibration_count"])
        + len(references) * 3,
        "bootstrap_replicates": 1000, "results": results,
    }
    receipt: dict[str, object] = {
        **body, "content_sha256": hashlib.sha256(canonical_bytes(body)).hexdigest()
    }
    _write(output / "evaluation.json", canonical_bytes(receipt))
    return receipt


def _event_metrics(samples: SampleSet, probability: np.ndarray, threshold: float,
                   head: str) -> dict[str, object]:
    subject_metrics = []
    false_alarms = event_count = matched = 0
    leads: list[float] = []
    response_rows: dict[str, list[tuple[int, int, int]]] = {}
    for subject in samples.event_ranges:
        index = np.asarray([value == subject for value in samples.subjects])
        order = np.argsort(samples.decisions_ms[index])
        times = samples.decisions_ms[index][order]
        alarms = _alarm_starts((probability[index][order] >= threshold), times)
        truth = samples.event_ranges[subject]
        used: set[int] = set()
        local_matched = 0
        for alarm in alarms:
            match = next((number for number, (start, end) in enumerate(truth)
                          if number not in used and _alarm_matches(
                              alarm, start, end, head,
                              60_000 if head == "current" else 0,
                          )), None)
            if match is None:
                false_alarms += 1
            else:
                used.add(match)
                matched += 1
                local_matched += 1
                leads.append((truth[match][0] - alarm) / 1000)
        event_count += len(truth)
        subject_metrics.append(local_matched / len(truth) if truth else 0.0)
        response = samples.response_by_subject[subject]
        response_rows.setdefault(response, []).append(
            (len(truth), local_matched, len(alarms) - local_matched)
        )
    exposure_hours = valid_negative_exposure_hours(samples.y)
    if exposure_hours <= 0:
        raise ValueError("locked_negative_exposure_invalid")
    sensitivity = matched / event_count if event_count else 0.0
    return {"event_count": event_count, "matched": matched,
            "false_alarms": false_alarms, "sensitivity": sensitivity,
            "far_per_hour": false_alarms / exposure_hours,
            "median_lead_seconds": float(np.median(leads)) if leads else None,
            "sensitivity_ci95": _bootstrap_ci(subject_metrics),
            "response_strata": {
                response: {"events": sum(row[0] for row in rows),
                           "matched": sum(row[1] for row in rows),
                           "false_alarms": sum(row[2] for row in rows),
                           "sensitivity": (sum(row[1] for row in rows)
                                           / sum(row[0] for row in rows))}
                for response, rows in sorted(response_rows.items())
            }}


def _bootstrap_ci(values: list[float]) -> list[float] | None:
    if not values:
        return None
    rng = np.random.default_rng(4242)
    array = np.asarray(values, dtype=float)
    means = np.asarray([np.mean(rng.choice(array, size=len(array), replace=True))
                        for _ in range(1000)])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _load_standards(refs: list[dict[str, str]]) -> dict[str, SourceStandardModel]:
    result = {}
    for ref in refs:
        content = local_path(ref["uri"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != ref["sha256"]:
            raise ValueError("standard_ref_sha_mismatch")
        value = json.loads(content)
        result[ref["family"]] = SourceStandardModel(
            value["family"], value["floors"], tuple(value["feature_order"]),
            value["schema_sha"],
        )
    return result


def _write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
