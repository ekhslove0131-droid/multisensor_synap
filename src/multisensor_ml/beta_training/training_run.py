"""Create-only local SYNTHETIC_BETA fitting and immutable bundle export."""

from __future__ import annotations

from pathlib import Path

from synapbridge_beta_runtime.codec import canonical_bytes
from synapbridge_beta_runtime.kernel import runtime_kernel_sha256

from . import artifacts as beta_artifacts
from .bundle import export_candidate_onnx
from .cohort import load_cohort
from .pipeline import BetaTrainConfig, bundle_model_role, expected_fit_attempts
from .samples import assemble_samples, fit_m0
from .selection import choose_candidate
from .stage_training import ValidationHead, select_policy_from_validation
from .train import fit_candidate_models
from .training_evidence import (
    complete_training_evidence,
    write_model_evidence,
    write_policy_evidence,
)


def train_beta(
    dataset_manifest: Path,
    output: Path,
    *,
    wire_root: Path,
    runtime_wheel: Path,
    config: BetaTrainConfig | None = None,
) -> dict[str, object]:
    settings = config or BetaTrainConfig()
    if output.exists():
        raise FileExistsError("beta_training_output_exists")
    output.mkdir(parents=True)
    cohort = load_cohort(dataset_manifest, ("TRAIN", "VALIDATION"))
    standards, references = fit_m0(cohort)
    standard_refs = beta_artifacts.write_standards(output, standards)
    reference_index = beta_artifacts.write_references(output, references)
    config_value = cohort.manifest["config"]
    origin = int(cohort.manifest["timeline_origin_ms"])
    calibration_start = origin + int(config_value["prefill_seconds"]) * 1000
    calibration_end = calibration_start + int(config_value["calibration_seconds"]) * 1000
    lineage = {
        "dataset_manifest_sha256": beta_artifacts.sha256(dataset_manifest.read_bytes()),
        "training_config_sha256": settings.sha256(),
        "generation_protocol_sha256": cohort.manifest["protocol_ref"]["sha256"],
        "kernel_sha256": runtime_kernel_sha256(),
        "runtime_lock_sha256": beta_artifacts.sha256(Path("uv.lock").read_bytes()),
        "standard_refs_sha256": beta_artifacts.sha256(canonical_bytes({
            "standard_refs": standard_refs,
        })),
        "personal_reference_sha256": reference_index["sha256"],
        "calibration_bounds_ms": [calibration_start, calibration_end],
    }
    models, failures, attempts = [], [], 0
    model_evidence_refs: list[dict[str, str]] = []
    stage_inputs: dict[tuple[str, str], ValidationHead] = {}
    for source_set in settings.source_sets:
        source_name = "+".join(source_set)
        for head in settings.heads:
            train = assemble_samples(cohort, references, source_set, head, "TRAIN")
            validation = assemble_samples(cohort, references, source_set, head, "VALIDATION")
            candidates = fit_candidate_models(train.x, train.y, source_name, head)
            attempts += len(candidates)
            probabilities = {
                candidate.algorithm: candidate.estimator.predict_proba(validation.x)[:, 1]
                for candidate in candidates
            }
            try:
                selected = choose_candidate(
                    candidates,
                    probabilities,
                    validation.y,
                    sensitivity_gate=(settings.current_sensitivity_gate
                                      if head == "current"
                                      else settings.onset_sensitivity_gate),
                    subjects=validation.subjects,
                    decisions_ms=validation.decisions_ms,
                    event_ranges=validation.event_ranges,
                )
            except ValueError as error:
                if str(error) != "NO_QUALIFIED_BETA_MODEL":
                    raise
                failures.append({"source_set": source_name, "head": head,
                                 "reason": str(error)})
                continue
            model_path = output / "models" / source_name / f"{head}.onnx"
            try:
                receipt = export_candidate_onnx(
                    selected.candidate, model_path,
                    validation.x[: min(64, len(validation.x))],
                    threshold=selected.threshold,
                )
            except ValueError as error:
                if str(error) not in {"ONNX_PARITY_FAILED", "EXPORT_DECISION_UNSTABLE"}:
                    raise
                failures.append({"source_set": source_name, "head": head,
                                 "algorithm": selected.algorithm,
                                 "reason": str(error), "kind": "EXPORT_REJECTED"})
                alternatives = tuple(candidate for candidate in candidates
                                     if candidate.algorithm != selected.algorithm)
                try:
                    selected = choose_candidate(
                        alternatives, probabilities, validation.y,
                        sensitivity_gate=(settings.current_sensitivity_gate
                                          if head == "current"
                                          else settings.onset_sensitivity_gate),
                        subjects=validation.subjects,
                        decisions_ms=validation.decisions_ms,
                        event_ranges=validation.event_ranges,
                    )
                    receipt = export_candidate_onnx(
                        selected.candidate, model_path,
                        validation.x[: min(64, len(validation.x))],
                        threshold=selected.threshold,
                    )
                except ValueError as fallback_error:
                    failures.append({"source_set": source_name, "head": head,
                                     "reason": str(fallback_error),
                                     "kind": "NO_EXPORTABLE_QUALIFIED_MODEL"})
                    continue
            model = {
                "source_set": source_name, "head": head, "algorithm": selected.algorithm,
                "threshold": selected.threshold, "validation_sensitivity": selected.sensitivity,
                "validation_far_per_hour": selected.far_per_hour,
                "validation_median_lead_seconds": selected.median_lead_seconds,
                "feature_schema_sha256": validation.schema_sha,
                "feature_names": validation.feature_names,
                "model_uri": model_path.resolve().as_uri(), "model_sha256": receipt["sha256"],
                "onnx_receipt": receipt,
            }
            models.append(model)
            model_evidence_refs.append(write_model_evidence(output, model, lineage))
            stage_inputs[(source_name, head)] = (
                validation, probabilities[selected.algorithm], selected.threshold,
            )
    if attempts != expected_fit_attempts(settings):
        raise ValueError("beta_fit_attempt_count_invalid")
    status = "BETA_SELECTED" if len(models) == 14 else "NO_QUALIFIED_BETA_MODEL"
    policy = select_policy_from_validation(stage_inputs)
    policy_evidence_ref = write_policy_evidence(output, policy, lineage)
    if status == "BETA_SELECTED":
        models, policy = complete_training_evidence(
            output, lineage, model_evidence_refs, policy_evidence_ref,
        )
    wheel_ref = beta_artifacts.copy_runtime_wheel(output, runtime_wheel)
    body = {
        "schema_version": "synthetic_beta_model_bundle_v1",
        "realm": "SYNTHETIC_BETA", "status": status,
        "model_role": bundle_model_role(status, len(models)), "allowed_use": "BETA_ONLY",
        "real_data_status": "SYNTHETIC_ONLY_NOT_REAL_VERIFIED", "stage": None,
        "beta_run_uuid": cohort.manifest["beta_run_uuid"],
        "run_epoch": cohort.manifest["run_epoch"],
        "dataset_manifest_sha256": lineage["dataset_manifest_sha256"],
        "training_config_sha256": lineage["training_config_sha256"],
        "generation_protocol_sha256": lineage["generation_protocol_sha256"],
        "standard_fit_count": 3,
        "personal_calibration_count": len(references) * 3,
        "calibration_support_kind": "QUALITY_QUALIFIED_FEATURE_STRIDE",
        "locked_personal_calibration_count": 0,
        "m2_fit_attempt_count": attempts,
        "standard_refs": standard_refs,
        "personal_reference_index": reference_index,
        "pattern_models": models,
        "selection_failures": failures,
        "calibration_bounds_ms": [calibration_start, calibration_end],
        "policy": policy,
        "policy_sha256": beta_artifacts.sha256(canonical_bytes(policy)),
        "wire_contract_refs": beta_artifacts.wire_refs(wire_root),
        "kernel_sha256": runtime_kernel_sha256(),
        "runtime_wheel_ref": wheel_ref,
        "runtime_lock_sha256": lineage["runtime_lock_sha256"],
        "locked_test_used": False,
    }
    manifest = {**body, "content_sha256": beta_artifacts.sha256(canonical_bytes(body))}
    beta_artifacts.write_create_only(output / "manifest.json", canonical_bytes(manifest))
    return manifest
