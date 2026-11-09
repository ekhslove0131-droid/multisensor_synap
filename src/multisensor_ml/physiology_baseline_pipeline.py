"""Receipt-bound individual physiology baseline; explicit real-data execution gate."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from multisensor_ml.bigquery_sdk_reader import (
    BigQueryClientProtocol,
    QueryJobConfigFactory,
    execute_read_only_sdk_query,
)
from multisensor_ml.physiology_baseline_artifact import (
    _write_baseline_artifact,
    validate_export_request,
)
from multisensor_ml.physiology_baseline_math import fit_center_scale
from multisensor_ml.physiology_handoff import (
    LOCATION,
    PROJECT,
    VIEW_FIELDS,
    PreparedPhysiologyDataset,
    prepare_dataset,
    query_contract,
    validate_receipt,
)


def _fit_baselines(prepared: PreparedPhysiologyDataset) -> tuple[list[Any], list[Any]]:
    if not isinstance(prepared, PreparedPhysiologyDataset):
        raise ValueError("PREPARED_DATASET_REQUIRED")
    models, validation = [], []
    for subject in sorted({s for s, _, _ in prepared.samples}):
        train = np.asarray(
            [v for s, role, v in prepared.samples if s == subject and role == "TRAIN"], dtype=float
        )
        valid = np.asarray(
            [v for s, role, v in prepared.samples if s == subject and role == "VALIDATION"],
            dtype=float,
        )
        statistics, _ = fit_center_scale(train.tolist(), valid.tolist(), prepared.feature_names)
        models.append(
            dict(
                training_subject_uuid=subject,
                center=[row["center"] for row in statistics],
                scale=[row["scale"] for row in statistics],
                train_row_count=len(train),
            )
        )
        validation.append(
            dict(
                training_subject_uuid=subject,
                row_count=len(valid),
                mean_absolute_residual=[
                    row["validation_mean_absolute_residual"] for row in statistics
                ],
                mean_residual=[row["validation_mean_residual"] for row in statistics],
            )
        )
    return models, validation


def run_physiology_baseline(
    *,
    receipt: Mapping[str, Any],
    schema: Mapping[str, Any],
    client: BigQueryClientProtocol,
    evidence_mode: str,
    allow_real_fit: bool = False,
    job_config_factory: QueryJobConfigFactory | None = None,
    export_request: Mapping[str, Any] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Only public fit entrypoint: read/bind before calling the internal trainer.

    CONTRACT_FIXTURE permits test-only numerical fitting, with separate counters.
    REAL_OBSERVED additionally requires explicit allow_real_fit=True. Neither mode
    can access private tables or LOCKED rows, select thresholds, or promote models.
    """
    out: dict[str, Any] = dict(
        status="BLOCKED_NO_REAL_COHORT",
        fit_call_count=0,
        fixture_fit_call_count=0,
        evaluation_status="NOT EVALUABLE",
        promotion_eligible=False,
        delivery_eligible=False,
    )
    try:
        if (export_request is None) != (output_path is None):
            raise ValueError("EXPORT_REQUEST_REQUIRED")
        if export_request is not None:
            validate_export_request(export_request)
        if output_path is not None and output_path.exists():
            raise FileExistsError(output_path)
        bound = validate_receipt(receipt, schema)
        if evidence_mode not in ("CONTRACT_FIXTURE", "REAL_OBSERVED") or (
            evidence_mode != bound["origin"]
        ):
            raise ValueError("EVIDENCE_MODE_MISMATCH")
        if evidence_mode == "REAL_OBSERVED" and allow_real_fit is not True:
            raise ValueError("REAL_FIT_NOT_AUTHORIZED")
        result = execute_read_only_sdk_query(
            client=client,
            query_contract=query_contract(bound),
            project=PROJECT,
            location=LOCATION,
            job_config_factory=job_config_factory,
        )
        if result.schema_fields != VIEW_FIELDS:
            raise ValueError("VIEW_SCHEMA_ORDER")
        prepared = prepare_dataset(result.rows, bound, schema)
        counter = "fit_call_count" if evidence_mode == "REAL_OBSERVED" else "fixture_fit_call_count"
        out[counter] = 1
        models, validation = _fit_baselines(prepared)
        out.update(
            status="TRAINED_NOT_EVALUATED"
            if evidence_mode == "REAL_OBSERVED"
            else "CONTRACT_EVIDENCE_ONLY",
            models=models,
            validation=validation,
            feature_names=list(prepared.feature_names),
            receipt=json.loads(prepared.receipt_json),
            task="individual_physiology_center_scale_residual_v1",
            scale_method="MAD_1.4826_ELSE_IQR_1.349_LINEAR_V1",
            validation_semantics="DESCRIPTIVE_RESIDUALS_NOT_PREDICTION_ACCURACY",
            real_data_status="NOT VERIFIED",
            stage=None,
        )
        if export_request is not None and output_path is not None:
            checksum = _write_baseline_artifact(output_path, export_request, out, schema)
            out["artifact_sha256"] = checksum
    except (ValueError, TypeError, KeyError, RuntimeError) as error:
        # Do not reflect SQL/service exceptions or row values into public receipts.
        safe = str(error)
        reason = safe if safe.isupper() and safe.replace("_", "").isalpha() else "INTAKE_INVALID"
        out["status"] = f"BLOCKED_{reason}"
    return out
