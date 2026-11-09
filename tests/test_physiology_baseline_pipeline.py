import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from multisensor_ml.physiology_baseline_pipeline import run_physiology_baseline
from multisensor_ml.physiology_handoff import VIEW_FIELDS

ROOT = Path(__file__).resolve().parents[1] / "contracts/physiology/v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def fixture(source="h10"):
    filename = {
        "h10": "h10_physio_core_v1",
        "watch": "watch_physio_core_v1",
        "muse": "muse_eeg_core_v1",
    }[source]
    schema = json.loads((ROOT / f"{filename}.json").read_text())
    envelope = dict(
        cohort_uuid="00000000-0000-4000-8000-000000000001",
        source_kind=source,
        feature_schema_uuid=schema["schema_uuid"],
        feature_schema_hash=digest(schema),
        native_hr_policy="polar_native_hr_bpm_v1" if source == "h10" else "NOT_APPLICABLE",
        split_policy="CHRONOLOGICAL_PER_SUBJECT",
        purge_seconds=10,
        eligibility_policy="REVIEWED_NO_PATTERN_V1",
        origin="CONTRACT_FIXTURE",
    )
    rows = []
    for i, (start, role, value) in enumerate(
        [(0, "TRAIN", 0.2), (10000, "TRAIN", 0.4), (30000, "VALIDATION", 0.9)]
    ):
        row = dict(
            envelope,
            training_subject_uuid="00000000-0000-5000-8000-000000000002",
            training_capture_set_uuid="00000000-0000-5000-8000-000000000003",
            exact_window_id=str(i + 1) * 64,
            source_row_digest=str(i + 4) * 64,
            window_start_ms=start,
            window_end_ms=start + 10000,
            support_start_ms=start,
            support_end_ms=start + 10000,
            feature_values={name: value for name in schema["core_features"]},
            quality_valid=True,
            training_eligible=True,
            admission="ELIGIBLE_NO_PATTERN",
            no_pattern_review_uuid="00000000-0000-4000-8000-000000000004",
            split_role=role,
        )
        rows.append(row)
    return schema, *seal(envelope, rows)


def seal(envelope, rows):
    members = [
        {k: v for k, v in r.items() if k not in {"public_cohort_digest", "public_split_digest"}}
        for r in rows
    ]
    members.sort(
        key=lambda r: (r["training_subject_uuid"], r["window_start_ms"], r["exact_window_id"])
    )
    split = digest(
        [
            [
                r["training_subject_uuid"],
                r["training_capture_set_uuid"],
                r["exact_window_id"],
                r["split_role"],
                r["support_start_ms"],
                r["support_end_ms"],
            ]
            for r in members
        ]
    )
    cohort = digest({"envelope": envelope, "rows": members})
    for r in rows:
        r.update(public_cohort_digest=cohort, public_split_digest=split)
    receipt = dict(
        envelope,
        contract_version="physiology_baseline_handoff_v1",
        public_cohort_digest=cohort,
        public_split_digest=split,
        train_row_count=sum(r["split_role"] == "TRAIN" for r in rows),
        validation_row_count=sum(r["split_role"] == "VALIDATION" for r in rows),
        locked_access=False,
        fit_call_count=0,
    )
    receipt["handoff_digest"] = digest(receipt)
    return receipt, [{k: r[k] for k in VIEW_FIELDS} for r in rows]


class Row:
    def __init__(self, row):
        self.row = row

    def items(self):
        return self.row.items()


class Client:
    def __init__(self, rows, fields=VIEW_FIELDS):
        self.rows, self.fields, self.calls = rows, fields, []

    def query(self, query, **kwargs):
        self.calls.append((query, kwargs))

        class Result:
            schema = tuple(SimpleNamespace(name=f) for f in self.fields)

            def __iter__(inner):
                return iter(Row(r) for r in self.rows)

        return SimpleNamespace(result=Result)


def run(schema, receipt, client, **kwargs):
    return run_physiology_baseline(
        schema=schema,
        receipt=receipt,
        client=client,
        job_config_factory=lambda *args: args,
        evidence_mode="CONTRACT_FIXTURE",
        **kwargs,
    )


@pytest.mark.parametrize("source,width", [("watch", 3), ("h10", 3), ("muse", 24)])
def test_single_source_train_only_baseline_and_bound_query(source, width):
    schema, receipt, rows = fixture(source)
    client = Client(rows)
    out = run(schema, receipt, client)
    assert out["status"] == "CONTRACT_EVIDENCE_ONLY"
    assert out["fit_call_count"] == 0 and out["fixture_fit_call_count"] == 1
    model = out["models"][0]
    assert model["center"] == pytest.approx([0.3] * width)
    assert model["scale"] == pytest.approx([0.14826] * width)
    assert out["validation"][0]["mean_absolute_residual"] == pytest.approx(
        [(0.9 - 0.3) / 0.14826] * width
    )
    query, options = client.calls[0]
    assert "kidsignal_model_training.physiology_baseline_train_validation_v1" in query
    assert receipt["cohort_uuid"] not in query
    assert options["job_config"] == ("cohort_uuid", "STRING", receipt["cohort_uuid"])
    assert "kidsignal_training_private" not in query


def test_validation_values_do_not_change_center_scale():
    schema, receipt, rows = fixture()
    original = run(schema, receipt, Client(rows))["models"]
    rows[-1]["feature_values"] = dict.fromkeys(schema["core_features"], 999.0)
    env = {
        k: receipt[k]
        for k in rows[0]
        if k in receipt and k not in {"public_cohort_digest", "public_split_digest"}
    }
    receipt, rows = seal(env, rows)
    assert run(schema, receipt, Client(rows))["models"] == original


@pytest.mark.parametrize(
    "fault",
    [
        "review",
        "quality",
        "locked",
        "private",
        "digest",
        "support",
        "duplicate",
        "native",
        "subject",
    ],
)
def test_fail_closed_before_fitting(fault):
    schema, receipt, rows = fixture()
    if fault == "review":
        rows[0]["admission"] = "REVIEW_REQUIRED"
    elif fault == "quality":
        rows[0]["quality_valid"] = False
    elif fault == "locked":
        rows[-1]["split_role"] = "LOCKED"
    elif fault == "private":
        rows[0]["person_uuid"] = "private"
    elif fault == "digest":
        rows[0]["feature_values"]["hr_median_bpm"] = 100.0
    elif fault == "support":
        rows[-1]["support_start_ms"] = 19000
    elif fault == "duplicate":
        rows[1]["exact_window_id"] = rows[0]["exact_window_id"]
    elif fault == "native":
        rows[0]["native_hr_policy"] = "corrected_hr_bpm"
    else:
        rows[-1]["training_subject_uuid"] = "00000000-0000-5000-8000-000000000005"
    out = run(schema, receipt, Client(rows))
    assert out["status"].startswith("BLOCKED")
    assert out["fit_call_count"] == out["fixture_fit_call_count"] == 0
    assert "models" not in out


def test_zero_rows_and_real_authorization_gate():
    schema, receipt, _ = fixture()
    assert run(schema, receipt, Client([]))["status"] == "BLOCKED_NO_REAL_COHORT"
    client = Client([])
    out = run_physiology_baseline(
        schema=schema, receipt=receipt, client=client, evidence_mode="REAL_OBSERVED"
    )
    assert out["status"].startswith("BLOCKED") and not client.calls


def test_degenerate_train_cannot_borrow_validation_scale():
    schema, receipt, rows = fixture()
    rows[1]["feature_values"] = copy.deepcopy(rows[0]["feature_values"])
    env = {
        k: receipt[k]
        for k in rows[0]
        if k in receipt and k not in {"public_cohort_digest", "public_split_digest"}
    }
    receipt, rows = seal(env, rows)
    out = run(schema, receipt, Client(rows))
    assert out["status"] == "BLOCKED_BASELINE_SCALE_ZERO"
    assert out["fit_call_count"] == 0
    assert "models" not in out


def test_export_roundtrip_create_only_and_identity(tmp_path):
    from multisensor_ml.physiology_baseline_artifact import read_baseline_artifact, residual

    schema, receipt, rows = fixture()
    request = dict(
        model_release_uuid="00000000-0000-4000-8000-000000000006",
        generated_at="2026-09-06T00:00:00Z",
    )
    out = run(
        schema, receipt, Client(rows), export_request=request, output_path=tmp_path / "first.json"
    )
    assert out["artifact_sha256"]
    artifact = read_baseline_artifact(tmp_path / "first.json")
    subject = rows[0]["training_subject_uuid"]
    assert residual(artifact, subject, (0.9, 0.9, 0.9)) == pytest.approx([0.6 / 0.14826] * 3)
    with pytest.raises(ValueError):
        residual(artifact, "unknown", (0.9, 0.9, 0.9))
    run(
        schema,
        receipt,
        Client(list(reversed(rows))),
        export_request=request,
        output_path=tmp_path / "second.json",
    )
    assert (tmp_path / "first.json").read_bytes() == (tmp_path / "second.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(
            schema,
            receipt,
            Client(rows),
            export_request=request,
            output_path=tmp_path / "first.json",
        )
    document = json.loads((tmp_path / "first.json").read_text())
    document["result"]["models"][0]["center"][0] += 1
    (tmp_path / "tampered.json").write_text(json.dumps(document))
    with pytest.raises(ValueError):
        read_baseline_artifact(tmp_path / "tampered.json")
    document["result"]["models"][0]["scale"][0] = 0
    document["content_sha256"] = digest(
        {k: v for k, v in document.items() if k != "content_sha256"}
    )
    with pytest.raises(ValueError, match="SCALE"):
        residual(document, subject, (0.9, 0.9, 0.9))


def test_semantic_purge_violation_rejected_even_with_matching_digests():
    schema, receipt, rows = fixture()
    rows[-1].update(
        window_start_ms=20000, window_end_ms=30000, support_start_ms=20000, support_end_ms=30000
    )
    env = {
        k: receipt[k]
        for k in rows[0]
        if k in receipt and k not in {"public_cohort_digest", "public_split_digest"}
    }
    receipt, rows = seal(env, rows)
    out = run(schema, receipt, Client(rows))
    assert out["status"] == "BLOCKED_PURGE_OR_SUPPORT_LEAKAGE"
    assert out["fixture_fit_call_count"] == 0


def test_zero_purge_is_not_silently_replaced_by_legacy_minimum():
    schema, receipt, rows = fixture()
    for row in rows:
        row["purge_seconds"] = 0
    rows[-1].update(
        window_start_ms=20000, window_end_ms=30000, support_start_ms=20000, support_end_ms=30000
    )
    env = {
        k: receipt[k]
        for k in rows[0]
        if k in receipt and k not in {"public_cohort_digest", "public_split_digest"}
    }
    env["purge_seconds"] = 0
    receipt, rows = seal(env, rows)
    assert run(schema, receipt, Client(rows))["status"] == "CONTRACT_EVIDENCE_ONLY"


def test_real_origin_still_requires_explicit_execution_gate_before_query():
    schema, receipt, rows = fixture()
    for row in rows:
        row["origin"] = "REAL_OBSERVED"
    env = {
        k: receipt[k]
        for k in rows[0]
        if k in receipt and k not in {"public_cohort_digest", "public_split_digest"}
    }
    env["origin"] = "REAL_OBSERVED"
    receipt, rows = seal(env, rows)
    client = Client(rows)
    out = run_physiology_baseline(
        schema=schema, receipt=receipt, client=client, evidence_mode="REAL_OBSERVED"
    )
    assert out["status"] == "BLOCKED_REAL_FIT_NOT_AUTHORIZED"
    assert out["fit_call_count"] == 0 and not client.calls


def test_bad_receipt_and_schema_fail_before_model_use():
    schema, receipt, rows = fixture()
    client = Client(rows)
    receipt["train_row_count"] = 5
    assert run(schema, receipt, client)["status"] == "BLOCKED_RECEIPT_DIGEST"
    assert not client.calls
    schema, receipt, rows = fixture()
    out = run(schema, receipt, Client(rows, tuple(reversed(VIEW_FIELDS))))
    assert out["status"] == "BLOCKED_VIEW_SCHEMA_ORDER"
    assert out["fixture_fit_call_count"] == 0


def test_review_required_is_rejected_even_after_resealing():
    schema, receipt, rows = fixture()
    rows[0]["admission"] = "REVIEW_REQUIRED"
    env = {
        k: receipt[k]
        for k in rows[0]
        if k in receipt and k not in {"public_cohort_digest", "public_split_digest"}
    }
    receipt, rows = seal(env, rows)
    out = run(schema, receipt, Client(rows))
    assert out["status"] == "BLOCKED_NO_PATTERN_OR_QUALITY_UNAPPROVED"
    assert out["fixture_fit_call_count"] == 0
