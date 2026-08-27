from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from multisensor_ml.model_sync_receipt import parse_model_sync_receipt

WATCH_SCHEMA_UUID = "9b842d8c-8889-5259-acca-77baa0c7729d"
WATCH_SCHEMA_HASH = "2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de"


def _seal(body: dict[str, object]) -> dict[str, object]:
    receipt = deepcopy(body)
    receipt["handoff_digest"] = hashlib.sha256(
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return receipt


def _receipt(plane: str = "standard") -> dict[str, object]:
    return _seal(
        {
            "contract_version": 1,
            "status": "READY_FOR_MODEL_SYNC_NOT_TRAINED",
            "plane": plane,
            "cohort_uuid": (
                "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
                if plane == "standard"
                else "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
            ),
            "public_cohort_digest": "a" * 64,
            "public_split_digest": "b" * 64,
            "feature_schema_uuid": WATCH_SCHEMA_UUID,
            "feature_schema_hash": WATCH_SCHEMA_HASH,
            "split_policy": "CHRONOLOGICAL_PER_SUBJECT",
            "purge_seconds": 1800,
            "authorized_field_count": 24 if plane == "standard" else 26,
            "train_row_count": 2,
            "validation_row_count": 1,
            "fit_call_count": 0,
        }
    )


@pytest.mark.parametrize(
    ("plane", "expected_field_count"),
    [("standard", 24), ("behavior", 26)],
)
def test_parser_accepts_hash_closed_standard_and_behavior_receipts(
    plane: str, expected_field_count: int
) -> None:
    parsed = parse_model_sync_receipt(_receipt(plane))

    assert parsed.plane == plane
    assert parsed.authorized_field_count == expected_field_count
    assert parsed.fit_call_count == 0
    assert parsed.train_row_count == 2
    assert parsed.validation_row_count == 1


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_parser_rejects_missing_or_extra_fields(mutation: str) -> None:
    receipt = _receipt()
    if mutation == "missing":
        receipt.pop("public_split_digest")
    else:
        receipt["person_uuid"] = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"

    with pytest.raises(ValueError, match="field"):
        parse_model_sync_receipt(receipt)


def test_parser_rejects_handoff_digest_mismatch() -> None:
    receipt = _receipt()
    receipt["train_row_count"] = 3

    with pytest.raises(ValueError, match="handoff_digest"):
        parse_model_sync_receipt(receipt)


@pytest.mark.parametrize("value", ["f" * 63, "F" * 64])
def test_parser_rejects_malformed_handoff_digest(value: str) -> None:
    receipt = _receipt()
    receipt["handoff_digest"] = value

    with pytest.raises(ValueError, match="handoff_digest"):
        parse_model_sync_receipt(receipt)


@pytest.mark.parametrize(
    ("plane", "field_count"),
    [("standard", 26), ("behavior", 24), ("unknown", 24)],
)
def test_parser_rejects_plane_and_authorized_field_count_drift(
    plane: str, field_count: int
) -> None:
    receipt = _receipt("standard")
    receipt["plane"] = plane
    receipt["authorized_field_count"] = field_count
    receipt = _seal({key: value for key, value in receipt.items() if key != "handoff_digest"})

    with pytest.raises(ValueError, match=r"plane|authorized_field_count"):
        parse_model_sync_receipt(receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [("train_row_count", 0), ("validation_row_count", -1), ("fit_call_count", 1)],
)
def test_parser_rejects_nonpositive_split_support_or_nonzero_fit(
    field: str, value: int
) -> None:
    receipt = _receipt()
    receipt[field] = value
    receipt = _seal({key: item for key, item in receipt.items() if key != "handoff_digest"})

    with pytest.raises(ValueError, match=field):
        parse_model_sync_receipt(receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cohort_uuid", "not-a-uuid"),
        ("cohort_uuid", "550e8400-e29b-51d4-a716-446655440000"),
        ("feature_schema_uuid", "not-a-uuid"),
        ("public_cohort_digest", "A" * 64),
        ("public_split_digest", "b" * 63),
        ("feature_schema_hash", "f" * 64),
    ],
)
def test_parser_rejects_malformed_identity_or_unsupported_schema(
    field: str, value: object
) -> None:
    receipt = _receipt()
    receipt[field] = value
    receipt = _seal({key: item for key, item in receipt.items() if key != "handoff_digest"})

    with pytest.raises(ValueError, match=field):
        parse_model_sync_receipt(receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract_version", 2),
        ("status", "TRAINED"),
        ("split_policy", "ROW_RANDOM"),
        ("purge_seconds", 1799),
    ],
)
def test_parser_rejects_version_status_or_split_contract_drift(
    field: str, value: object
) -> None:
    receipt = _receipt()
    receipt[field] = value
    receipt = _seal({key: item for key, item in receipt.items() if key != "handoff_digest"})

    with pytest.raises(ValueError, match=field):
        parse_model_sync_receipt(receipt)
