from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import nbformat
import pytest

from multisensor_ml.bigquery_training_preflight import EXPECTED_VIEW_FIELDS
from multisensor_ml.observational_contract import FEATURE_NAMES
from multisensor_ml.standard_baseline_bigquery import STANDARD_BASELINE_VIEW_FIELDS

BEHAVIOR_COHORT_UUID = "10000000-0000-4000-8000-000000000001"
STANDARD_COHORT_UUID = "00000000-0000-4000-8000-000000000099"
MODEL_READER = "kidsignal-model-reader@multi-app-kidsignal-260801.iam.gserviceaccount.com"
WATCH_SCHEMA_UUID = "9b842d8c-8889-5259-acca-77baa0c7729d"
WATCH_SCHEMA_HASH = "2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de"
PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOK_BUILDER = PROJECT_ROOT / "scripts/build_kaggle_bigquery_intake_notebook.py"


def _build_notebook() -> nbformat.NotebookNode:
    spec = importlib.util.spec_from_file_location(
        "kidsignal_kaggle_bigquery_intake_notebook_builder",
        NOTEBOOK_BUILDER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load notebook builder: {NOTEBOOK_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_notebook = getattr(module, "build_notebook", None)
    if not callable(build_notebook):
        raise RuntimeError("notebook builder does not expose build_notebook")
    notebook = build_notebook()
    if not isinstance(notebook, nbformat.NotebookNode):
        raise RuntimeError("notebook builder returned an invalid notebook")
    return notebook


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _behavior_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    index = 0
    for split, subject_index in (("TRAIN", 1), ("VALIDATION", 2)):
        for evaluation, disposition in (
            ("POSITIVE", "TARGET_EVENT"),
            ("NEGATIVE", "VALID_NON_EVENT"),
        ):
            index += 1
            rows.append(
                {
                    "training_cohort_uuid": BEHAVIOR_COHORT_UUID,
                    "cohort_digest": "a" * 64,
                    "split_digest": "b" * 64,
                    "public_cohort_digest": "",
                    "public_split_digest": "",
                    "feature_schema_uuid": WATCH_SCHEMA_UUID,
                    "feature_schema_hash": WATCH_SCHEMA_HASH,
                    "split_policy": "PERSON_GROUP",
                    "purge_seconds": 1800,
                    "truth_state": "REVIEWED_REAL",
                    "training_subject_uuid": f"20000000-0000-4000-8000-{subject_index:012d}",
                    "training_capture_set_uuid": f"20000000-0000-4000-8000-{10 + index:012d}",
                    "exact_window_id": f"{index:064x}",
                    "source_set": ["watch"],
                    "window_start_ms": index * 2_000_000,
                    "window_end_ms": index * 2_000_000 + 1_000,
                    "feature_values": {
                        name: float(position + 1)
                        for position, name in enumerate(FEATURE_NAMES)
                    },
                    "label_uuid": f"20000000-0000-4000-8000-{100 + index:012d}",
                    "label_revision_uuid": f"20000000-0000-4000-8000-{200 + index:012d}",
                    "review_uuid": f"20000000-0000-4000-8000-{300 + index:012d}",
                    "review_disposition": disposition,
                    "evaluation_class": evaluation,
                    "temporal_stage": 1,
                    "observation_code": (
                        "TARGET_EVENT" if evaluation == "POSITIVE" else "NO_EVENT"
                    ),
                    "split_role": split,
                    "source_row_digest": f"{1000 + index:064x}",
                }
            )
    members = sorted(
        [
            [
                row["training_subject_uuid"],
                row["training_capture_set_uuid"],
                row["exact_window_id"],
                row["split_role"],
                row["window_start_ms"],
                row["source_row_digest"],
            ]
            for row in rows
        ],
        key=lambda member: (member[0], member[4], member[2]),
    )
    split_digest = _sha256_json(
        [[member[0], member[2], member[3]] for member in members]
    )
    cohort_digest = _sha256_json(
        [
            WATCH_SCHEMA_UUID,
            WATCH_SCHEMA_HASH,
            "PERSON_GROUP",
            1800,
            "REVIEWED_REAL",
            members,
        ]
    )
    for row in rows:
        row["public_cohort_digest"] = cohort_digest
        row["public_split_digest"] = split_digest
    return rows


def _standard_rows() -> list[dict[str, object]]:
    subject = str(uuid5(NAMESPACE_URL, "intake-standard-subject"))
    capture = str(uuid5(NAMESPACE_URL, "intake-standard-capture"))
    rows: list[dict[str, object]] = []
    for index, split in enumerate(("TRAIN", "VALIDATION")):
        target = 0.25 + index * 0.5
        features = {
            name: float(position + 1) / 10.0
            for position, name in enumerate(FEATURE_NAMES)
        }
        features["watch_load_median_300"] = target
        rows.append(
            {
                "standard_cohort_uuid": STANDARD_COHORT_UUID,
                "cohort_digest": "c" * 64,
                "split_digest": "d" * 64,
                "public_cohort_digest": "",
                "public_split_digest": "",
                "stable_standard_version": "stable-stress-standard-v1",
                "feature_schema_uuid": WATCH_SCHEMA_UUID,
                "feature_schema_hash": WATCH_SCHEMA_HASH,
                "split_policy": "CHRONOLOGICAL_PER_SUBJECT",
                "purge_seconds": 1800,
                "eligibility_policy": "ELIGIBLE_NO_PATTERN_REAL",
                "training_subject_uuid": subject,
                "training_capture_set_uuid": capture,
                "standard_hour_uuid": str(
                    uuid5(NAMESPACE_URL, f"intake-standard-hour-{index}")
                ),
                "source_set": ["watch"],
                "hour_start_ms": index * 3 * 3_600_000,
                "hour_end_ms": index * 3 * 3_600_000 + 3_600_000,
                "feature_values": features,
                "target_name": "no_pattern_median",
                "target_value": target,
                "target_unit": "positive_robust_z",
                "valid_coverage_seconds": 3000,
                "split_role": split,
                "source_row_digest": f"{index + 1:064x}",
            }
        )
    members = sorted(
        [
            [
                row["training_subject_uuid"],
                row["training_capture_set_uuid"],
                row["standard_hour_uuid"],
                row["split_role"],
                row["hour_start_ms"],
                row["source_row_digest"],
            ]
            for row in rows
        ]
    )
    split_digest = _sha256_json(
        [[member[0], member[2], member[3]] for member in members]
    )
    cohort_digest = _sha256_json(
        [
            "stable-stress-standard-v1",
            WATCH_SCHEMA_UUID,
            WATCH_SCHEMA_HASH,
            "CHRONOLOGICAL_PER_SUBJECT",
            1800,
            "ELIGIBLE_NO_PATTERN_REAL",
            members,
        ]
    )
    for row in rows:
        row["public_cohort_digest"] = cohort_digest
        row["public_split_digest"] = split_digest
    return rows


class _SchemaField:
    def __init__(self, name: str) -> None:
        self.name = name


class _Row:
    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def items(self):
        return self._values.items()


class _RowIterator:
    def __init__(self, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
        self.schema = tuple(_SchemaField(name) for name in fields)
        self._rows = tuple(
            _Row({field: row[field] for field in fields}) for row in rows
        )

    def __iter__(self):
        return iter(self._rows)


class _Job:
    def __init__(self, iterator: _RowIterator) -> None:
        self._iterator = iterator

    def result(self) -> _RowIterator:
        return self._iterator


class _Client:
    def __init__(self, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
        self._iterator = _RowIterator(rows, fields)
        self._credentials = SimpleNamespace(service_account_email=MODEL_READER)

    def query(self, query: str, **kwargs: object) -> _Job:
        del query, kwargs
        return _Job(self._iterator)


def _fake_bigquery_module(client: _Client) -> ModuleType:
    google = ModuleType("google")
    google.__path__ = []  # type: ignore[attr-defined]
    cloud = ModuleType("google.cloud")
    cloud.__path__ = []  # type: ignore[attr-defined]
    bigquery = ModuleType("google.cloud.bigquery")
    bigquery.Client = lambda **kwargs: client  # type: ignore[attr-defined]
    bigquery.QueryJobConfig = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    bigquery.ScalarQueryParameter = (  # type: ignore[attr-defined]
        lambda name, parameter_type, value: {
            "name": name,
            "type": parameter_type,
            "value": value,
        }
    )
    google.cloud = cloud  # type: ignore[attr-defined]
    cloud.bigquery = bigquery  # type: ignore[attr-defined]
    return google


def _install_fake_bigquery(monkeypatch: pytest.MonkeyPatch, client: _Client) -> None:
    google = _fake_bigquery_module(client)
    cloud = google.cloud  # type: ignore[attr-defined]
    bigquery = cloud.bigquery  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", bigquery)


def _execute_code_cells(notebook: nbformat.NotebookNode) -> dict[str, object]:
    namespace: dict[str, object] = {}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            exec(compile(cell.source, f"<intake-cell-{index}>", "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    ("mode", "cohort_uuid", "rows", "fields", "expected_train_shape"),
    [
        ("standard", STANDARD_COHORT_UUID, _standard_rows, STANDARD_BASELINE_VIEW_FIELDS, [1, 15]),
        ("behavior", BEHAVIOR_COHORT_UUID, _behavior_rows, EXPECTED_VIEW_FIELDS, [2, 16]),
    ],
)
def test_generated_notebook_executes_bounded_sdk_intake_for_both_planes(
    mode: str,
    cohort_uuid: str,
    rows,
    fields: tuple[str, ...],
    expected_train_shape: list[int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = rows()
    client = _Client(values, fields)
    _install_fake_bigquery(monkeypatch, client)
    output = tmp_path / f"{mode}.json"
    monkeypatch.setenv("KIDSIGNAL_INTAKE_MODE", mode)
    monkeypatch.setenv("KIDSIGNAL_COHORT_UUID", cohort_uuid)
    monkeypatch.setenv(
        "KIDSIGNAL_PUBLIC_COHORT_DIGEST", str(values[0]["public_cohort_digest"])
    )
    monkeypatch.setenv(
        "KIDSIGNAL_PUBLIC_SPLIT_DIGEST", str(values[0]["public_split_digest"])
    )
    monkeypatch.setenv("KIDSIGNAL_INTAKE_OUTPUT", str(output))

    _execute_code_cells(_build_notebook())
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["mode"] == mode
    assert artifact["readiness_receipt"]["status"] == "READY_FOR_SYNC_NOT_TRAINED"
    assert artifact["readiness_receipt"]["fit_call_count"] == 0
    assert artifact["summary"]["train_shape"] == expected_train_shape
    serialized = json.dumps(artifact, sort_keys=True)
    for forbidden in (
        "account_uuid",
        "person_uuid",
        "membership_uuid",
        "raw_uri",
        "feature_values",
        "training_subject_uuid",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("mode", "cohort_uuid", "rows", "fields"),
    [
        ("standard", STANDARD_COHORT_UUID, _standard_rows, STANDARD_BASELINE_VIEW_FIELDS),
        ("behavior", BEHAVIOR_COHORT_UUID, _behavior_rows, EXPECTED_VIEW_FIELDS),
    ],
)
def test_intake_consumer_blocks_zero_rows_and_schema_mismatch_without_fit(
    mode: str,
    cohort_uuid: str,
    rows,
    fields: tuple[str, ...],
) -> None:
    from multisensor_ml.kaggle_bigquery_intake import run_kaggle_bigquery_intake

    values = rows()
    common = {
        "mode": mode,
        "cohort_uuid": cohort_uuid,
        "expected_public_cohort_digest": str(values[0]["public_cohort_digest"]),
        "expected_public_split_digest": str(values[0]["public_split_digest"]),
        "observed_at_utc": "2026-08-21T09:00:00Z",
        "observed_principal": MODEL_READER,
        "job_config_factory": lambda name, parameter_type, value: {
            "name": name,
            "type": parameter_type,
            "value": value,
        },
    }

    zero = run_kaggle_bigquery_intake(client=_Client([], fields), **common)
    mismatch = run_kaggle_bigquery_intake(
        client=_Client([], tuple(reversed(fields))), **common
    )
    digest_mismatch = run_kaggle_bigquery_intake(
        client=_Client(values, fields),
        **{
            **common,
            "expected_public_cohort_digest": "e" * 64,
        },
    )

    assert zero["readiness_receipt"]["status"] == "BLOCKED_NO_REAL_COHORT"
    assert zero["readiness_receipt"]["fit_call_count"] == 0
    assert zero["summary"]["train_shape"] is None
    assert mismatch["readiness_receipt"]["status"] == (
        "BLOCKED_SCHEMA_CONTRACT_MISMATCH"
    )
    assert mismatch["readiness_receipt"]["fit_call_count"] == 0
    assert mismatch["summary"]["train_shape"] is None
    assert digest_mismatch["readiness_receipt"]["status"] == (
        "BLOCKED_EXPECTED_PUBLIC_DIGEST_MISMATCH"
    )
    assert digest_mismatch["readiness_receipt"]["fit_call_count"] == 0
    assert digest_mismatch["summary"]["train_shape"] is None


def test_notebook_fails_clearly_when_bigquery_sdk_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KIDSIGNAL_INTAKE_MODE", "standard")
    monkeypatch.setenv("KIDSIGNAL_COHORT_UUID", STANDARD_COHORT_UUID)
    monkeypatch.setenv("KIDSIGNAL_PUBLIC_COHORT_DIGEST", "a" * 64)
    monkeypatch.setenv("KIDSIGNAL_PUBLIC_SPLIT_DIGEST", "b" * 64)
    monkeypatch.setenv("KIDSIGNAL_INTAKE_OUTPUT", str(tmp_path / "missing.json"))
    real_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object):
        if name in {"google", "google.cloud", "google.cloud.bigquery"}:
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(RuntimeError, match="google-cloud-bigquery"):
        _execute_code_cells(_build_notebook())


def test_notebook_fails_clearly_when_adc_principal_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _standard_rows()
    client = _Client(values, STANDARD_BASELINE_VIEW_FIELDS)
    client._credentials = SimpleNamespace()
    _install_fake_bigquery(monkeypatch, client)
    monkeypatch.setenv("KIDSIGNAL_INTAKE_MODE", "standard")
    monkeypatch.setenv("KIDSIGNAL_COHORT_UUID", STANDARD_COHORT_UUID)
    monkeypatch.setenv(
        "KIDSIGNAL_PUBLIC_COHORT_DIGEST", str(values[0]["public_cohort_digest"])
    )
    monkeypatch.setenv(
        "KIDSIGNAL_PUBLIC_SPLIT_DIGEST", str(values[0]["public_split_digest"])
    )
    monkeypatch.setenv("KIDSIGNAL_INTAKE_OUTPUT", str(tmp_path / "missing-adc.json"))

    with pytest.raises(RuntimeError, match="ADC model-reader principal"):
        _execute_code_cells(_build_notebook())


def test_on_disk_notebook_is_generated_and_valid() -> None:
    path = Path(__file__).parents[1] / "kaggle/10_kidsignal_bigquery_intake.ipynb"
    generated = _build_notebook()
    actual = nbformat.read(path, as_version=4)

    nbformat.validate(generated)
    nbformat.validate(actual)
    assert json.loads(nbformat.writes(generated)) == json.loads(nbformat.writes(actual))
