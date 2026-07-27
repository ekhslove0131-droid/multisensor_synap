from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from multisensor_ml.receipts import validate_stage_receipt
from multisensor_ml.registry import sha256_file

VersionKind = Literal[
    "dataset",
    "label_set",
    "baseline",
    "standard_type",
    "stage_model",
    "behavior_model",
    "evaluation",
    "prediction",
    "router",
]
ReleaseStatus = Literal["candidate", "champion", "retired"]

VERSION_KINDS: frozenset[str] = frozenset(
    {
        "dataset",
        "label_set",
        "baseline",
        "standard_type",
        "stage_model",
        "behavior_model",
        "evaluation",
        "prediction",
        "router",
    }
)
RELEASE_VERSION_KEYS: tuple[str, ...] = (
    "dataset",
    "label_set",
    "baseline",
    "standard_type",
    "stage_model",
    "behavior_model",
    "router",
)

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS artifact_versions (
    version_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    artifact_uri TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS releases (
    release_id TEXT PRIMARY KEY,
    source_domain TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('candidate', 'champion', 'retired')),
    dataset_version TEXT NOT NULL REFERENCES artifact_versions(version_id),
    label_version TEXT NOT NULL REFERENCES artifact_versions(version_id),
    baseline_version TEXT NOT NULL REFERENCES artifact_versions(version_id),
    standard_type_version TEXT NOT NULL REFERENCES artifact_versions(version_id),
    stage_model_version TEXT NOT NULL REFERENCES artifact_versions(version_id),
    behavior_model_version TEXT NOT NULL REFERENCES artifact_versions(version_id),
    router_version TEXT NOT NULL REFERENCES artifact_versions(version_id),
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS one_champion_per_domain
ON releases(source_domain)
WHERE status = 'champion';

CREATE TABLE IF NOT EXISTS promotion_events (
    promotion_id INTEGER PRIMARY KEY AUTOINCREMENT,
    release_id TEXT NOT NULL REFERENCES releases(release_id),
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL,
    audit_reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS locked_test_runs (
    model_sha256 TEXT NOT NULL,
    dataset_sha256 TEXT NOT NULL,
    audit_reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (model_sha256, dataset_sha256)
);

CREATE TABLE IF NOT EXISTS stage_receipts (
    pipeline_run_id TEXT NOT NULL,
    stage_id TEXT NOT NULL,
    receipt_sha256 TEXT NOT NULL,
    receipt_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (pipeline_run_id, stage_id)
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")


class ModelRegistry:
    """Local immutable artifact and release registry backed by SQLite."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def register_version(
        self,
        *,
        kind: str,
        version_id: str,
        artifact_uri: str,
        artifact_sha256: str,
        metadata: dict[str, object],
    ) -> Literal["CREATED", "REUSED"]:
        if kind not in VERSION_KINDS:
            raise ValueError(f"unsupported version kind: {kind}")
        _validate_sha256(artifact_sha256, "artifact_sha256")
        metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT kind, artifact_uri, artifact_sha256, metadata_json
                FROM artifact_versions
                WHERE version_id = ?
                """,
                (version_id,),
            ).fetchone()
            candidate = (kind, artifact_uri, artifact_sha256, metadata_json)
            if existing is not None:
                stored = (
                    existing["kind"],
                    existing["artifact_uri"],
                    existing["artifact_sha256"],
                    existing["metadata_json"],
                )
                if stored != candidate:
                    raise ValueError(f"immutable version conflict: {version_id}")
                return "REUSED"
            connection.execute(
                """
                INSERT INTO artifact_versions (
                    version_id, kind, artifact_uri, artifact_sha256,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (version_id, *candidate, _now()),
            )
        return "CREATED"

    def create_release(
        self,
        *,
        release_id: str,
        versions: dict[str, str],
        source_domain: str,
    ) -> None:
        if set(versions) != set(RELEASE_VERSION_KEYS):
            raise ValueError(
                "release versions must be exactly: "
                + ", ".join(RELEASE_VERSION_KEYS)
            )
        with self._connect() as connection:
            for key in RELEASE_VERSION_KEYS:
                row = connection.execute(
                    "SELECT kind FROM artifact_versions WHERE version_id = ?",
                    (versions[key],),
                ).fetchone()
                if row is None or row["kind"] != key:
                    raise ValueError(f"release has invalid {key} version: {versions[key]}")
            connection.execute(
                """
                INSERT INTO releases (
                    release_id, source_domain, status, dataset_version,
                    label_version, baseline_version, standard_type_version,
                    stage_model_version, behavior_model_version, router_version,
                    created_at
                ) VALUES (?, ?, 'candidate', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    release_id,
                    source_domain,
                    versions["dataset"],
                    versions["label_set"],
                    versions["baseline"],
                    versions["standard_type"],
                    versions["stage_model"],
                    versions["behavior_model"],
                    versions["router"],
                    _now(),
                ),
            )

    def release_status(self, release_id: str) -> ReleaseStatus:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM releases WHERE release_id = ?",
                (release_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown release: {release_id}")
        return str(row["status"])  # type: ignore[return-value]

    def promote_release(self, release_id: str, *, audit_reason: str) -> None:
        if len(audit_reason.strip()) < 8:
            raise ValueError("promotion audit_reason must contain at least 8 characters")
        with self._connect() as connection:
            release = connection.execute(
                "SELECT source_domain, status FROM releases WHERE release_id = ?",
                (release_id,),
            ).fetchone()
            if release is None:
                raise ValueError(f"unknown release: {release_id}")
            if release["status"] != "candidate":
                raise ValueError(f"only candidate releases can be promoted: {release_id}")
            previous = connection.execute(
                """
                SELECT release_id FROM releases
                WHERE source_domain = ? AND status = 'champion'
                """,
                (release["source_domain"],),
            ).fetchone()
            if previous is not None:
                connection.execute(
                    "UPDATE releases SET status = 'retired' WHERE release_id = ?",
                    (previous["release_id"],),
                )
                connection.execute(
                    """
                    INSERT INTO promotion_events (
                        release_id, from_status, to_status, audit_reason, created_at
                    ) VALUES (?, 'champion', 'retired', ?, ?)
                    """,
                    (previous["release_id"], audit_reason, _now()),
                )
            connection.execute(
                "UPDATE releases SET status = 'champion' WHERE release_id = ?",
                (release_id,),
            )
            connection.execute(
                """
                INSERT INTO promotion_events (
                    release_id, from_status, to_status, audit_reason, created_at
                ) VALUES (?, 'candidate', 'champion', ?, ?)
                """,
                (release_id, audit_reason, _now()),
            )

    def record_locked_test(
        self,
        *,
        model_sha256: str,
        dataset_sha256: str,
        audit_reason: str,
    ) -> None:
        _validate_sha256(model_sha256, "model_sha256")
        _validate_sha256(dataset_sha256, "dataset_sha256")
        if len(audit_reason.strip()) < 8:
            raise ValueError("locked_test audit_reason must contain at least 8 characters")
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO locked_test_runs (
                        model_sha256, dataset_sha256, audit_reason, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (model_sha256, dataset_sha256, audit_reason, _now()),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "duplicate locked_test for model/dataset hash pair"
            ) from exc

    def locked_test_exists(
        self,
        *,
        model_sha256: str,
        dataset_sha256: str,
    ) -> bool:
        _validate_sha256(model_sha256, "model_sha256")
        _validate_sha256(dataset_sha256, "dataset_sha256")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM locked_test_runs
                WHERE model_sha256 = ? AND dataset_sha256 = ?
                """,
                (model_sha256, dataset_sha256),
            ).fetchone()
        return row is not None

    def record_stage_receipt(
        self,
        receipt_path: Path,
    ) -> Literal["CREATED", "REUSED"]:
        receipt = validate_stage_receipt(receipt_path)
        receipt_json = receipt_path.read_text(encoding="utf-8")
        receipt_sha = sha256_file(receipt_path)
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT receipt_sha256 FROM stage_receipts
                WHERE pipeline_run_id = ? AND stage_id = ?
                """,
                (receipt.pipeline_run_id, receipt.stage_id),
            ).fetchone()
            if existing is not None:
                if existing["receipt_sha256"] != receipt_sha:
                    raise ValueError(
                        "immutable stage receipt conflict: "
                        f"{receipt.pipeline_run_id}/{receipt.stage_id}"
                    )
                return "REUSED"
            connection.execute(
                """
                INSERT INTO stage_receipts (
                    pipeline_run_id, stage_id, receipt_sha256,
                    receipt_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    receipt.pipeline_run_id,
                    receipt.stage_id,
                    receipt_sha,
                    receipt_json,
                    _now(),
                ),
            )
        return "CREATED"

    def list_versions(self, *, kind: str | None = None) -> list[dict[str, object]]:
        if kind is not None and kind not in VERSION_KINDS:
            raise ValueError(f"unsupported version kind: {kind}")
        query = """
            SELECT version_id, kind, artifact_uri, artifact_sha256,
                   metadata_json, created_at
            FROM artifact_versions
        """
        parameters: tuple[str, ...] = ()
        if kind is not None:
            query += " WHERE kind = ?"
            parameters = (kind,)
        query += " ORDER BY created_at, version_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            {
                "version_id": row["version_id"],
                "kind": row["kind"],
                "artifact_uri": row["artifact_uri"],
                "artifact_sha256": row["artifact_sha256"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def release_details(self, release_id: str) -> dict[str, object]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM releases WHERE release_id = ?",
                (release_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"unknown release: {release_id}")
        return {key: row[key] for key in row}


def import_oracle_bundle(registry: ModelRegistry, bundle_root: Path) -> str:
    """Register an existing Goal 1.5 Oracle bundle as an explicit legacy candidate."""

    bundle = bundle_root.resolve()
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle_schema") != "goal1.5/oracle-model-bundle/v1":
        raise ValueError("unsupported Oracle bundle schema")
    if manifest.get("status") != "oracle/sanity":
        raise ValueError("Oracle import requires oracle/sanity status")
    lineage = manifest["lineage"]
    files = manifest["files"]
    baseline_path = bundle / "global_baseline.json"
    if sha256_file(baseline_path) != files["global_baseline.json"]:
        raise ValueError("Oracle bundle baseline hash mismatch")

    bundle_sha = sha256_file(manifest_path)
    series_id = str(lineage["prepared_series_id"])
    dataset_sha = str(lineage["prepared_manifest_sha256"])
    suffix = bundle_sha[:12]
    versions = {
        "dataset": f"dataset-{series_id}-{dataset_sha[:12]}",
        "label_set": f"labels-oracle-legacy-{suffix}",
        "baseline": f"baseline-oracle-legacy-{files['global_baseline.json'][:12]}",
        "standard_type": f"standard-type-legacy-k1-{suffix}",
        "stage_model": f"stage-model-oracle-legacy-{suffix}",
        "behavior_model": f"behavior-model-not-available-{suffix}",
        "router": f"router-legacy-en-{suffix}",
    }
    common_metadata: dict[str, object] = {
        "source_domain": "synthetic_truth_oracle",
        "status": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
        "legacy_import": True,
    }
    registrations: dict[str, tuple[str, str, dict[str, object]]] = {
        "dataset": (
            str(lineage["prepared_root"]),
            dataset_sha,
            {"series_id": series_id},
        ),
        "label_set": (
            str(manifest_path),
            bundle_sha,
            {"availability": "legacy_event_binary_and_forecast_60s"},
        ),
        "baseline": (
            str(baseline_path),
            str(files["global_baseline.json"]),
            {"availability": "global_and_personal_legacy"},
        ),
        "standard_type": (
            str(manifest_path),
            bundle_sha,
            {"availability": "NOT_AVAILABLE", "fallback_k": 1},
        ),
        "stage_model": (
            str(manifest_path),
            bundle_sha,
            {"availability": "legacy_binary_and_forecast_models"},
        ),
        "behavior_model": (
            str(manifest_path),
            bundle_sha,
            {"availability": "NOT_AVAILABLE"},
        ),
        "router": (
            str(manifest_path),
            bundle_sha,
            {"availability": "legacy_english_only"},
        ),
    }
    for kind in RELEASE_VERSION_KEYS:
        artifact_uri, artifact_sha, metadata = registrations[kind]
        registry.register_version(
            kind=kind,
            version_id=versions[kind],
            artifact_uri=artifact_uri,
            artifact_sha256=artifact_sha,
            metadata={**common_metadata, **metadata},
        )

    release_material = json.dumps(
        {"series_id": series_id, "versions": versions},
        sort_keys=True,
    ).encode()
    release_id = (
        f"oracle-legacy-{series_id}-"
        f"{hashlib.sha256(release_material).hexdigest()[:12]}"
    )
    try:
        registry.create_release(
            release_id=release_id,
            versions=versions,
            source_domain="synthetic_truth_oracle",
        )
    except sqlite3.IntegrityError:
        if registry.release_status(release_id) not in {
            "candidate",
            "champion",
            "retired",
        }:
            raise
    return release_id
