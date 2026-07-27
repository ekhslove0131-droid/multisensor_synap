from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ReceiptStatus = Literal["SUCCESS", "REUSED", "FAILED"]


class StageReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["goal1.5/stage-receipt/v1"] = (
        "goal1.5/stage-receipt/v1"
    )
    pipeline_run_id: str = Field(min_length=1)
    stage_id: str = Field(min_length=1)
    status: ReceiptStatus
    artifact_uri: str
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    versions: dict[str, str]
    started_at: datetime
    finished_at: datetime
    message_ko: str = Field(min_length=1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def create_stage_receipt(
    *,
    pipeline_run_id: str,
    stage_id: str,
    artifact_uri: Path,
    output_path: Path,
    status: ReceiptStatus,
    versions: dict[str, str],
    message_ko: str,
) -> StageReceipt:
    if output_path.exists():
        raise FileExistsError(f"receipt already exists: {output_path}")
    artifact = artifact_uri.resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"receipt artifact is not a file: {artifact}")
    now = datetime.now(UTC)
    receipt = StageReceipt(
        pipeline_run_id=pipeline_run_id,
        stage_id=stage_id,
        status=status,
        artifact_uri=str(artifact),
        artifact_sha256=_sha256_file(artifact),
        versions=versions,
        started_at=now,
        finished_at=now,
        message_ko=message_ko,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def validate_stage_receipt(path: Path) -> StageReceipt:
    receipt = StageReceipt.model_validate_json(path.read_text(encoding="utf-8"))
    artifact = Path(receipt.artifact_uri)
    if not artifact.is_file():
        raise ValueError(f"receipt artifact is missing: {artifact}")
    if _sha256_file(artifact) != receipt.artifact_sha256:
        raise ValueError(f"artifact hash mismatch: {artifact}")
    return receipt
