from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    code: str
    severity: str
    message: str
    evidence: dict[str, object]


@dataclass(frozen=True, slots=True)
class ValidationReport:
    status: str
    scope: str
    run_id: str | None
    checks_run: int
    findings: tuple[ValidationFinding, ...]
    metrics: dict[str, object]


def write_validation_report(path: Path, report: ValidationReport) -> None:
    path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
