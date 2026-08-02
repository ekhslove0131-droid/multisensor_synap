"""Build a reproducible Korean comparison report for the verified model artifacts.

The report intentionally reads validation artifacts only.  It does not merge
synthetic oracle metrics with Neon observations or with the locked test split.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, cast

import pandas as pd

PROFILE_LABELS: dict[str, str] = {
    "watch_only": "Watch 단독",
    "polar_only": "Polar 단독",
    "muse_only": "Muse 단독",
    "watch_polar": "Watch + Polar",
    "watch_muse": "Watch + Muse",
    "polar_muse": "Polar + Muse",
    "watch_polar_muse": "Watch + Polar + Muse",
}

_METRIC_COLUMNS: tuple[str, ...] = (
    "aucpr",
    "event_recall",
    "event_f1",
    "false_alerts_per_hour",
    "brier_score",
    "calibration_error",
    "macro_f1",
)

_CHARTS: tuple[dict[str, Any], ...] = (
    {
        "id": "event_aucpr",
        "title": "사건 AUCPR 비교",
        "subtitle": "validation · oracle/sanity · 사건 head",
        "metric": "aucpr",
        "head": "event",
        "unit": "AUCPR",
    },
    {
        "id": "event_recall",
        "title": "사건 event recall 비교",
        "subtitle": "validation · oracle/sanity · 사건 단위 recall",
        "metric": "event_recall",
        "head": "event",
        "unit": "recall",
    },
    {
        "id": "event_f1",
        "title": "사건 event F1 비교",
        "subtitle": "validation · oracle/sanity · threshold는 각 모델 validation 기준",
        "metric": "event_f1",
        "head": "event",
        "unit": "F1",
    },
    {
        "id": "false_alerts_per_hour",
        "title": "시간당 false alerts 비교",
        "subtitle": "낮을수록 좋음 · validation · oracle/sanity",
        "metric": "false_alerts_per_hour",
        "head": "event",
        "unit": "alerts/hour",
    },
    {
        "id": "calibration",
        "title": "확률 보정 오차 비교",
        "subtitle": "낮을수록 좋음 · Brier score와 ECE를 함께 표시",
        "metric": "calibration",
        "head": "event",
        "unit": "error",
    },
    {
        "id": "stage_macro_f1",
        "title": "5단계 macro-F1 비교",
        "subtitle": "NO_EVENT 포함 5단계 decoder · validation · oracle/sanity",
        "metric": "macro_f1",
        "head": "stage",
        "unit": "macro-F1",
    },
)


def _as_float(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_metric_file(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"comparison metric artifact is missing: {path}")
    frame = pd.read_parquet(path)
    missing = {"head", "model_name"}.difference(frame.columns)
    if missing:
        raise ValueError(f"comparison metric artifact missing columns {sorted(missing)}: {path}")
    for column in _METRIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame


def _normalise_row(
    row: dict[str, object],
    *,
    model_id: str,
    label: str,
    sensor_profile: str,
    model_name: str,
) -> dict[str, object]:
    result: dict[str, object] = {
        "model_id": model_id,
        "model_label_ko": label,
        "sensor_profile": sensor_profile,
        "head": str(row["head"]),
        "model_name": model_name,
        "role": "validation",
        "data_status": "oracle/sanity",
        "locked_test_read": False,
    }
    result.update({column: _as_float(row.get(column)) for column in _METRIC_COLUMNS})
    return result


def collect_comparison_metrics(project_root: Path) -> pd.DataFrame:
    """Collect current validation metrics for all sensor profiles and baseline models."""

    root = project_root.resolve()
    rows: list[dict[str, object]] = []
    profile_root = (
        root
        / "artifacts"
        / "availability-model"
        / "oracle-sanity-v3"
        / "extracted"
        / "profiles"
    )
    for profile_id, label in PROFILE_LABELS.items():
        frame = _read_metric_file(profile_root / profile_id / "validation_metrics.parquet")
        for raw in frame.to_dict(orient="records"):
            row: dict[str, object] = {str(key): value for key, value in raw.items()}
            model_name = str(row["model_name"])
            if model_name != "logistic_regression":
                continue
            rows.append(
                _normalise_row(
                    row,
                    model_id=profile_id,
                    label=label,
                    sensor_profile=profile_id,
                    model_name=model_name,
                )
            )

    hierarchical = _read_metric_file(
        root
        / "artifacts"
        / "registry"
        / "mvp3-oracle-v1"
        / "stage-model"
        / "validation_metrics.parquet"
    )
    for raw in hierarchical.to_dict(orient="records"):
        hierarchical_row: dict[str, object] = {
            str(key): value for key, value in raw.items()
        }
        model_name = str(hierarchical_row["model_name"])
        if model_name not in {"logistic_regression", "hist_gradient_boosting"}:
            continue
        model_id = (
            "hierarchical_logistic"
            if model_name == "logistic_regression"
            else "hierarchical_hgb"
        )
        label = (
            "전체 계층형 · Logistic"
            if model_name == "logistic_regression"
            else "전체 계층형 · HGB"
        )
        rows.append(
            _normalise_row(
                hierarchical_row,
                model_id=model_id,
                label=label,
                sensor_profile="watch_polar_muse",
                model_name=model_name,
            )
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("no comparison metrics were found")
    required = {"model_id", "head", "role", "data_status", "locked_test_read"}
    if not required.issubset(result.columns):
        raise ValueError("comparison output is missing required contract columns")
    if set(result["role"]) != {"validation"} or set(result["data_status"]) != {"oracle/sanity"}:
        raise ValueError("comparison is restricted to validation oracle/sanity metrics")
    if result["locked_test_read"].any():
        raise ValueError("locked test metrics cannot be included in comparison report")
    return result.sort_values(["head", "model_id"], kind="stable").reset_index(drop=True)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def build_comparison_artifact(frame: pd.DataFrame, *, generated_at: str) -> dict[str, object]:
    """Return the portable report contract plus the reviewed comparison rows."""

    records = [_json_safe(record) for record in frame.to_dict(orient="records")]
    charts = [
        {
            "id": str(chart["id"]),
            "title": str(chart["title"]),
            "subtitle": str(chart["subtitle"]),
            "type": "bar",
            "dataset": "model_comparison",
            "metric": str(chart["metric"]),
            "head": str(chart["head"]),
            "unit": str(chart["unit"]),
        }
        for chart in _CHARTS
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "센서 조합·계층형 모델 비교 보고서",
            "description": "현재 검증된 synthetic oracle/sanity validation 결과만 비교한 보고서.",
            "language": "ko",
            "generatedAt": generated_at,
            "data_status": "oracle/sanity",
            "real_data_status": "NOT VERIFIED",
            "locked_test_read": False,
            "font_family": "NanumGothic",
            "charts": charts,
            "sources": [
                "artifacts/availability-model/oracle-sanity-v3",
                "artifacts/registry/mvp3-oracle-v1/stage-model/validation_metrics.parquet",
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {"model_comparison": records},
        },
    }


def _bar_svg(
    rows: list[dict[str, object]],
    *,
    metric: str,
    title: str,
    unit: str,
    font_family: str,
    grouped_metrics: tuple[str, ...] = (),
) -> str:
    usable = [row for row in rows if _as_float(row.get(metric)) is not None]
    if grouped_metrics:
        usable = [
            row
            for row in rows
            if any(_as_float(row.get(item)) is not None for item in grouped_metrics)
        ]
    usable = usable[:20]
    label_width = 220
    chart_width = 760
    row_height = 42 if not grouped_metrics else 56
    chart_height = max(110, row_height * len(usable) + 70)
    values: list[float] = []
    for row in usable:
        if grouped_metrics:
            values.extend(
                value
                for value in (_as_float(row.get(item)) for item in grouped_metrics)
                if value is not None
            )
        else:
            value = _as_float(row.get(metric))
            if value is not None:
                values.append(value)
    maximum = max(values, default=1.0)
    if maximum <= 0:
        maximum = 1.0
    if metric != "false_alerts_per_hour" and not grouped_metrics:
        maximum = 1.0
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{html.escape(title)}" '
        f'viewBox="0 0 1040 {chart_height}" style="width:100%;height:auto">',
        (
            f'<style>text{{font-family:"{html.escape(font_family)}",sans-serif;fill:#172033}} '
            ".grid{stroke:#e5e7eb;stroke-width:1} .value{font-size:12px;font-weight:600} "
            ".label{font-size:12px} .axis{font-size:11px;fill:#5b6475}</style>"
        ),
        f'<text x="{label_width}" y="20" class="axis">0</text>',
        (
            f'<text x="{label_width + chart_width}" y="20" text-anchor="end" '
            f'class="axis">{maximum:.2f} {html.escape(unit)}</text>'
        ),
    ]
    for index, row in enumerate(usable):
        y = 34 + index * row_height
        label = html.escape(str(row.get("model_label_ko", row.get("model_id", ""))))
        parts.append(f'<text x="8" y="{y + 15}" class="label">{label}</text>')
        parts.append(
            f'<line x1="{label_width}" y1="{y + 22}" '
            f'x2="{label_width + chart_width}" y2="{y + 22}" class="grid"/>'
        )
        metrics = grouped_metrics or (metric,)
        for group_index, group_metric in enumerate(metrics):
            value = _as_float(row.get(group_metric))
            if value is None:
                continue
            bar_y = y + group_index * 18
            width = max(1.0, chart_width * value / maximum)
            color = "#20639b" if group_index == 0 else "#f28e2b"
            parts.append(
                f'<rect x="{label_width}" y="{bar_y}" width="{width:.2f}" '
                f'height="13" rx="3" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{label_width + width + 6:.2f}" y="{bar_y + 11}" '
                f'class="value">{value:.3f}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def render_comparison_html(
    artifact: dict[str, object],
    output: Path,
    *,
    font_family: str = "NanumGothic",
) -> None:
    """Render a self-contained static HTML reader and its canonical artifact JSON."""

    manifest = cast(dict[str, object], artifact["manifest"])
    snapshot = cast(dict[str, object], artifact["snapshot"])
    datasets = cast(dict[str, object], snapshot["datasets"])
    raw_records = cast(list[object], datasets["model_comparison"])
    records = [cast(dict[str, object], row) for row in raw_records]
    cards: list[str] = []
    for chart in _CHARTS:
        head = str(chart["head"])
        scoped = [row for row in records if str(row.get("head")) == head]
        metric = str(chart["metric"])
        grouped: tuple[str, ...] = ()
        if metric == "calibration":
            grouped = ("brier_score", "calibration_error")
            scoped = [row for row in scoped if row.get("model_id")]
            metric = "brier_score"
        svg = _bar_svg(
            scoped,
            metric=metric,
            title=str(chart["title"]),
            unit=str(chart["unit"]),
            font_family=font_family,
            grouped_metrics=grouped,
        )
        cards.append(
            f'<section class="card"><h2>{html.escape(str(chart["title"]))}</h2>'
            f'<p class="subtitle">{html.escape(str(chart["subtitle"]))}</p>{svg}</section>'
        )
    table_headers = (
        "모델",
        "Head",
        "AUCPR",
        "Event Recall",
        "Event F1",
        "False alerts/hour",
        "Brier",
        "ECE",
        "Stage macro-F1",
    )
    def display_value(value: object, digits: int) -> str:
        number = _as_float(value)
        return "-" if number is None else f"{number:.{digits}f}"

    table_rows: list[str] = []
    for row in records:
        table_rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(value)}</td>"
                for value in (
                    str(row.get("model_label_ko", "")),
                    str(row.get("head", "")),
                    display_value(row.get("aucpr"), 3),
                    display_value(row.get("event_recall"), 3),
                    display_value(row.get("event_f1"), 3),
                    display_value(row.get("false_alerts_per_hour"), 4),
                    display_value(row.get("brier_score"), 3),
                    display_value(row.get("calibration_error"), 3),
                    display_value(row.get("macro_f1"), 3),
                )
            )
            + "</tr>"
        )
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in table_headers)
    output.parent.mkdir(parents=True, exist_ok=True)
    page = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>센서 조합·계층형 모델 비교</title>
<style>
:root{{font-family:"{html.escape(font_family)}", "Apple SD Gothic Neo", sans-serif;
color:#172033;background:#f7f8fb}}
body{{max-width:1180px;margin:0 auto;padding:32px 22px 64px}}
header{{background:#172033;color:white;border-radius:18px;padding:26px 30px;margin-bottom:20px}}
h1{{margin:0 0 8px;font-size:28px}}
h2{{font-size:18px;margin:0 0 6px}}
.muted,.subtitle{{color:#667085;font-size:13px}}
header .muted{{color:#c7d0df}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}}
.card{{background:white;border:1px solid #e5e7eb;border-radius:16px;padding:18px;overflow:hidden}}
.notice{{background:#fff8e6;border:1px solid #e7b84b;border-radius:12px;
padding:12px 15px;margin:16px 0;color:#5b4500}}
.table-wrap{{overflow:auto;background:white;border:1px solid #e5e7eb;
border-radius:16px;margin-top:18px}}
table{{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}}
th,td{{padding:9px 10px;border-bottom:1px solid #eef0f3;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
th{{background:#f4f6f8;color:#475467}}
code{{font-family:ui-monospace,monospace}}
@media(max-width:700px){{body{{padding:18px 12px}}
.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header>
<h1>센서 조합·계층형 모델 비교</h1>
<p class="muted">검증 기준: validation · synthetic <code>oracle/sanity</code> ·
locked test 미사용 · 생성시각 {html.escape(str(manifest.get('generatedAt', '')))}</p>
</header>
<div class="notice">실제 Neon 데이터 정확도는 <strong>NOT VERIFIED</strong>입니다.
아래 비교는 합성 데이터에서 구조와 센서 가용성 차이를 확인하는 용도이며,
의료·행동 판단 성능으로 해석하지 않습니다. 보정 그래프의 낮은 값은 더 좋습니다.</div>
<div class="grid">{"".join(cards)}</div>
<div class="table-wrap"><table><thead><tr>{header_html}</tr></thead>
<tbody>{"".join(table_rows)}</tbody></table></div>
<p class="muted">폰트: {html.escape(font_family)}. Nanum 폰트가 설치되지 않은 환경에서는
Apple SD Gothic Neo로 대체됩니다.</p>
</body>
</html>"""
    output.write_text(page, encoding="utf-8")
    output.with_suffix(".artifact.json").write_text(
        json.dumps(_json_safe(artifact), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
