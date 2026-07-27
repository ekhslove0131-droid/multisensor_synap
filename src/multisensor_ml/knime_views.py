from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from multisensor_ml.contracts import ORACLE_LATENT_FACTORS


def _manifest(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _factory_view(project: Path, series: str, view: str) -> pd.DataFrame:
    outcome_root = project / "data" / "outcomes" / series
    outcome = _manifest(outcome_root / "manifest.json")
    registry_root = project / "data" / "registry" / series
    registry = _manifest(registry_root / "manifest.json")
    if view == "status":
        return pd.DataFrame(
            [
                {"항목": "실행 상태", "값": outcome["status"]},
                {"항목": "실제 데이터 성능", "값": outcome["real_data_status"]},
                {"항목": "장비 동기화", "값": outcome["synchronization_status"]},
                {"항목": "사건 수", "값": outcome["event_count"]},
                {"항목": "행동 양성 라벨 수", "값": outcome["behavior_positive_count"]},
            ]
        )
    if view == "participants":
        splits = pq.read_table(registry_root / "splits.parquet").to_pandas()
        return cast(
            pd.DataFrame,
            splits.groupby(["run_id", "split_role"], as_index=False)
            .agg(참여자수=("person_id", "nunique"))
            .rename(columns={"run_id": "생성실행", "split_role": "데이터역할"}),
        )
    if view == "baseline":
        rows: list[pd.DataFrame] = []
        for run in cast(list[dict[str, object]], registry["runs"]):
            truth = Path(str(run["source_path"])) / "truth" / "latent_timeline.parquet"
            frame = pq.read_table(
                truth,
                columns=["run_id", "person_id", *ORACLE_LATENT_FACTORS],
            ).to_pandas()
            summary = frame.groupby(["run_id", "person_id"], as_index=False)[
                list(ORACLE_LATENT_FACTORS)
            ].median()
            rows.append(summary)
        result = pd.concat(rows, ignore_index=True)
        return result.rename(
            columns={
                "run_id": "생성실행",
                "person_id": "참여자",
                **{factor: f"생성QA_{factor}" for factor in ORACLE_LATENT_FACTORS},
            }
        )
    if view == "timeline":
        stages = pq.read_table(outcome_root / "outcome_stages.parquet").to_pandas()
        event_rows = stages.loc[stages["stage_code"] != "NO_EVENT"]
        baseline = stages.loc[stages["stage_code"] == "NO_EVENT"].iloc[::600]
        return cast(
            pd.DataFrame,
            pd.concat([event_rows.iloc[::10], baseline], ignore_index=True)
            .sort_values(["run_id", "person_id", "timestamp_utc"])
            .head(20_000)
            .rename(
                columns={
                    "run_id": "생성실행",
                    "person_id": "참여자",
                    "timestamp_utc": "시각",
                    "event_id": "사건ID",
                    "stage_code": "5단계라벨",
                }
            ),
        )
    if view == "behaviors":
        behaviors = pq.read_table(outcome_root / "outcome_behaviors.parquet").to_pandas()
        return cast(
            pd.DataFrame,
            behaviors.groupby(["run_id", "person_id", "event_id"], as_index=False)
            .agg(
                행동조합=("behavior_code", lambda values: " | ".join(sorted(values))),
                행동수=("behavior_code", "size"),
                라벨신뢰도=("label_confidence", "mean"),
            )
            .rename(
                columns={
                    "run_id": "생성실행",
                    "person_id": "참여자",
                    "event_id": "사건ID",
                }
            ),
        )
    if view == "target_compare":
        events = pq.read_table(outcome_root / "outcome_events.parquet").to_pandas()
        behaviors = pq.read_table(outcome_root / "outcome_behaviors.parquet").to_pandas()
        counts = (
            behaviors.groupby(["run_id", "person_id", "event_id"])
            .size()
            .rename("행동라벨수")
            .reset_index()
        )
        return cast(
            pd.DataFrame,
            events.merge(
                counts,
                on=["run_id", "person_id", "event_id"],
                how="left",
                validate="one_to_one",
            )
            .assign(
                비교구분=lambda frame: np.where(frame["is_target"], "타깃 사건", "하드 네거티브")
            )[
                [
                    "run_id",
                    "person_id",
                    "event_id",
                    "비교구분",
                    "event_type",
                    "hard_negative_kind",
                    "행동라벨수",
                ]
            ]
            .fillna({"행동라벨수": 0})
            .rename(
                columns={
                    "run_id": "생성실행",
                    "person_id": "참여자",
                    "event_id": "사건ID",
                    "event_type": "사건유형",
                    "hard_negative_kind": "유사비사건유형",
                }
            ),
        )
    if view == "split":
        return cast(
            pd.DataFrame,
            pq.read_table(registry_root / "splits.parquet")
            .to_pandas()
            .rename(
                columns={
                    "run_id": "생성실행",
                    "person_id": "참여자",
                    "split_role": "데이터역할",
                }
            ),
        )
    if view == "devices":
        return pd.DataFrame(
            [
                {
                    "장비": device,
                    "현재 구현": "다음 Goal 인터페이스",
                    "동기화 상태": "NOT_AVAILABLE_TRUTH_ONLY",
                    "기준 장비": "Polar H10",
                }
                for device in ("Muse", "Polar H10", "Galaxy Watch")
            ]
        )
    raise ValueError(f"unsupported factory view: {view}")


def _registry_view(project: Path, series: str, view: str) -> pd.DataFrame:
    root = project / "artifacts" / "registry" / series
    if view == "dataset":
        splits = pq.read_table(
            project / "data" / "registry" / series / "splits.parquet"
        ).to_pandas()
        return cast(
            pd.DataFrame,
            splits.groupby("split_role", as_index=False)
            .agg(참여자수=("person_id", "nunique"))
            .rename(columns={"split_role": "데이터역할"}),
        )
    if view == "baseline":
        memberships = pq.read_table(root / "types" / "person_standard_types.parquet").to_pandas()
        ood = pq.read_table(root / "types" / "ood_status.parquet").to_pandas()
        return cast(
            pd.DataFrame,
            memberships.merge(
                ood[["person_key", "distance", "status"]],
                on="person_key",
                validate="one_to_one",
            ).rename(
                columns={
                    "person_key": "참여자",
                    "split_role": "데이터역할",
                    "distance": "분포거리",
                    "status": "분포상태",
                }
            ),
        )
    if view == "performance":
        stage = pq.read_table(root / "stage-model" / "validation_metrics.parquet").to_pandas()
        behavior = pq.read_table(root / "behavior-model" / "validation_metrics.parquet").to_pandas()
        return cast(
            pd.DataFrame,
            pd.concat([stage, behavior], ignore_index=True, sort=False).rename(
                columns={
                    "head": "모델헤드",
                    "model_name": "후보모델",
                    "behavior_code": "행동코드",
                    "status": "승격상태",
                }
            ),
        )
    if view == "stress":
        return cast(
            pd.DataFrame,
            pq.read_table(root / "evaluate" / "stress_metrics.parquet")
            .to_pandas()
            .rename(
                columns={
                    "scenario": "노이즈유형",
                    "magnitude": "노이즈크기",
                    "aucpr_degradation": "AUCPR감소율",
                }
            ),
        )
    if view == "registry":
        database = project / "data" / "model_registry" / "goal15.sqlite"
        with sqlite3.connect(database) as connection:
            releases = pd.read_sql_query(
                """
                SELECT release_id, status, source_domain, created_at
                FROM releases ORDER BY created_at DESC
                """,
                connection,
            )
        return releases.rename(
            columns={
                "release_id": "릴리스",
                "status": "상태",
                "source_domain": "데이터영역",
                "created_at": "생성시각",
            }
        )
    if view == "korean":
        return cast(
            pd.DataFrame,
            pq.read_table(root / "route-ko" / "korean_results.parquet").to_pandas(),
        )
    raise ValueError(f"unsupported registry view: {view}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--domain", choices=("factory", "registry"), required=True)
    parser.add_argument("--series", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project = args.project_root.resolve()
    frame = (
        _factory_view(project, args.series, args.view)
        if args.domain == "factory"
        else _registry_view(project, args.series, args.view)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
