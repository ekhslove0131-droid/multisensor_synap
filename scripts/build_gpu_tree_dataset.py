#!/usr/bin/env python3
"""Build a deterministic, schema-current bounded Dataset for the GPU tree run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_rows(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    positive = frame["event_binary"].astype(bool).to_numpy()
    hard_negative = frame["hard_negative"].astype(bool).to_numpy()
    must_keep = np.flatnonzero(positive | hard_negative)
    if len(must_keep) > max_rows:
        rng = np.random.default_rng(seed)
        positive_indices = np.flatnonzero(positive)
        hard_indices = np.flatnonzero(hard_negative & ~positive)
        positive_quota = min(len(positive_indices), max(1, max_rows * 4 // 5))
        hard_quota = min(len(hard_indices), max_rows - positive_quota)
        chosen_positive = rng.choice(positive_indices, positive_quota, replace=False)
        chosen_hard = rng.choice(hard_indices, hard_quota, replace=False)
        selected = np.unique(np.concatenate([chosen_positive, chosen_hard]))
    else:
        selected = must_keep
    remaining = max_rows - len(selected)
    if remaining > 0:
        candidates = np.setdiff1d(
            np.arange(len(frame), dtype="int64"), selected, assume_unique=True
        )
        rng = np.random.default_rng(seed + 7919)
        selected = np.concatenate(
            [selected, rng.choice(candidates, min(remaining, len(candidates)), replace=False)]
        )
    return frame.iloc[np.sort(selected)].reset_index(drop=True)


def build_dataset(
    source_root: Path, output_root: Path, *, max_rows_per_person: int
) -> dict[str, Any]:
    source_manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    people_root = output_root / "people"
    people_root.mkdir(parents=True, exist_ok=True)
    people: list[dict[str, Any]] = []
    source_people = [
        entry
        for entry in source_manifest["people"]
        if entry["split_role"] in {"train", "validation"}
    ]
    for index, entry in enumerate(source_people):
        source_path = source_root / str(entry["path"])
        frame = pd.read_parquet(source_path)
        selected = _select_rows(frame, max_rows_per_person, 20260725 + index)
        output_path = people_root / source_path.name
        selected.to_parquet(output_path, index=False, compression="zstd")
        people.append(
            {
                **entry,
                "path": f"people/{output_path.name}",
                "row_count": len(selected),
                "logical_hash": _sha256(output_path),
            }
        )
    manifest: dict[str, Any] = {
        **source_manifest,
        "dataset_id": "mvp3-oracle-v1-gpu-bounded-current",
        "source_dataset_id": source_manifest.get("dataset_id"),
        "prepared_schema": "goal1.5/prepared/v1",
        "subset_policy": {
            "max_rows_per_person": max_rows_per_person,
            "selection": (
                "all positive and hard-negative rows when within cap; "
                "deterministic baseline fill"
            ),
            "seed": 20260725,
            "locked_test_included": False,
        },
        "split_counts": {
            role: sum(item["split_role"] == role for item in people)
            for role in ("train", "validation")
        },
        "locked_test_read": False,
        "people": people,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-rows-per-person", type=int, default=20_000)
    args = parser.parse_args()
    manifest = build_dataset(
        args.source_root.resolve(),
        args.output_root.resolve(),
        max_rows_per_person=args.max_rows_per_person,
    )
    print(
        json.dumps(
            {
                "status": "READY",
                "dataset_id": manifest["dataset_id"],
                "people": len(manifest["people"]),
                "rows": sum(int(item["row_count"]) for item in manifest["people"]),
                "split_counts": {
                    role: sum(item["split_role"] == role for item in manifest["people"])
                    for role in ("train", "validation", "locked_test")
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
