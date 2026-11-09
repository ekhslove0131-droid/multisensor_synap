"""TRAIN24/VAL6-only preparation; original full-cohort path is unchanged."""
from __future__ import annotations

import hashlib
import random
import resource
import time
from collections import deque
from pathlib import Path
from typing import Any
from uuid import UUID

from synapbridge_beta_runtime.codec import canonical_bytes
from synapbridge_beta_runtime.features import derive_features

from .artifacts import sha256, write_create_only
from .dataset import _write_feature_schemas

SYNTH = Path("/Users/baital/dev/multisensor_synth")
BASE_PROTOCOL = SYNTH / "output/beta-loop-v1/protocol.json"


def write_json(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    write_create_only(path, payload)
    return sha256(payload)


def _uuid(*values: object) -> str:
    digest = hashlib.sha256(canonical_bytes(values)).digest()
    return str(UUID(bytes=digest[:16], version=4))


def code_hashes() -> dict[str, str]:
    paths = [Path(__file__), Path(__file__).with_name("bounded_native.py")]
    runtime_root = Path(__file__).resolve().parents[3] / "beta_runtime/src/synapbridge_beta_runtime"
    paths += [runtime_root / name for name in (
        "codec.py", "features.py", "feature_catalog.py", "feature_types.py", "types.py",
    )]
    paths += [SYNTH / "src/multisensor_synth/beta_observed" / name for name in
              ("cardiac.py", "eda.py", "eeg.py", "generator.py", "codec.py", "config.py")]
    return {str(path): sha256(path.read_bytes()) for path in paths}


def create_recipe(*, seed: int) -> dict[str, Any]:
    if type(seed) is not int or seed < 0:
        raise ValueError("recipe_seed_invalid")
    subjects = []
    run_uuid = _uuid("schedule_independent_A_v1", seed)
    candidates = [(a, b) for a in range(3220000, 4060001, 10000)
                  for b in range(a + 480000, 4540001, 10000)]
    for index in range(30):
        role = "TRAIN" if index < 24 else "VALIDATION"
        subject = _uuid(seed, role, index, "subject")
        event_seed = int(sha256(canonical_bytes([seed, role, index, "event_schedule"])), 16)
        a, b = random.Random(event_seed).choice(candidates)
        noise_seed = int(sha256(canonical_bytes([seed, role, index, "waveform_noise"]))[:8], 16)
        response_kind = ("RESPONSIVE", "WEAK", "INVERSE", "NO_RESPONSE")[index % 4]
        subjects.append({"subject_uuid": subject, "role": role, "order": index,
                         "waveform_seed": noise_seed,
                         "source_uuids": {f: _uuid(run_uuid, subject, f) for f in
                                          ("watch", "h10", "muse")},
                         "baseline_hr": 66.0 + index % 9,
                         "response_kind": response_kind,
                         "events_ms": [[a, a + 180000], [b, b + 180000]]})
    core = {"schema_version": "synthetic_beta_trainval_recipe_v1", "realm": "SYNTHETIC_BETA",
            "seed": seed, "run_uuid": run_uuid, "epoch": 2, "timeline_origin_ms": 1000000,
            "config": {"seed": seed, "subject_count": 30, "duration_seconds": 3900,
                       "prefill_seconds": 300, "calibration_seconds": 1800,
                       "scored_seconds": 1800, "train_subjects": 24,
                       "validation_subjects": 6, "test_subjects": 0, "events_per_subject": 2},
            "subjects": subjects, "code_hashes": code_hashes()}
    protocol = {"schema_version": "synthetic_beta_schedule_independent_protocol_v1",
                "base_protocol_ref": {"uri": BASE_PROTOCOL.as_uri(),
                                      "sha256": sha256(BASE_PROTOCOL.read_bytes())},
                "changed_variable": "event_schedule_independent_of_background",
                "schedule_method": "role_subject_namespace_uniform_eligible_pair_v1",
                "event_duration_ms": 180000, "minimum_onset_separation_ms": 480000,
                "background_and_numeric_generator_changed": False,
                "no_response_remains_positive": True,
                "recipe_core_sha256": sha256(canonical_bytes(core))}
    body = {**core, "protocol": protocol}
    return {**body, "content_sha256": sha256(canonical_bytes(body))}


def validate_recipe(recipe: dict[str, Any]) -> None:
    if (recipe != create_recipe(seed=recipe.get("seed", -1))
            or any(s["role"] not in ("TRAIN", "VALIDATION") for s in recipe["subjects"])):
        raise ValueError("recipe_integrity_or_role_invalid")


class Budget:
    def __init__(self, root: Path, *, wall_seconds: float = 900,
                 scratch_bytes: int = 256 * 2**20, rss_bytes: int = 2**30) -> None:
        self.root, self.started = root, time.monotonic()
        self.wall_seconds, self.scratch_bytes, self.rss_bytes = (
            wall_seconds, scratch_bytes, rss_bytes)
        self.peak_scratch_bytes = 0

    def check(self, *, elapsed: float | None = None, rss: int | None = None) -> None:
        wall = time.monotonic() - self.started if elapsed is None else elapsed
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if rss is None else rss
        if wall > self.wall_seconds:
            raise RuntimeError("BUDGET_WALL_EXCEEDED")
        if peak > self.rss_bytes:
            raise RuntimeError("BUDGET_RSS_EXCEEDED")
        scratch = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        self.peak_scratch_bytes = max(self.peak_scratch_bytes, scratch)
        if scratch > self.scratch_bytes:
            raise RuntimeError("BUDGET_SCRATCH_EXCEEDED")


def prepared_manifest(recipe: dict[str, Any], output: Path, label_refs: list[dict[str, Any]],
                      feature_refs: list[dict[str, Any]], schema_refs: list[dict[str, Any]],
                      source_sha: str) -> dict[str, Any]:
    body = {"schema_version": "synthetic_beta_prepared_dataset_v1", "realm": "SYNTHETIC_BETA",
            "beta_run_uuid": recipe["run_uuid"], "run_epoch": recipe["epoch"],
            "config": recipe["config"], "timeline_origin_ms": recipe["timeline_origin_ms"],
            "source_manifest_sha256": source_sha,
            "protocol_ref": {"uri": (output / "protocol.json").resolve().as_uri(),
                             "sha256": sha256(canonical_bytes(recipe["protocol"]))},
            "split_artifact": {"uri": (output / "split.json").resolve().as_uri(),
                               "sha256": sha256(canonical_bytes([
                                   {"subject_uuid": s["subject_uuid"], "split": s["role"]}
                                   for s in recipe["subjects"]]))},
            "label_refs": label_refs, "feature_schema_refs": schema_refs,
            "feature_refs": feature_refs}
    return {**body, "content_sha256": sha256(canonical_bytes(body))}


def _labels(subject: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"event_uuid": _uuid(subject["subject_uuid"], a),
             "subject_uuid": subject["subject_uuid"], "split": subject["role"],
             "onset_ms": a, "offset_ms": b, "pre_early_start_ms": a - 60000,
             "pre_late_start_ms": a - 30000, "recovery_early_start_ms": b,
             "recovery_late_start_ms": b + 60000, "phase_end_ms": b + 120000,
             "response_kind": subject["response_kind"]} for a, b in subject["events_ms"]]


def prepare_trainval(recipe: dict[str, Any], output: Path, budget: Budget) -> Path:
    from .bounded_native import NativeRegenerator

    validate_recipe(recipe)
    budget.check()
    if output.exists():
        raise FileExistsError("bounded_preparation_output_exists")
    output.mkdir(parents=True)
    recipe_sha = write_json(output / "recipe.json", recipe)
    protocol_sha = write_json(output / "protocol.json", recipe["protocol"])
    split = [{"subject_uuid": s["subject_uuid"], "split": s["role"]} for s in recipe["subjects"]]
    split_sha = write_json(output / "split.json", split)
    label_refs = []
    for role in ("TRAIN", "VALIDATION"):
        rows = [row for s in recipe["subjects"] if s["role"] == role for row in _labels(s)]
        path = output / "labels" / f"{role.lower()}.jsonl"
        write_create_only(path, b"".join(canonical_bytes(row) + b"\n" for row in rows))
        label_refs.append({"role": role, "uri": path.resolve().as_uri(),
                           "sha256": sha256(path.read_bytes()), "row_count": len(rows)})
    schemas = _write_feature_schemas(output)
    feature_refs = []
    native_path = output / "native-index.jsonl"
    regenerator = NativeRegenerator(recipe)
    with native_path.open("xb") as index_handle:
        for subject in recipe["subjects"]:
            histories: dict[str, Any] = {
                family: deque(maxlen=18) for family in ("watch", "h10", "muse")}
            handles, counts, paths = {}, {}, {}
            for family in histories:
                path = output / "features" / subject["subject_uuid"] / f"{family}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                handles[family], counts[family], paths[family] = path.open("xb"), 0, path
            try:
                for start in range(1000000, 4900000, 10000):
                    for family, history in histories.items():
                        row, part = regenerator.part(subject["subject_uuid"], family, start)
                        index_handle.write(canonical_bytes(row) + b"\n")
                        history.append(part)
                        frames = derive_features(tuple(history), part.end_ms)
                        for frame in frames:
                            handles[family].write(canonical_bytes({
                                "decision_ms": frame.decision_ms, "family": family,
                                "values": dict(frame.values),
                                "valid_support_ms": frame.valid_support_ms,
                                "quality": frame.quality, "schema_sha": frame.schema_sha}) + b"\n")
                            counts[family] += 1
                    if (start - 1000000) % 100000 == 0:
                        budget.check()
                print(f"PREPARED_SUBJECT {subject['order']} {subject['role']}", flush=True)
            finally:
                for handle in handles.values():
                    handle.close()
            for family, path in paths.items():
                feature_refs.append({"subject_uuid": subject["subject_uuid"],
                                     "split": subject["role"],
                                     "family": family, "uri": path.resolve().as_uri(),
                                     "sha256": sha256(path.read_bytes()),
                                     "row_count": counts[family]})
    observed = {"schema_version": "synthetic_beta_regenerable_observed_v1",
                "realm": "SYNTHETIC_BETA",
                "recipe_ref": {"uri": (output / "recipe.json").resolve().as_uri(),
                               "sha256": recipe_sha},
                "native_index_ref": {"uri": native_path.resolve().as_uri(),
                                     "sha256": sha256(native_path.read_bytes())},
                "raw_permanently_saved_bytes": 0, "regeneration_verified": True,
                "parts": len(recipe["subjects"]) * 390 * 3}
    observed_sha = write_json(output / "observed_manifest.json", observed)
    source = {"schema_version": "synthetic_beta_trainval_regenerable_source_v1",
              "realm": "SYNTHETIC_BETA",
              "run_uuid": recipe["run_uuid"], "epoch": recipe["epoch"],
              "recipe_ref": observed["recipe_ref"],
              "protocol_ref": {"uri": (output / "protocol.json").resolve().as_uri(),
                               "sha256": protocol_sha},
              "observed_manifest_ref": {
                  "uri": (output / "observed_manifest.json").resolve().as_uri(),
                  "sha256": observed_sha},
              "split_ref": {"uri": (output / "split.json").resolve().as_uri(), "sha256": split_sha},
              "locked_waveform_or_labels_generated": False}
    source_sha = write_json(output / "source_manifest.json", source)
    write_json(output / "manifest.json", prepared_manifest(recipe, output, label_refs,
                                                         feature_refs, schemas, source_sha))
    budget.check()
    return output / "manifest.json"


def regenerate_part(recipe: dict[str, Any], subject: str, family: str, start: int,
                    *, expected: dict[str, Any] | None = None) -> tuple[dict[str, Any], Any]:
    from .bounded_native import NativeRegenerator

    return NativeRegenerator(recipe).part(subject, family, start, expected=expected)
