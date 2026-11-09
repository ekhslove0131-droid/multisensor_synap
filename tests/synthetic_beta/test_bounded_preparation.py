from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def module():
    name = "multisensor_ml.beta_training.bounded_preparation"
    assert importlib.util.find_spec(name) is not None, "TRAIN/VAL bounded preparation is missing"
    return importlib.import_module(name)


def test_plan_is_disjoint_deterministic_and_not_shared_event_clock() -> None:
    plan = module().create_recipe(seed=20261012)
    assert plan == module().create_recipe(seed=20261012)
    assert len(plan["subjects"]) == 30
    assert [s["role"] for s in plan["subjects"]] == ["TRAIN"] * 24 + ["VALIDATION"] * 6
    assert len({s["subject_uuid"] for s in plan["subjects"]}) == 30
    schedules = {tuple(map(tuple, s["events_ms"])) for s in plan["subjects"]}
    assert len(schedules) > 20
    for subject in plan["subjects"]:
        events = subject["events_ms"]
        assert len(events) == 2
        assert all(b - a == 180000 and 3220000 <= a <= 4540000 for a, b in events)
        assert events[1][0] - events[0][0] >= 480000
    assert any(s["response_kind"] == "NO_RESPONSE" for s in plan["subjects"])


def test_recipe_closes_over_feature_derivation_code_hashes() -> None:
    hashes = module().code_hashes()
    expected = {
        "codec.py", "features.py", "feature_catalog.py", "feature_types.py", "types.py",
    }
    runtime_root = Path(module().__file__).parents[3] / "beta_runtime/src/synapbridge_beta_runtime"
    hashed_names = {Path(path).name for path in hashes}
    assert expected <= hashed_names
    for name in expected:
        path = runtime_root / name
        assert str(path) in hashes


def test_locked_subject_and_modified_recipe_are_rejected() -> None:
    plan = module().create_recipe(seed=20261012)
    plan["subjects"][0]["role"] = "LOCKED_TEST"
    with pytest.raises(ValueError, match="recipe"):
        module().validate_recipe(plan)


def test_budget_has_fail_closed_wall_scratch_and_rss(tmp_path: Path) -> None:
    budget = module().Budget(tmp_path, wall_seconds=1, scratch_bytes=8, rss_bytes=100)
    with pytest.raises(RuntimeError, match="WALL"):
        budget.check(elapsed=2, rss=0)
    (tmp_path / "data").write_bytes(b"123456789")
    with pytest.raises(RuntimeError, match="SCRATCH"):
        budget.check(elapsed=0, rss=0)
    with pytest.raises(RuntimeError, match="RSS"):
        budget.check(elapsed=0, rss=101)


def test_regeneration_matches_native_hash_without_saved_raw(tmp_path: Path) -> None:
    mod = module()
    plan = mod.create_recipe(seed=20261012)
    mod.write_json(tmp_path / "recipe.json", plan)
    subject = plan["subjects"][0]
    row, part = mod.regenerate_part(plan, subject["subject_uuid"], "watch", 1300000)
    again, _ = mod.regenerate_part(plan, subject["subject_uuid"], "watch", 1300000)
    assert row == again
    assert part.scope.run_uuid == plan["run_uuid"]
    assert not list(tmp_path.rglob("*.npz"))
    bad = dict(row, content_sha256="0" * 64)
    with pytest.raises(ValueError, match="native_hash"):
        mod.regenerate_part(plan, subject["subject_uuid"], "watch", 1300000, expected=bad)


def test_train_val_prepared_contract_has_no_fake_locked(tmp_path: Path) -> None:
    mod = module()
    plan = mod.create_recipe(seed=20261012)
    # Test the boundary with complete recipe and minimal real feature schema files;
    # waveform derivation itself is covered by native regeneration/owner tests.
    refs = [{"role": role, "uri": "file:///fixture", "sha256": "a" * 64,
             "row_count": count} for role, count in (("TRAIN", 48), ("VALIDATION", 12))]
    manifest = mod.prepared_manifest(plan, tmp_path, refs, [], [], "b" * 64)
    assert set(r["role"] for r in manifest["label_refs"]) == {"TRAIN", "VALIDATION"}
    assert manifest["config"]["test_subjects"] == 0
    assert manifest["config"]["subject_count"] == 30
    assert manifest["run_epoch"] == 2
    assert manifest["source_manifest_sha256"] == "b" * 64
    from multisensor_ml.beta_training.cohort import _manifest_contract

    _manifest_contract(json.loads(json.dumps(manifest)))
