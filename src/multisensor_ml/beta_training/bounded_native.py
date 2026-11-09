"""Recipe-bound native regeneration; no nonexistent native URI is advertised."""
from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from synapbridge_beta_runtime.codec import canonical_bytes, decode_native
from synapbridge_beta_runtime.types import BetaScope

from .artifacts import sha256


class NativeRegenerator:
    def __init__(self, recipe: dict[str, Any]) -> None:
        from .bounded_preparation import validate_recipe

        validate_recipe(recipe)
        self.recipe = json.loads(canonical_bytes(recipe))
        self.subjects = {s["subject_uuid"]: s for s in self.recipe["subjects"]}

    def part(self, subject: str, family: str, start: int,
             *, expected: dict[str, Any] | None = None) -> tuple[dict[str, Any], Any]:
        from multisensor_synth.beta_observed.codec import encode_part
        from multisensor_synth.beta_observed.config import BetaSignalState
        from multisensor_synth.beta_observed.generator import generate_part

        value = self.subjects.get(subject)
        if value is None or value["role"] not in ("TRAIN", "VALIDATION"):
            raise ValueError("recipe_subject_or_locked_invalid")
        if family not in ("watch", "h10", "muse") or not 1000000 <= start < 4900000:
            raise ValueError("recipe_part_range_invalid")
        state = BetaSignalState(self.recipe["run_uuid"], self.recipe["epoch"], subject,
                                MappingProxyType(value["source_uuids"]), value["waveform_seed"],
                                1000000, value["order"], value["role"], value["baseline_hr"],
                                value["response_kind"], tuple(map(tuple, value["events_ms"])))
        manifest, content = encode_part(generate_part(state, family, start))
        row = {"subject_uuid": subject,
               "source_uuid": value["source_uuids"][family],
               "family": family, "start_ms": start, "end_ms": start + 10000,
               "manifest_sha256": sha256(canonical_bytes(manifest)),
               "content_sha256": sha256(content),
               "compressed_bytes": len(content)}
        if expected is not None and row != expected:
            raise ValueError("native_hash_mismatch")
        return row, decode_native(manifest, content)


class ObservedRecipeReader:
    def __init__(self, root: Path, source_set: tuple[str, ...]) -> None:
        observed = json.loads((root / "observed_manifest.json").read_bytes())
        recipe_bytes = (root / "recipe.json").read_bytes()
        index_bytes = (root / "native-index.jsonl").read_bytes()
        if (sha256(recipe_bytes) != observed["recipe_ref"]["sha256"]
                or sha256(index_bytes) != observed["native_index_ref"]["sha256"]):
            raise ValueError("observed_recipe_ref_invalid")
        self.generator = NativeRegenerator(json.loads(recipe_bytes))
        self.source_set = source_set
        self.index = {(v["subject_uuid"], v["family"], v["start_ms"]): v
                      for v in map(json.loads, index_bytes.splitlines())}

    def parts(self, scope: BetaScope, start_ms: int, end_ms: int) -> tuple[Any, ...]:
        recipe = self.generator.recipe
        if scope.run_uuid != recipe["run_uuid"] or scope.epoch != recipe["epoch"]:
            raise ValueError("native_recipe_scope_invalid")
        parts = []
        for start in range(start_ms, end_ms, 10000):
            for family in self.source_set:
                expected = self.index.get((scope.subject_uuid, family, start))
                if expected is None:
                    raise ValueError("native_recipe_index_missing")
                _, part = self.generator.part(scope.subject_uuid, family, start, expected=expected)
                parts.append(part)
        return tuple(parts)
