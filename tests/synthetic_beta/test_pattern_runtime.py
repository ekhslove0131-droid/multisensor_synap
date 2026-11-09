from __future__ import annotations

from types import MappingProxyType

import pytest
from synapbridge_beta_runtime.feature_catalog import feature_order, standard_feature_order
from synapbridge_beta_runtime.feature_types import FeatureFrame
from synapbridge_beta_runtime.model_types import SourceReference
from synapbridge_beta_runtime.pattern_runtime import assemble_model_frame
from synapbridge_beta_runtime.standard import score_reference
from synapbridge_beta_runtime.types import BetaScope


def _frame(family: str, decision_ms: int, value: float) -> FeatureFrame:
    values = {name: value + index for index, name in enumerate(feature_order(family))}
    return FeatureFrame(
        BetaScope(
            "10000000-0000-4000-8000-000000000001",
            1,
            "20000000-0000-4000-8000-000000000001",
        ),
        decision_ms,
        family,
        MappingProxyType(values),
        (decision_ms - 10_000, decision_ms),
        1.0,
        family[0] * 64,
    )


def _reference(family: str) -> SourceReference:
    names = standard_feature_order(family)[:2]
    return SourceReference(
        family,
        MappingProxyType({name: 1.0 for name in names}),
        MappingProxyType({name: 2.0 for name in names}),
        names,
        family[-1] * 64,
        1800,
    )


def test_exact_source_set_is_assembled_without_padding() -> None:
    history = [_frame("watch", 120_000, 2.0), _frame("watch", 180_000, 5.0)]
    model = assemble_model_frame(history, {"watch": _reference("watch")}, ("watch",))
    assert model is not None
    assert model.source_set == ("watch",)
    assert model.values.shape == (len(feature_order("watch")) + 5,)
    assert model.feature_names[-3:] == (
        "watch.z_rms",
        "watch.z_rms_delta_60",
        "watch.quality",
    )
    prior_deviation = score_reference(history[0], _reference("watch")).z_rms
    assert prior_deviation is not None
    assert model.values[-2] == pytest.approx(
        (model.current_deviation - prior_deviation) / 60
    )
    assert assemble_model_frame(history, {"watch": _reference("watch")}, ("watch", "h10")) is None


def test_source_set_order_is_watch_h10_muse() -> None:
    history = [
        _frame(family, decision, 2.0 + index)
        for decision in (120_000, 180_000)
        for index, family in enumerate(("watch", "h10", "muse"))
    ]
    references = {family: _reference(family) for family in ("watch", "h10", "muse")}
    model = assemble_model_frame(history, references, ("muse", "watch", "h10"))
    assert model is not None
    assert model.source_set == ("watch", "h10", "muse")
    assert model.feature_names[0].startswith("watch.")


def test_partial_feature_catalog_is_not_repacked_into_a_new_schema() -> None:
    history = [_frame("watch", 120_000, 2.0), _frame("watch", 180_000, 5.0)]
    current = history[-1]
    values = dict(current.values)
    values.pop(next(iter(values)))
    history[-1] = FeatureFrame(
        current.scope, current.decision_ms, current.family, MappingProxyType(values),
        current.valid_support_ms, current.quality, current.schema_sha,
    )
    assert assemble_model_frame(
        history, {"watch": _reference("watch")}, ("watch",)
    ) is None


def test_model_frame_rejects_history_mixed_across_subject_scopes() -> None:
    history = [_frame("watch", 120_000, 2.0), _frame("watch", 180_000, 5.0)]
    current = history[-1]
    history[-1] = FeatureFrame(
        BetaScope(current.scope.run_uuid, current.scope.epoch,
                  "20000000-0000-4000-8000-000000000099"),
        current.decision_ms, current.family, current.values,
        current.valid_support_ms, current.quality, current.schema_sha,
    )
    with pytest.raises(ValueError, match="model_frame_scope_mismatch"):
        assemble_model_frame(history, {"watch": _reference("watch")}, ("watch",))
