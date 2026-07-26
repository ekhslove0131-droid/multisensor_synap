from __future__ import annotations

from multisensor_synth.domain.seeds import SeedTree


def test_namespace_seed_is_stable() -> None:
    first = SeedTree(20260725).child_seed("person/P001/baseline")
    second = SeedTree(20260725).child_seed("person/P001/baseline")

    assert first == second


def test_namespace_seed_is_independent_of_call_order() -> None:
    first_tree = SeedTree(20260725)
    expected = first_tree.child_seed("person/P001/day/0")
    first_tree.child_seed("unrelated/new/namespace")

    second_tree = SeedTree(20260725)
    second_tree.child_seed("unrelated/new/namespace")
    actual = second_tree.child_seed("person/P001/day/0")

    assert actual == expected


def test_root_or_namespace_changes_seed() -> None:
    base = SeedTree(20260725).child_seed("population")

    assert SeedTree(20260726).child_seed("population") != base
    assert SeedTree(20260725).child_seed("split") != base


def test_used_namespace_map_is_reconstructable() -> None:
    tree = SeedTree(7)
    seed = tree.child_seed("person/P001/event/E001")

    assert tree.used_namespaces == {"person/P001/event/E001": seed}
