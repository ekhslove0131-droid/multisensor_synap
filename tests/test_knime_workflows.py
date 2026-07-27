from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {"k": "http://www.knime.org/2008/09/XMLConfig"}
ROOT = Path(__file__).parents[1]


def _workflow_root(archive: zipfile.ZipFile) -> ET.Element:
    name = next(
        value
        for value in archive.namelist()
        if value.endswith("/workflow.knime") and value.count("/") == 1
    )
    return ET.fromstring(archive.read(name))


def _config(parent: ET.Element, key: str) -> ET.Element:
    return next(
        child
        for child in parent.findall("k:config", NS)
        if child.get("key") == key
    )


def _value(parent: ET.Element, key: str) -> str:
    entry = next(
        child
        for child in parent.findall("k:entry", NS)
        if child.get("key") == key
    )
    return str(entry.get("value"))


def _graph(path: Path) -> tuple[set[int], set[tuple[int, int]]]:
    with zipfile.ZipFile(path) as archive:
        root = _workflow_root(archive)
        nodes = {
            int(_value(node, "id"))
            for node in _config(root, "nodes").findall("k:config", NS)
        }
        edges = {
            (
                int(_value(connection, "sourceID")),
                int(_value(connection, "destID")),
            )
            for connection in _config(root, "connections").findall(
                "k:config",
                NS,
            )
        }
    return nodes, edges


def _assert_table_views_show_all_columns(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        table_views = [
            name
            for name in archive.namelist()
            if "보기 " in name and name.endswith("/settings.xml")
        ]
        assert table_views
        for name in table_views:
            root = ET.fromstring(archive.read(name))
            view = _config(root, "view")
            displayed = _config(view, "displayedColumnsV2")
            assert _value(displayed, "mode") == "PATTERN"
            assert _value(_config(displayed, "patternFilter"), "pattern") == "*"


def test_factory_workflow_is_one_connected_receipt_graph() -> None:
    path = ROOT / "knime" / "Multisensor_Synthetic_Factory_Goal1_5.knwf"
    nodes, edges = _graph(path)
    assert nodes == {1, 2, 3} | set(range(10, 26))
    assert len(edges) == 18
    assert {node for edge in edges for node in edge} == nodes
    assert (1, 2) in edges
    _assert_table_views_show_all_columns(path)


def test_training_workflow_has_serial_stage_receipt_chain() -> None:
    path = ROOT / "knime" / "Multisensor_ML_Training_Registry.knwf"
    nodes, edges = _graph(path)
    assert nodes == set(range(1, 11)) | set(range(20, 32))
    assert len(edges) == 21
    assert {node for edge in edges for node in edge} == nodes
    assert {(node, node + 1) for node in range(1, 9)} <= edges
    _assert_table_views_show_all_columns(path)


def test_oracle_benchmark_workflow_is_preserved() -> None:
    assert (ROOT / "knime" / "Multisensor_ML_Goal1_5.knwf").is_file()
