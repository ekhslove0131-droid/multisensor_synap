from __future__ import annotations

import copy
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = "http://www.knime.org/2008/09/XMLConfig"
ET.register_namespace("", NS)
TAG = f"{{{NS}}}"
PROJECT = Path("/Users/baital/dev/multisensor_ml")
TEMPLATE = PROJECT / "knime" / "Multisensor_ML_Goal1_5.knwf"


def _config(parent: ET.Element, key: str) -> ET.Element:
    for child in parent.findall(f"{TAG}config"):
        if child.get("key") == key:
            return child
    raise KeyError(key)


def _entry(parent: ET.Element, key: str) -> ET.Element:
    for child in parent.findall(f"{TAG}entry"):
        if child.get("key") == key:
            return child
    raise KeyError(key)


def _set_entry(parent: ET.Element, key: str, value: str) -> None:
    _entry(parent, key).set("value", value)


def _write_xml(tree: ET.ElementTree[ET.Element[str]], path: Path) -> None:
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def _clear_children(element: ET.Element[str]) -> None:
    for child in list(element):
        element.remove(child)


def _set_annotation(settings: ET.Element, text: str) -> None:
    try:
        annotation = _config(settings, "nodeAnnotation")
    except KeyError:
        annotation = ET.SubElement(settings, f"{TAG}config", key="nodeAnnotation")
        defaults = [
            ("text", "xstring", ""),
            ("contentType", "xstring", "text/plain"),
            ("bgcolor", "xint", "16777215"),
            ("x-coordinate", "xint", "0"),
            ("y-coordinate", "xint", "0"),
            ("width", "xint", "180"),
            ("height", "xint", "45"),
            ("alignment", "xstring", "CENTER"),
            ("borderSize", "xint", "0"),
            ("borderColor", "xint", "0"),
            ("defFontSize", "xint", "-1"),
            ("annotation-version", "xint", "20230412"),
        ]
        for key, kind, value in defaults:
            ET.SubElement(
                annotation,
                f"{TAG}entry",
                key=key,
                type=kind,
                value=value,
            )
        ET.SubElement(annotation, f"{TAG}config", key="styles")
    _set_entry(annotation, "text", text)
    _set_entry(annotation, "contentType", "text/plain")
    _set_entry(annotation, "width", "180")
    _set_entry(annotation, "height", "45")


def _table_creator(
    template: Path,
    destination: Path,
    *,
    columns: list[tuple[str, str]],
    annotation: str,
) -> None:
    shutil.copytree(template, destination)
    settings_path = destination / "settings.xml"
    tree = ET.parse(settings_path)
    root = tree.getroot()
    model = _config(root, "model")
    column_config = _config(model, "columns")
    _clear_children(column_config)
    for index, (name, value) in enumerate(columns):
        column = ET.SubElement(column_config, f"{TAG}config", key=str(index))
        ET.SubElement(
            column,
            f"{TAG}entry",
            key="name",
            type="xstring",
            value=name,
        )
        cell_type = ET.SubElement(column, f"{TAG}config", key="type")
        ET.SubElement(
            cell_type,
            f"{TAG}entry",
            key="cell_class",
            type="xstring",
            value="org.knime.core.data.def.StringCell",
        )
        ET.SubElement(
            cell_type,
            f"{TAG}entry",
            key="is_null",
            type="xboolean",
            value="false",
        )
        values = ET.SubElement(column, f"{TAG}config", key="values")
        ET.SubElement(
            values,
            f"{TAG}entry",
            key="array-size",
            type="xint",
            value="1",
        )
        ET.SubElement(
            values,
            f"{TAG}entry",
            key="0",
            type="xstring",
            value=value,
        )
    _set_entry(model, "numRows", "1")
    _set_entry(root, "state", "IDLE")
    _set_entry(root, "name", "입력 설정")
    _set_annotation(root, annotation)
    _write_xml(tree, settings_path)


def _external_tool(
    template: Path,
    destination: Path,
    *,
    title: str,
    script: Path,
    arguments: list[str],
    input_csv: Path,
    output_csv: Path,
) -> None:
    shutil.copytree(template, destination)
    settings_path = destination / "settings.xml"
    tree = ET.parse(settings_path)
    root = tree.getroot()
    model = _config(root, "model")
    _set_entry(model, "Inputfile name", str(input_csv))
    _set_entry(model, "IncludeColHdr", "true")
    _set_entry(model, "IncludeRowHdr", "false")
    _set_entry(model, "PathToExtExec", str(script))
    _set_entry(model, "ExtExecWorkDir", str(PROJECT))
    _set_entry(model, "ExtExecArgs", " ".join(arguments))
    _set_entry(model, "Output filename", str(output_csv))
    _set_entry(model, "ContainsColHdr", "true")
    _set_entry(model, "ContainsRowHdr", "false")
    _set_entry(root, "state", "IDLE")
    _set_entry(root, "name", title)
    _set_annotation(root, title)
    _write_xml(tree, settings_path)


def _table_view(template: Path, destination: Path, *, title: str) -> None:
    shutil.copytree(template, destination)
    settings_path = destination / "settings.xml"
    tree = ET.parse(settings_path)
    root = tree.getroot()
    view = _config(root, "view")
    displayed_columns = _config(view, "displayedColumnsV2")
    _set_entry(displayed_columns, "mode", "PATTERN")
    pattern_filter = _config(displayed_columns, "patternFilter")
    _set_entry(pattern_filter, "pattern", "*")
    _set_entry(root, "state", "IDLE")
    _set_entry(root, "name", title)
    _set_annotation(root, title)
    _write_xml(tree, settings_path)


def _workflow_node(
    template_node: ET.Element,
    *,
    node_id: int,
    settings_file: str,
    x: int,
    y: int,
) -> ET.Element:
    node = copy.deepcopy(template_node)
    node.set("key", f"node_{node_id}")
    _set_entry(node, "id", str(node_id))
    _set_entry(node, "node_settings_file", settings_file)
    bounds = _config(_config(node, "ui_settings"), "extrainfo.node.bounds")
    _set_entry(bounds, "0", str(x))
    _set_entry(bounds, "1", str(y))
    return node


def _connection(
    template_connection: ET.Element,
    *,
    index: int,
    source: int,
    destination: int,
) -> ET.Element:
    connection = copy.deepcopy(template_connection)
    connection.set("key", f"connection_{index}")
    _set_entry(connection, "sourceID", str(source))
    _set_entry(connection, "destID", str(destination))
    _set_entry(connection, "sourcePort", "1")
    _set_entry(connection, "destPort", "1")
    return connection


def _build_package(
    *,
    template_root: Path,
    package: Path,
    workflow_name: str,
    nodes: list[tuple[int, str, int, int]],
    connections: list[tuple[int, int]],
) -> Path:
    source_workflow = template_root / "Goal 1_5 Ora (#17)" / "workflow.knime"
    tree = ET.parse(source_workflow)
    root = tree.getroot()
    _set_entry(root, "name", workflow_name)
    _set_entry(root, "state", "IDLE")
    node_config = _config(root, "nodes")
    connection_config = _config(root, "connections")
    template_node = copy.deepcopy(next(iter(node_config)))
    template_connection = copy.deepcopy(next(iter(connection_config)))
    _clear_children(node_config)
    _clear_children(connection_config)
    for node_id, settings_file, x, y in nodes:
        node_config.append(
            _workflow_node(
                template_node,
                node_id=node_id,
                settings_file=settings_file,
                x=x,
                y=y,
            )
        )
    for index, (source, destination) in enumerate(connections):
        connection_config.append(
            _connection(
                template_connection,
                index=index,
                source=source,
                destination=destination,
            )
        )
    _write_xml(tree, package / "workflow.knime")
    for name in ("workflow-metadata.xml", "workflowset.meta"):
        shutil.copy2(template_root / name, package / name)
    return package


def _zip_package(package: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package.parent))


def build_factory(template_root: Path, output_root: Path) -> Path:
    name = "Multisensor_Synthetic_Factory_Goal1_5"
    package_root = output_root / name
    nodes: list[tuple[int, str, int, int]] = []
    connections: list[tuple[int, int]] = []
    creator_dir = "입력 설정 (#1)"
    _table_creator(
        template_root / "Goal 1_5 Ora (#17)" / "Table Creator (#3)",
        package_root / creator_dir,
        columns=[("실행", "합성 데이터 공장")],
        annotation="합성 데이터 생성 설정\nquick12-oracle-v1",
    )
    nodes.append((1, f"{creator_dir}/settings.xml", 40, 240))
    factory_dir = "합성 생성과 라벨 (#2)"
    factory_input = Path("/private/tmp/goal15_factory_input.csv")
    factory_output = Path("/private/tmp/goal15_factory_status.csv")
    _external_tool(
        template_root / "Goal 1_5 Ora (#17)" / "External Tool (#4)",
        package_root / factory_dir,
        title="합성 생성 + outcome 라벨",
        script=PROJECT / "scripts" / "knime_run_synthetic_factory.sh",
        arguments=[
            str(PROJECT),
            str(PROJECT / "configs" / "factory_quick.yaml"),
            str(PROJECT / "data" / "outcomes" / "quick12-oracle-v1" / "knime-factory-receipt.json"),
            str(factory_output),
        ],
        input_csv=factory_input,
        output_csv=factory_output,
    )
    nodes.append((2, f"{factory_dir}/settings.xml", 260, 240))
    connections.append((1, 2))
    status_view_dir = "생성 상태 보기 (#3)"
    _table_view(
        template_root / "Goal 1_5 Ora (#17)" / "Table View (#13)",
        package_root / status_view_dir,
        title="생성 실행 상태",
    )
    nodes.append((3, f"{status_view_dir}/settings.xml", 500, 80))
    connections.append((2, 3))

    views = [
        ("status", "생성 설정과 실행 상태"),
        ("participants", "참여자·기간·분할"),
        ("baseline", "사람별 생성 QA 기준선"),
        ("timeline", "사건과 5단계 타임라인"),
        ("behaviors", "행동 라벨 조합"),
        ("target_compare", "하드 네거티브와 타깃 비교"),
        ("split", "데이터 분할과 manifest"),
        ("devices", "세 장비와 동기화 상태"),
    ]
    for offset, (view_name, title) in enumerate(views):
        external_id = 10 + offset * 2
        view_id = external_id + 1
        input_csv = Path(f"/private/tmp/goal15_factory_view_{view_name}_input.csv")
        output_csv = Path(f"/private/tmp/goal15_factory_view_{view_name}.csv")
        external_dir = f"{title} 준비 (#{external_id})"
        view_dir = f"{title} 보기 (#{view_id})"
        _external_tool(
            template_root / "Goal 1_5 Ora (#17)" / "External Tool (#4)",
            package_root / external_dir,
            title=f"{title} 데이터",
            script=PROJECT / "scripts" / "knime_export_view.sh",
            arguments=[
                str(PROJECT),
                "factory",
                "quick12-oracle-v1",
                view_name,
                str(input_csv),
                str(output_csv),
            ],
            input_csv=input_csv,
            output_csv=output_csv,
        )
        _table_view(
            template_root / "Goal 1_5 Ora (#17)" / "Table View (#13)",
            package_root / view_dir,
            title=title,
        )
        y = 80 + offset * 110
        nodes.extend(
            [
                (external_id, f"{external_dir}/settings.xml", 500, y),
                (view_id, f"{view_dir}/settings.xml", 760, y),
            ]
        )
        connections.extend([(2, external_id), (external_id, view_id)])
    package = _build_package(
        template_root=template_root,
        package=package_root,
        workflow_name="합성데이터 공장 Goal 1.5",
        nodes=nodes,
        connections=connections,
    )
    destination = PROJECT / "knime" / f"{name}.knwf"
    _zip_package(package, destination)
    return destination


def build_registry(template_root: Path, output_root: Path) -> Path:
    name = "Multisensor_ML_Training_Registry"
    package_root = output_root / name
    nodes: list[tuple[int, str, int, int]] = []
    connections: list[tuple[int, int]] = []
    factory_receipt = (
        PROJECT / "data" / "outcomes" / "quick12-oracle-v1" / "knime-factory-receipt.json"
    )
    creator_dir = "Factory receipt 입력 (#1)"
    _table_creator(
        template_root / "Goal 1_5 Ora (#17)" / "Table Creator (#3)",
        package_root / creator_dir,
        columns=[
            ("receipt", str(factory_receipt)),
            ("status", "PASS"),
            ("stage", "labels"),
            ("message_ko", "합성 공장 receipt 입력"),
        ],
        annotation="Factory receipt\n학습 레지스트리 시작",
    )
    nodes.append((1, f"{creator_dir}/settings.xml", 30, 260))
    stages = [
        ("labels", "라벨 QA"),
        ("baseline", "전역·개인 기준선"),
        ("types", "STD 타입·OOD"),
        ("stage-model", "모델 1 사건·5단계"),
        ("behavior-model", "모델 2 행동 다중 라벨"),
        ("evaluate", "validation·stress·locked test"),
        ("predict", "예측 감사"),
        ("route-ko", "한국어 결과 라우터"),
    ]
    previous = 1
    for offset, (stage, title) in enumerate(stages):
        node_id = 2 + offset
        input_csv = Path(f"/private/tmp/goal15_registry_{stage}_input.csv")
        output_csv = Path(f"/private/tmp/goal15_registry_{stage}.csv")
        receipt = (
            PROJECT
            / "artifacts"
            / "registry"
            / "quick12-oracle-v1"
            / "knime-receipts"
            / f"{stage}.json"
        )
        node_dir = f"{title} (#{node_id})"
        _external_tool(
            template_root / "Goal 1_5 Ora (#17)" / "External Tool (#4)",
            package_root / node_dir,
            title=title,
            script=PROJECT / "scripts" / "knime_run_registry_stage.sh",
            arguments=[
                str(PROJECT),
                str(PROJECT / "configs" / "training_registry.yaml"),
                stage,
                "knime-training-quick12-v1",
                str(input_csv),
                str(receipt),
                str(output_csv),
            ],
            input_csv=input_csv,
            output_csv=output_csv,
        )
        nodes.append((node_id, f"{node_dir}/settings.xml", 240 * node_id, 260))
        connections.append((previous, node_id))
        previous = node_id

    final_view_dir = "최종 receipt 보기 (#10)"
    _table_view(
        template_root / "Goal 1_5 Ora (#17)" / "Table View (#13)",
        package_root / final_view_dir,
        title="최종 연결 상태",
    )
    nodes.append((10, f"{final_view_dir}/settings.xml", 2400, 100))
    connections.append((previous, 10))
    views = [
        ("dataset", "Dataset & Split"),
        ("baseline", "Personal Baseline & STD"),
        ("performance", "Pattern & Behavior Performance"),
        ("stress", "Noise Stress"),
        ("registry", "Model Registry"),
        ("korean", "한국어 예측 결과"),
    ]
    for offset, (view_name, title) in enumerate(views):
        external_id = 20 + offset * 2
        view_id = external_id + 1
        input_csv = Path(f"/private/tmp/goal15_registry_view_{view_name}_input.csv")
        output_csv = Path(f"/private/tmp/goal15_registry_view_{view_name}.csv")
        external_dir = f"{title} 준비 (#{external_id})"
        view_dir = f"{title} 보기 (#{view_id})"
        _external_tool(
            template_root / "Goal 1_5 Ora (#17)" / "External Tool (#4)",
            package_root / external_dir,
            title=f"{title} 데이터",
            script=PROJECT / "scripts" / "knime_export_view.sh",
            arguments=[
                str(PROJECT),
                "registry",
                "quick12-oracle-v1",
                view_name,
                str(input_csv),
                str(output_csv),
            ],
            input_csv=input_csv,
            output_csv=output_csv,
        )
        _table_view(
            template_root / "Goal 1_5 Ora (#17)" / "Table View (#13)",
            package_root / view_dir,
            title=title,
        )
        y = 60 + offset * 120
        nodes.extend(
            [
                (external_id, f"{external_dir}/settings.xml", 2400, y),
                (view_id, f"{view_dir}/settings.xml", 2680, y),
            ]
        )
        connections.extend([(previous, external_id), (external_id, view_id)])
    package = _build_package(
        template_root=template_root,
        package=package_root,
        workflow_name="학습·모델 버전 레지스트리 Goal 1.5",
        nodes=nodes,
        connections=connections,
    )
    destination = PROJECT / "knime" / f"{name}.knwf"
    _zip_package(package, destination)
    return destination


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="goal15-knime-build-") as temporary:
        workspace = Path(temporary)
        with zipfile.ZipFile(TEMPLATE) as archive:
            archive.extractall(workspace / "template")
        template_root = workspace / "template" / "Multisensor_ML_Goal1_5"
        output_root = workspace / "output"
        output_root.mkdir()
        factory = build_factory(template_root, output_root)
        registry = build_registry(template_root, output_root)
        print(factory)
        print(registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
