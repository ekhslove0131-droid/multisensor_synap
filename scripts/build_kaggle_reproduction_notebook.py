from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def _path_literal(value: Path | None) -> str:
    return "None" if value is None else repr(str(value.resolve()))


def build_notebook(
    *,
    model_root_override: Path | None = None,
    dataset_root_override: Path | None = None,
) -> nbformat.NotebookNode:
    notebook = new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    notebook.cells = [
        new_markdown_cell(
            "# Goal 1.5 전체 계층형 합성 모델 재현\n\n"
            "## 목표\n기존 1차 모델을 새 학습 없이 합성 validation 샘플에 적용합니다. "
            "결과 범위는 `oracle/sanity`, 실제 정확도는 `NOT VERIFIED`입니다."
        ),
        new_markdown_cell("## 설정"),
        new_code_cell(
            "from pathlib import Path\n"
            "import hashlib\nimport importlib.util\nimport json\nimport subprocess\n"
            "import sys\nimport tarfile\n\n"
            "import pandas as pd\n\n"
            "RUN_TRAINING = False\nRUN_LOCKED_TEST = False\nUSE_GPU = False\n"
            f"MODEL_ROOT_OVERRIDE = {_path_literal(model_root_override)}\n"
            f"DATASET_ROOT_OVERRIDE = {_path_literal(dataset_root_override)}\n"
            "assert not RUN_TRAINING and not RUN_LOCKED_TEST and not USE_GPU\n"
            "print({\n"
            "    'run_training': RUN_TRAINING,\n"
            "    'run_locked_test': RUN_LOCKED_TEST,\n"
            "    'use_gpu': USE_GPU,\n"
            "})"
        ),
        new_markdown_cell("## 입력 탐색"),
        new_code_cell(
            "def sha256_file(path: Path) -> str:\n"
            "    digest = hashlib.sha256()\n"
            "    with path.open('rb') as handle:\n"
            "        for chunk in iter(lambda: handle.read(1024 * 1024), b''):\n"
            "            digest.update(chunk)\n"
            "    return digest.hexdigest()\n\n"
            "if MODEL_ROOT_OVERRIDE:\n"
            "    model_root = Path(MODEL_ROOT_OVERRIDE)\n"
            "else:\n"
            "    manifests = sorted(Path('/kaggle/input').rglob('model_manifest.json'))\n"
            "    if len(manifests) != 1:\n"
            "        raise RuntimeError(f'Kaggle Model manifest must be unique: {manifests}')\n"
            "    model_root = manifests[0].parent\n"
            "if DATASET_ROOT_OVERRIDE:\n"
            "    dataset_root = Path(DATASET_ROOT_OVERRIDE)\n"
            "else:\n"
            "    candidates = []\n"
            "    for path in Path('/kaggle/input').rglob('manifest.json'):\n"
            "        try:\n"
            "            manifest = json.loads(path.read_text())\n"
            "            if manifest.get('prepared_schema') == 'goal1.5/prepared/v1':\n"
            "                candidates.append(path.parent)\n"
            "        except (json.JSONDecodeError, OSError):\n"
            "            pass\n"
            "    if len(candidates) != 1:\n"
            "        message = f'prepared synthetic Dataset must be unique: {candidates}'\n"
            "        raise RuntimeError(message)\n"
            "    dataset_root = candidates[0]\n"
            "print({'model_root': str(model_root), 'dataset_root': str(dataset_root)})"
        ),
        new_markdown_cell("## 무결성 검증"),
        new_code_cell(
            "outer = json.loads((model_root / 'model_manifest.json').read_text())\n"
            "archive = model_root / outer['archive_path']\n"
            "if sha256_file(archive) != outer['archive_sha256']:\n"
            "    raise RuntimeError('outer package hash mismatch')\n"
            "payload_root = model_root / 'extracted'\n"
            "if not payload_root.is_dir():\n"
            "    payload_root = Path('/kaggle/working/model_payload')\n"
            "    payload_root.mkdir(parents=True, exist_ok=False)\n"
            "    with tarfile.open(archive, 'r:gz') as bundle:\n"
            "        unsafe = any(\n"
            "            member.name.startswith('/') or '..' in Path(member.name).parts\n"
            "            for member in bundle.getmembers()\n"
            "        )\n"
            "        if unsafe:\n"
            "            raise RuntimeError('unsafe archive member')\n"
            "        bundle.extractall(payload_root, filter='data')\n"
            "wheel = next((payload_root / 'wheel').glob('multisensor_ml-*.whl'))\n"
            "if importlib.util.find_spec('multisensor_ml') is None:\n"
            "    command = [sys.executable, '-m', 'pip', 'install', '--no-deps', str(wheel)]\n"
            "    subprocess.check_call(command)\n"
            "from multisensor_ml.kaggle_model_package import verify_payload\n"
            "verify_payload(payload_root)\n"
            "print({'package_sha256': outer['archive_sha256'], 'integrity': 'VERIFIED'})"
        ),
        new_markdown_cell("## 모델 로딩과 계층형 추론"),
        new_code_cell(
            "from multisensor_ml.kaggle_reproduce import (\n"
            "    compare_expected, load_hierarchical_package, predict_hierarchical,\n"
            ")\n"
            "package = load_hierarchical_package(payload_root)\n"
            "sample = pd.read_parquet(payload_root / 'sample/sample_input.parquet')\n"
            "expected = pd.read_parquet(payload_root / 'sample/expected_output.parquet')\n"
            "prediction = predict_hierarchical(package, sample)\n"
            "comparison = compare_expected(prediction, expected)\n"
            "prediction.head(10)"
        ),
        new_markdown_cell("## 재현 비교와 주요 결과"),
        new_code_cell(
            "receipt = {\n"
            "    'status': comparison.status,\n"
            "    'compared_rows': comparison.compared_rows,\n"
            "    'first_mismatch_column': comparison.first_mismatch_column,\n"
            "    'model_package_sha256': outer['archive_sha256'],\n"
            "    'dataset_root': str(dataset_root),\n"
            "    'device': 'cpu',\n"
            "    'run_training': RUN_TRAINING,\n"
            "    'run_locked_test': RUN_LOCKED_TEST,\n"
            "    'data_status': 'oracle/sanity',\n"
            "    'real_data_status': 'NOT VERIFIED',\n"
            "}\n"
            "receipt_text = json.dumps(receipt, indent=2, sort_keys=True) + '\\n'\n"
            "Path('reproduction_receipt.json').write_text(receipt_text)\n"
            "print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))\n"
            "if comparison.status != 'REPRODUCED':\n"
            "    raise RuntimeError(f'FAILED: {comparison.first_mismatch_column}')"
        ),
        new_markdown_cell(
            "## 다음 아이디어\n\n개인 기준선, STD-A, 사건, 5단계, 행동 확률을 함께 보되 "
            "행동은 보조지표로 해석합니다. 실제 센서 단계에서는 PostgreSQL/TimescaleDB "
            "운영 저장계약과 원시 Parquet를 분리해 검증합니다."
        ),
    ]
    return notebook


def main() -> None:
    destination = Path("kaggle/07_hierarchical_model_reproduction.ipynb")
    notebook = build_notebook()
    nbformat.validate(notebook)
    nbformat.write(notebook, destination)


if __name__ == "__main__":
    main()
