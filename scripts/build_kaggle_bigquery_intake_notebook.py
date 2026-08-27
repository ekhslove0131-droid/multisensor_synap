# mypy: disable-error-code="no-untyped-call,no-any-return"

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

# Notebook cells are intentionally literal and reviewable.
# ruff: noqa: E501


def build_notebook() -> nbformat.NotebookNode:
    notebook = new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    notebook.cells = [
        new_markdown_cell(
            "# Kidsignal 실제 BigQuery cohort intake\n\n"
            "이 노트북은 전용 model-reader ADC로 authorized TRAIN/VALIDATION view 한 개만 읽고, "
            "24-field 표준 plane 또는 26-field 행동 plane의 schema·UUIDv4·public digest·split을 "
            "검증합니다. cohort 선택값은 외부 create-only model-sync receipt 하나에서만 가져오며, "
            "행 원본과 개인 식별자는 저장하지 않고 준비된 배열의 shape만 기록합니다.\n\n"
            "성공 상태도 `READY_FOR_SYNC_NOT_TRAINED`이며 fit·평가·ONNX·bundle·승격을 실행하지 않습니다.",
            id="overview",
        ),
        new_code_cell(
            "import json\n"
            "import os\n"
            "from pathlib import Path\n"
            "\n"
            "from multisensor_ml.bigquery_training_preflight import GCP_LOCATION, GCP_PROJECT\n"
            "from multisensor_ml.kaggle_bigquery_intake import (\n"
            "    run_receipt_bound_kaggle_bigquery_intake,\n"
            "    write_kaggle_intake_artifact,\n"
            ")\n"
            "\n"
            "def required_env(name: str) -> str:\n"
            "    value = os.environ.get(name, '').strip()\n"
            "    if not value:\n"
            "        raise RuntimeError(f'{name} must be configured externally')\n"
            "    return value\n"
            "\n"
            "MODEL_SYNC_RECEIPT_PATH = Path(required_env('KIDSIGNAL_MODEL_SYNC_RECEIPT_PATH'))\n"
            "try:\n"
            "    MODEL_SYNC_RECEIPT = json.loads(MODEL_SYNC_RECEIPT_PATH.read_text(encoding='utf-8'))\n"
            "except (OSError, json.JSONDecodeError) as error:\n"
            "    raise RuntimeError('model sync receipt JSON could not be read') from error\n"
            "OUTPUT_PATH = Path(os.environ.get('KIDSIGNAL_INTAKE_OUTPUT', '/kaggle/working/kidsignal_bigquery_intake_receipt.json'))\n"
            "RUN_TRAINING = False\n"
            "RUN_EVALUATION = False\n"
            "RUN_ONNX_EXPORT = False\n"
            "RUN_BUNDLE = False\n"
            "RUN_PROMOTION = False\n"
            "assert not any((RUN_TRAINING, RUN_EVALUATION, RUN_ONNX_EXPORT, RUN_BUNDLE, RUN_PROMOTION))",
            id="configuration",
        ),
        new_code_cell(
            "from datetime import datetime, timezone\n"
            "\n"
            "try:\n"
            "    from google.cloud import bigquery\n"
            "except ModuleNotFoundError as error:\n"
            "    raise RuntimeError('google-cloud-bigquery is a late runtime dependency for this execution cell') from error\n"
            "\n"
            "try:\n"
            "    bigquery_client = bigquery.Client(project=GCP_PROJECT, location=GCP_LOCATION)\n"
            "except Exception as error:\n"
            "    raise RuntimeError('BigQuery ADC credentials are required at runtime') from error\n"
            "credentials = getattr(bigquery_client, '_credentials', None)\n"
            "observed_principal = getattr(credentials, 'service_account_email', None)\n"
            "if not isinstance(observed_principal, str) or not observed_principal:\n"
            "    raise RuntimeError('ADC model-reader principal could not be verified')\n"
            "\n"
            "observed_at_utc = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')\n"
            "intake_artifact = run_receipt_bound_kaggle_bigquery_intake(\n"
            "    model_sync_receipt=MODEL_SYNC_RECEIPT,\n"
            "    client=bigquery_client,\n"
            "    observed_at_utc=observed_at_utc,\n"
            "    observed_principal=observed_principal,\n"
            ")\n"
            "write_kaggle_intake_artifact(OUTPUT_PATH, intake_artifact)",
            id="read-only-sdk-execution",
        ),
        new_code_cell(
            "import json\n"
            "\n"
            "receipt = intake_artifact['readiness_receipt']\n"
            "if receipt['fit_call_count'] != 0:\n"
            "    raise RuntimeError('intake boundary attempted model fitting')\n"
            "public_summary = {\n"
            "    'mode': intake_artifact['mode'],\n"
            "    'status': receipt['status'],\n"
            "    'row_count': receipt['row_count'],\n"
            "    'training_ready': receipt['training_ready'],\n"
            "    'fit_call_count': receipt['fit_call_count'],\n"
            "    'shape_summary': intake_artifact['summary'],\n"
            "}\n"
            "print(json.dumps(public_summary, ensure_ascii=False, sort_keys=True))",
            id="bounded-readback",
        ),
        new_markdown_cell(
            "## 실행 경계\n\n"
            "`BLOCKED_*`는 입력 계약 또는 실제 cohort 준비가 충족되지 않았음을 뜻합니다. "
            "`READY_FOR_SYNC_NOT_TRAINED`도 학습 승인이 아니며, 별도의 명시적 실제 cohort 학습 작업으로만 이어질 수 있습니다.",
            id="boundary",
        ),
    ]
    nbformat.validate(notebook)
    return notebook


def main() -> None:
    destination = Path("kaggle/10_kidsignal_bigquery_intake.ipynb")
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), destination)


if __name__ == "__main__":
    main()
