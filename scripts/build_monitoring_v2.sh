#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${project_root}/.venv/bin/python"
cli="${project_root}/.venv/bin/multisensor-ml"

if [[ ! -x "${cli}" ]]; then
  echo "multisensor-ml virtualenv entrypoint is missing: ${cli}" >&2
  exit 127
fi

cd "${project_root}"
"${cli}" materialize-synthetic \
  --config configs/standard_pattern_v2_quick.yaml
"${cli}" monitor-v2 run \
  --config configs/monitoring_v2_quick.yaml

"${python_bin}" - <<'PY'
import json
from pathlib import Path

result = Path("artifacts/monitoring-v2/standard-pattern-v2-quick/result.json")
payload = json.loads(result.read_text(encoding="utf-8"))
assert payload["locked_test_read"] is False
assert payload["real_data_status"] == "NOT VERIFIED"
assert set(payload["variants"]) == {"galaxy_watch", "galaxy_watch_h10"}
print(json.dumps({"status": "READY", "result": str(result.resolve())}, ensure_ascii=False))
PY
