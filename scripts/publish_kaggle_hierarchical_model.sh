#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <verified-package-directory>" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
package_root="$(cd "$1" && pwd)"
model_handle="bjcoding/multisensor-goal15-hierarchical"
variation_handle="${model_handle}/scikitLearn/oracle-sanity-v1"
remote_variation_handle="${model_handle}/ScikitLearn/oracle-sanity-v1"
model_url="https://www.kaggle.com/models/bjcoding/multisensor-goal15-hierarchical"
multisensor-ml() {
  "${repo_root}/.venv/bin/multisensor-ml" "$@"
}

multisensor-ml kaggle-model verify --package "${package_root}" >/dev/null
local_hash="$(${repo_root}/.venv/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["archive_sha256"])' \
  "${package_root}/model_manifest.json")"

readback_root="$(mktemp -d /private/tmp/multisensor-model-readback.XXXXXX)"
status=""

latest_version_once() {
  local rows
  rows="$(kaggle models variations versions list "${remote_variation_handle}" \
    --page-size 200 --format json 2>/dev/null)" || return 1
  printf '%s' "${rows}" | "${repo_root}/.venv/bin/python" -c '
import json
import sys

rows = json.load(sys.stdin)
versions = [row.get("versionNumber", row.get("version")) for row in rows]
versions = [int(value) for value in versions if value is not None]
if versions:
    print(max(versions))
'
}

latest_version() {
  local minimum_version="${1:-1}"
  local version
  for attempt in {1..12}; do
    version="$(latest_version_once || true)"
    if [[ -n "${version}" ]] && (( version >= minimum_version )); then
      printf '%s\n' "${version}"
      return 0
    fi
    sleep 5
  done
  echo "Kaggle variation exists but no version became readable" >&2
  return 1
}

variation_exists=false
expected_version=1
current_version="$(latest_version_once || true)"
if [[ -n "${current_version}" ]]; then
  variation_exists=true
fi

download_and_verify() {
  local version="$1"
  local destination="${readback_root}/v${version}"
  mkdir -p "${destination}"
  kaggle models variations versions download \
    "${remote_variation_handle}/${version}" -p "${destination}" --untar -f -q >/dev/null
  local manifest
  manifest="$(find "${destination}" -name model_manifest.json -type f -print -quit)"
  if [[ -z "${manifest}" ]]; then
    echo "remote readback is missing model_manifest.json" >&2
    exit 1
  fi
  multisensor-ml kaggle-model verify --package "$(dirname "${manifest}")" >/dev/null
  printf '%s\n' "${manifest}"
}

if [[ "${variation_exists}" == false ]]; then
  kaggle models create -p "${package_root}"
  kaggle models variations create -p "${package_root}" -r skip
  status="CREATED"
else
  current_manifest="$(download_and_verify "${current_version}")"
  remote_hash="$(${repo_root}/.venv/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["archive_sha256"])' \
    "${current_manifest}")"
  if [[ "${remote_hash}" == "${local_hash}" ]]; then
    status="REUSED"
    version="${current_version}"
    remote_manifest="${current_manifest}"
  else
    expected_version="$((current_version + 1))"
    kaggle models variations versions create "${remote_variation_handle}" \
      -p "${package_root}" -n "Verified hierarchical oracle/sanity candidate" -r skip
    status="VERSIONED"
  fi
fi

if [[ "${status}" != "REUSED" ]]; then
  version="$(latest_version "${expected_version}")"
  remote_manifest="$(download_and_verify "${version}")"
fi
remote_hash="$(${repo_root}/.venv/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["archive_sha256"])' \
  "${remote_manifest}")"
if [[ "${remote_hash}" != "${local_hash}" ]]; then
  echo "remote package hash differs from verified local package" >&2
  exit 1
fi

git_commit="$(git -C "${repo_root}" rev-parse HEAD)"
file_inventory="$(find "$(dirname "${remote_manifest}")" -maxdepth 1 -type f \
  -exec basename {} \; | sort | "${repo_root}/.venv/bin/python" -c \
  'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')"
"${repo_root}/.venv/bin/python" - \
  "${package_root}/remote-receipt.json" "${status}" "${variation_handle}" \
  "${version}" "${model_url}" "${local_hash}" "${git_commit}" "${file_inventory}" <<'PY'
import datetime
import json
import pathlib
import sys

destination, status, handle, version, url, package_hash, commit, files = sys.argv[1:]
receipt = {
    "status": status,
    "handle": handle,
    "version": int(version),
    "url": f"{url}/scikitLearn/oracle-sanity-v1/{version}",
    "package_sha256": package_hash,
    "files": json.loads(files),
    "is_private": True,
    "locked_test_read": False,
    "git_commit": commit,
    "readback_time_utc": datetime.datetime.now(datetime.UTC).isoformat(),
}
pathlib.Path(destination).write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
PY
