#!/bin/zsh
set -euo pipefail

project_root="${1:?ML project path is required}"
config_path="${2:?training registry config path is required}"
stage="${3:?stage is required}"
run_id="${4:?run id is required}"
input_csv="${5:?upstream CSV is required}"
receipt_path="${6:?output receipt path is required}"
status_output="${7:?status CSV path is required}"

if command -v uv >/dev/null 2>&1; then
  uv_binary="$(command -v uv)"
elif [[ -x "${project_root}/.bootstrap-uv/bin/uv" ]]; then
  uv_binary="${project_root}/.bootstrap-uv/bin/uv"
elif [[ -x "${project_root}/../multisensor_synth/.bootstrap-uv/bin/uv" ]]; then
  uv_binary="${project_root}/../multisensor_synth/.bootstrap-uv/bin/uv"
else
  print -u2 "uv executable was not found"
  exit 127
fi

upstream_line="$(sed -n '2p' "${input_csv}")"
upstream_receipt="${upstream_line%%,*}"
upstream_receipt="${upstream_receipt#\"}"
upstream_receipt="${upstream_receipt%\"}"
if [[ ! -f "${upstream_receipt}" ]]; then
  print -u2 "upstream receipt is missing: ${upstream_receipt}"
  exit 66
fi

cd "${project_root}"
"${uv_binary}" run --frozen multisensor-ml registry run-stage \
  --config "${config_path}" \
  --stage "${stage}" \
  --run-id "${run_id}" \
  --input-receipt "${upstream_receipt}" \
  --output-receipt "${receipt_path}"

print 'receipt,status,stage,message_ko' > "${status_output}"
print "${receipt_path},PASS,${stage},${stage} 단계 완료" >> "${status_output}"
