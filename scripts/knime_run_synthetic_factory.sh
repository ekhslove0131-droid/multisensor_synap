#!/bin/zsh
set -euo pipefail

project_root="${1:?ML project path is required}"
config_path="${2:?factory config path is required}"
receipt_path="${3:?factory receipt path is required}"
status_output="${4:?status CSV path is required}"

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

cd "${project_root}"
"${uv_binary}" run --frozen multisensor-ml factory run \
  --config "${config_path}" \
  --output-receipt "${receipt_path}"

print 'receipt,status,stage,message_ko' > "${status_output}"
print "${receipt_path},PASS,labels,합성 데이터와 결과 라벨 생성 완료" >> "${status_output}"
