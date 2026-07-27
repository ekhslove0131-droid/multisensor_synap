#!/bin/zsh
set -euo pipefail

project_root="${1:?ML project path is required}"
domain="${2:?factory or registry domain is required}"
series_id="${3:?series id is required}"
view_name="${4:?view name is required}"
input_csv="${5:?upstream CSV is required}"
output_csv="${6:?view CSV path is required}"

if [[ ! -s "${input_csv}" ]]; then
  print -u2 "upstream status CSV is missing or empty: ${input_csv}"
  exit 66
fi

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
"${uv_binary}" run --frozen python -m multisensor_ml.knime_views \
  --project-root "${project_root}" \
  --domain "${domain}" \
  --series "${series_id}" \
  --view "${view_name}" \
  --output "${output_csv}"
