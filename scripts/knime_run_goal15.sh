#!/bin/zsh
set -euo pipefail

project_root="${1:?ML project path is required}"
config_path="${2:?experiment config path is required}"
run_mode="${3:-run-all}"

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
case "${run_mode}" in
  run-all)
    exec "${uv_binary}" run --frozen multisensor-ml run-all --config "${config_path}"
    ;;
  materialize-synthetic)
    exec "${uv_binary}" run --frozen multisensor-ml materialize-synthetic --config "${config_path}"
    ;;
  train)
    exec "${uv_binary}" run --frozen multisensor-ml train --experiment "${config_path}"
    ;;
  *)
    print -u2 "unsupported run mode: ${run_mode}"
    exit 64
    ;;
esac
