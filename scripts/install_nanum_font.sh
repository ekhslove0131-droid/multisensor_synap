#!/usr/bin/env bash
set -euo pipefail

user_home="$(python3 -c 'from pathlib import Path; print(Path.home())')"
font_dir="${user_home}/Library/Fonts"
font_path="${font_dir}/NanumGothic-Regular.ttf"
font_url="https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-Regular.ttf"

mkdir -p "${font_dir}"
if [[ ! -s "${font_path}" ]]; then
  curl --fail --location --silent --show-error "${font_url}" --output "${font_path}"
fi
printf 'NanumGothic installed: %s\n' "${font_path}"

