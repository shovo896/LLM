#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cli="$project_dir/.tools/arduino-cli/arduino-cli"
config="$project_dir/arduino-cli.yaml"

if [[ ! -x "$cli" ]]; then
  echo "Arduino CLI is missing. Run the project setup again."
  exit 1
fi

"$cli" --config-file "$config" core update-index
"$cli" --config-file "$config" core install arduino:avr
"$cli" --config-file "$config" compile \
  --fqbn arduino:avr:uno \
  --output-dir "$project_dir/build" \
  "$project_dir"
