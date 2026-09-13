#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWARMD="${SWARMD:-"$ROOT/bin/swarmd"}"

if [[ ! -x "$SWARMD" ]]; then
  echo "swarmd was not built: $SWARMD" >&2
  echo "Run scripts/build.sh first." >&2
  exit 1
fi

exec "$SWARMD" join "$@"
