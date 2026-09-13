#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${1:-"${LLAMA_BACKEND:-auto}"}"

if ! command -v go >/dev/null 2>&1; then
  echo "Go is required to build swarmd" >&2
  exit 1
fi

"$ROOT/scripts/build-llama.sh" "$BACKEND"
mkdir -p "$ROOT/bin"
(cd "$ROOT" && go build -trimpath -o "$ROOT/bin/swarmd" ./cmd/swarmd)

echo
echo "layers is ready."
echo "  CLI:     $ROOT/bin/swarmd"
echo "  backend: $BACKEND"
