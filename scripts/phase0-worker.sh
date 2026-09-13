#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RPC_SERVER="${RPC_SERVER:-"$ROOT/build/llama/bin/ggml-rpc-server"}"
RPC_HOST="${RPC_HOST:-0.0.0.0}"
RPC_PORT="${RPC_PORT:-50052}"

if [[ ! -x "$RPC_SERVER" ]]; then
  echo "rpc-server not found: $RPC_SERVER" >&2
  echo "Run scripts/build-llama.sh first or set RPC_SERVER." >&2
  exit 1
fi

exec "$RPC_SERVER" --host "$RPC_HOST" --port "$RPC_PORT"
