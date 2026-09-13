#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CLI="${LLAMA_CLI:-"$ROOT/build/llama/bin/llama-cli"}"
MODEL="${1:-"${MODEL:-"$ROOT/models/model.gguf"}"}"
RPC="${RPC:-}"
PROMPT="${PROMPT:-hello}"

if [[ ! -x "$LLAMA_CLI" ]]; then
  echo "llama-cli not found: $LLAMA_CLI" >&2
  echo "Run scripts/build-llama.sh first or set LLAMA_CLI." >&2
  exit 1
fi
if [[ -z "$RPC" ]]; then
  echo "Set RPC to the worker address, for example:" >&2
  echo "  RPC=192.168.1.20:50052 $0 models/qwen3-1.7b-q4_k.gguf" >&2
  exit 2
fi
if [[ ! -f "$MODEL" ]]; then
  echo "model not found: $MODEL" >&2
  exit 1
fi

exec "$LLAMA_CLI" \
  -m "$MODEL" \
  --rpc "$RPC" \
  --split-mode layer \
  -ngl 99 \
  -p "$PROMPT"
