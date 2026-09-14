#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${LLAMA_SOURCE:-"${XDG_CACHE_HOME:-"$HOME/.cache"}/layers/llama.cpp"}"
BUILD="${LLAMA_BUILD:-"$ROOT/build/llama"}"
BACKEND="${1:-"${LLAMA_BACKEND:-auto}"}"
REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
REF="${LLAMA_REF:-}"

if [[ -z "$REF" && -f "$ROOT/llamacpp/llama.cpp.ref" ]]; then
  REF="$(awk -F= '$1 == "commit" {print $2}' "$ROOT/llamacpp/llama.cpp.ref")"
fi

if [[ ! -f "$SOURCE/CMakeLists.txt" ]]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "git is required to download llama.cpp" >&2
    exit 1
  fi
  cat >&2 <<EOF
llama.cpp was not found at:
  $SOURCE

It is kept outside the layers source tree. Set LLAMA_SOURCE to an existing
checkout, or let this script download it.
Example:
  LLAMA_SOURCE=/path/to/llama.cpp $0 cuda
EOF
  mkdir -p "$(dirname "$SOURCE")"
  git clone --filter=blob:none --no-checkout "$REPO" "$SOURCE"
  if [[ -n "$REF" ]]; then
    git -C "$SOURCE" fetch --depth 1 origin "$REF"
    git -C "$SOURCE" checkout --detach "$REF"
  fi
fi

if [[ -n "$REF" && -d "$SOURCE/.git" ]]; then
  current_ref="$(git -C "$SOURCE" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$current_ref" != "$REF" ]]; then
    echo "Updating llama.cpp checkout to pinned commit $REF."
    git -C "$SOURCE" fetch --depth 1 origin "$REF"
    git -C "$SOURCE" checkout --detach "$REF"
  fi
fi

if [[ -f "$BUILD/CMakeCache.txt" ]]; then
  configured_source="$(awk -F= '$1 == "CMAKE_HOME_DIRECTORY" {print $2}' "$BUILD/CMakeCache.txt")"
  if [[ "$configured_source" != "$SOURCE" ]]; then
    echo "Resetting stale CMake cache for the new llama.cpp location."
    rm -rf "$BUILD"
  fi
fi

case "$BACKEND" in
  auto)
    if [[ "$(uname -s)" == "Darwin" ]]; then
      BACKEND=metal
    elif command -v nvidia-smi >/dev/null 2>&1; then
      BACKEND=cuda
    elif command -v vulkaninfo >/dev/null 2>&1; then
      BACKEND=vulkan
    else
      BACKEND=cpu
    fi
    ;;
  cuda)
    if ! command -v nvcc >/dev/null 2>&1; then
      echo "CUDA backend requested, but nvcc was not found." >&2
      exit 2
    fi
    ;;
  metal)
    if [[ "$(uname -s)" != "Darwin" ]]; then
      echo "Metal is only available on macOS. This machine is $(uname -s); use: $0 cuda" >&2
      exit 2
    fi
    ;;
  vulkan)
    if ! command -v vulkaninfo >/dev/null 2>&1; then
      echo "Vulkan backend requested, but vulkaninfo was not found." >&2
      exit 2
    fi
    ;;
  cpu)
    ;;
  *)
    echo "usage: $0 [auto|cuda|metal|vulkan|cpu]" >&2
    exit 2
    ;;
esac

declare -a cmake_args=(
  -S "$SOURCE"
  -B "$BUILD"
  -DGGML_RPC=ON
  -DLLAMA_BUILD_SERVER=ON
  -DLLAMA_BUILD_COMMON=ON
  -DGGML_NATIVE=ON
  -DGGML_CUDA=OFF
  -DGGML_METAL=OFF
  -DGGML_VULKAN=OFF
)

case "$BACKEND" in
  cuda)
    cmake_args+=(-DGGML_CUDA=ON)
    ;;
  metal)
    cmake_args+=(-DGGML_METAL=ON)
    ;;
  vulkan)
    cmake_args+=(-DGGML_VULKAN=ON)
    ;;
  cpu)
    ;;
esac

echo "Configuring llama.cpp backend=$BACKEND"
cmake "${cmake_args[@]}"
cmake --build "$BUILD" --config Release --target ggml-rpc-server llama-server llama-cli -j "${JOBS:-$(nproc 2>/dev/null || echo 4)}"

echo
echo "Built binaries:"
find "$BUILD" -type f \( -name ggml-rpc-server -o -name llama-server -o -name llama-cli \) -print
