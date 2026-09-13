# layers

Native LAN swarm: one GGUF split across devices via llama.cpp RPC.
Read [PLAN.md](PLAN.md) before writing code.

Project name: layers.
Idea only — not a fork of SwarmLLM.

## First build

```bash
cd native
git clone https://github.com/ggml-org/llama.cpp.git llamacpp/llama.cpp
scripts/build-llama.sh vulkan
go run ./cmd/swarmd host --model ./models/model.gguf
```

The first milestone is a manual two-machine llama.cpp RPC test. The Go room
control plane comes after that test passes.

The current llama.cpp checkout names its RPC executable `ggml-rpc-server`.

## Build on another machine

Copy this `native` directory to the second machine, then run:

```bash
cd native
scripts/build.sh auto
```

Use an explicit backend when needed:

```bash
scripts/build.sh cuda    # NVIDIA
scripts/build.sh metal   # Apple Silicon
scripts/build.sh vulkan  # Linux/Windows GPU
scripts/build.sh cpu     # CPU fallback
```

The build detects the machine's architecture and GPU backend. Do not copy the
CUDA or Metal llama.cpp binaries from another computer; build them locally.

### Start a LAN room

On the host machine:

```bash
scripts/run-host.sh \
  --llama-hf tensorblock/Qwen_Qwen3-1.7B-GGUF:Q3_K_M \
  --model "Qwen3 1.7B Q3_K_M" \
  --vram-mb 120000
```

Copy the printed room code and host LAN IP. On the second machine:

```bash
scripts/run-worker.sh \
  --host HOST_LAN_IP \
  --code ROOM_CODE \
  --vram-mb 4096
```

The host needs TCP port `7830` reachable from workers. Each worker needs TCP
port `50052` reachable from the host. Keep this limited to a trusted LAN; the
current llama.cpp RPC endpoint is not designed for public internet exposure.

## Local UI and LLM

For a solo local test, start llama.cpp's OpenAI-compatible server manually:

```bash
./build/llama/bin/llama-server \
  -m ./models/model.gguf \
  --host 127.0.0.1 \
  --port 7841 \
  -ngl 99
```

Then start the layers host:

```bash
go run ./cmd/swarmd host --model ./models/model.gguf --llama-model ./models/model.gguf
```

Open <http://127.0.0.1:7840>. The UI proxies chat requests to llama-server
and shows the room code, worker count, model, and LLM readiness.

For the automatic swarm path, let `swarmd` start llama-server from Hugging Face:

```bash
go run ./cmd/swarmd host \
  --llama-hf tensorblock/Qwen_Qwen3-1.7B-GGUF:Q3_K_M \
  --model "Qwen3 1.7B Q3_K_M" \
  --vram-mb 120000
```

Workers can start their RPC process automatically:

```bash
go run ./cmd/swarmd join \
  --host HOST_IP \
  --code ROOM_CODE \
  --vram-mb 4096
```

Use `--no-rpc` when you only want to test the control-plane handshake.
