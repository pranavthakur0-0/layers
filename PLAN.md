# Native swarm — implementation plan

A desktop + phone app that **splits one GGUF across devices on the LAN**.
Idea only from SwarmLLM (host + workers, contiguous layers, small activation hop, room code).
**No SwarmLLM source, protocol, kernels, or wire format.**

Speed rule: **do not write an inference engine.** llama.cpp already splits layers over RPC.
We write the **room, process supervisor, and UI.**

---

## 1. Goal

Two or more devices join a room. The strongest device is the **host**. It runs llama.cpp
with `--rpc` workers and `--split-mode layer`. Users chat through a local OpenAI-compatible
endpoint. Phones hold a small layer tail (Phase 4).

**Success (Phase 1):** two laptops on the same Wi‑Fi, one room code, tokens stream,
both GPUs busy, `curl` chat works.

**Success (product):** same thing with a windowed UI and, later, an iPhone worker.

---

## 2. Non-goals (until the LAN swarm works)

- SwarmLLM / WebGPU / PeerJS / any copy of their frames
- Custom GEMV, Mojo, TensorRT, WebRTC
- Cross-internet / NAT / TURN
- Speculative decoding across hops
- Bit-identical output vs SwarmLLM
- Untrusted networks (`ggml-rpc-server` is LAN-only and unsafe on the public internet)

---

## 3. Locked stack

| Piece | Choice |
|---|---|
| Engine | llama.cpp (git submodule, **one pinned commit** on every device) |
| Compute path | `ggml-rpc-server` + `llama-server` `--rpc` `--split-mode layer` |
| Room / CLI | Go (`swarmd`) |
| Chat API | llama-server `/v1/chat/completions` (we proxy) |
| Desktop UI v1 | Static HTML served by `swarmd` on `127.0.0.1:7840` |
| Desktop UI v2 | Tauri loading that URL (optional, after Phase 2) |
| Phone | Phase 4 — Swift/Kotlin wrapping llama.cpp `ggml-rpc-server` |
| LAN compute | llama.cpp RPC over TCP |
| LAN control | Our JSON protocol (HTTP or a single TCP port) |
| Model v1 | Qwen3 1.7B or 4B Q4_K GGUF |
| Model v2 | Qwen 3.8 27B Q4 GGUF (only after Phase 1 works) |

**Do not add** Electron, Flutter-for-compute, or a second engine.

---

## 4. Architecture

```
┌─ Host ──────────────────────────────────────────────────┐
│  swarmd                                                 │
│    control plane: room code, peers, deal, heartbeats    │
│    proc: spawn ggml-rpc-server (local) + llama-server  │
│    ui:   :7840  →  proxy → llama-server :7841           │
│  llama-server                                           │
│    --rpc <worker1>:50052,<worker2>:50052                │
│    --split-mode layer --tensor-split <ratios>           │
└────────────┬────────────────────────────────────────────┘
             │ control  (our JSON)
             │ compute  (llama.cpp RPC — not our code)
┌────────────▼──────────┐  ┌────────────▼──────────┐
│ swarmd join           │  │ swarmd join           │
│ ggml-rpc-server       │  │ ggml-rpc-server       │
│ CUDA / Metal / Vulkan │  │ CUDA / Metal / Vulkan │
└───────────────────────┘  └───────────────────────┘
```

Two planes:

1. **Control (we write)** — membership, VRAM, layer deal, start/stop children.
2. **Compute (we exec)** — ggml moves hidden states between host and `ggml-rpc-server`s.

---

## 5. Ideas we keep vs things we invent

### Keep (idea)

- One host owns the conversation and the sampler.
- Workers hold contiguous layer ranges.
- Strongest device is host; phones get a small tail.
- Human join path: short room code on the LAN.
- First product is capability (model too big for one box), not beating solo llama.cpp.

### Invent (ours)

- Control messages (`hello`, `welcome`, `start`, `ready`, `bye`).
- How we spawn and supervise llama.cpp binaries.
- UI and packaging.

### Do not invent in v1

- Activation packing, f16 frames, MTP rollback, custom kernels.

---

## 6. Control protocol (v1)

Version every handshake. Mismatch → fail loudly.

| Message | Direction | Body |
|---|---|---|
| `hello` | worker → host | `{proto, llama_sha, vram_mb, backend, rpc_port, hostname}` |
| `welcome` | host → worker | `{room, model_sha256, model_name}` |
| `start` | host → worker | `{rpc_bind, rpc_port}` |
| `ready` | worker → host | `{ok, rpc_addr}` |
| `heartbeat` | both, 2s | `{ok}` — miss 3 → drop peer, stop generation |
| `bye` | either | `{}` |

**Deal (v1):** `ratio[i] = vram_mb[i] / sum(vram_mb)`.
Host is always device 0 in `--tensor-split`.
Example: host 16 GB, phone 4 GB → `--tensor-split 4,1`.

**Model identity:** SHA-256 of the GGUF. Workers that cannot see the same file
(or a documented shard) must error before `start`. v1: **each node has the full
GGUF on disk** (simplest). Shard-only downloads are a later optimization.

Default ports:

| Service | Port |
|---|---|
| Control (host) | `7830` |
| UI (host, localhost) | `7840` |
| llama-server (localhost) | `7841` |
| ggml-rpc-server (each worker) | `50052` |

Bind RPC to the LAN IP, **not** `0.0.0.0` on a public interface if you can avoid it.
Never forward these ports through a router.

---

## 7. Repository layout

Create this **outside** the SwarmLLM tree (`~/Desktop/swarm-native`).

```
swarm-native/
  PLAN.md                 this file
  README.md               how to build and run
  cmd/swarmd/main.go      host | join | version
  internal/
    room/                 code gen, peer list
    proto/                JSON types + version
    deal/                 VRAM → tensor-split
    proc/                 start/stop llama binaries
    net/                  listen, dial, heartbeat
    ui/                   embed Phase 2 static files
  ui/index.html           chat + room
  scripts/
    build-llama.sh        CUDA | Metal | Vulkan + GGML_RPC=ON
    phase0-host.sh
    phase0-worker.sh
  llamacpp/llama.cpp      git submodule, pinned
  models/                 GGUF files (gitignored)
  docs/protocol.md        copy of section 6 once stable
```

---

## 8. Phases

### Phase 0 — Engine proof (days 1–2)

No Go room. Two machines, same Wi‑Fi, same llama.cpp commit.

Worker:

```bash
cmake -B build -DGGML_RPC=ON -DGGML_METAL=ON    # or GGML_CUDA=ON / GGML_VULKAN=ON
scripts/build-llama.sh cuda
./build/llama/bin/ggml-rpc-server --host 0.0.0.0 --port 50052
```

Host:

```bash
./build/llama/bin/llama-cli -m models/qwen3-1.7b-q4_k.gguf \
  --rpc 192.168.x.x:50052 \
  --split-mode layer \
  -ngl 99 \
  -p "hello"
```

**Exit criteria**

- [ ] Tokens print on the host
- [ ] Worker GPU/CPU shows load (`Activity Monitor` / `nvidia-smi`)
- [ ] You recorded tok/s vs the same GGUF solo on the host
- [ ] Both binaries report the same `git rev-parse HEAD`

**If this fails, stop. Do not write `swarmd`.**

---

### Phase 1 — `swarmd` (days 3–7)

CLI:

```bash
swarmd host --model ./models/qwen3-1.7b-q4_k.gguf
# prints room code + LAN IP

swarmd join --code ABC123 --host 192.168.x.x
```

Implementation order:

1. `proto` types + `proto` version constant
2. `room` — 6-char code, peer map, mutex
3. `net` — host listen `:7830`, worker dial, heartbeat
4. `proc` — look for `LLAMA_BIN` or `./build/llama/bin`
5. On `start`, worker execs `ggml-rpc-server`
6. When all workers `ready`, host execs `llama-server` with `--rpc` and `--tensor-split`
7. `curl 127.0.0.1:7841/v1/chat/completions`

**Exit criteria**

- [ ] Second laptop joins by code
- [ ] Host starts llama-server without hand-typed IPs
- [ ] Chat via curl
- [ ] Kill worker → host stops cleanly and prints a clear error
- [ ] Wrong `llama_sha` → reject at `hello`

This is a **working native swarm.** UI is optional after this.

---

### Phase 2 — Local chat UI (days 8–10)

`swarmd host` also serves `ui/index.html` on `127.0.0.1:7840`.

Page: room code, peer list, backend/VRAM, chat box.
Chat calls our proxy → llama-server SSE.

**Exit criteria**

- [ ] Type a prompt, stream tokens, no curl
- [ ] Room code visible for the other human
- [ ] Empty / error / busy states shown

Tauri wrapper: only after this page works in a normal browser.

---

### Phase 3 — 27B (days 11–13)

Same binary. Swap GGUF. Tune `--tensor-split` so the big machine holds
embed + head + most layers.

**Exit criteria**

- [ ] Completes a short prompt without OOM
- [ ] tok/s written down (solo vs split)
- [ ] Honest README: split is for “doesn’t fit,” not “faster than one fat GPU”

---

### Phase 4 — Phone worker (weeks 3–5)

Do not start before Phase 1 is green.

- iOS: llama.cpp static lib, Metal, in-process RPC listen on LAN
- App UI: room code, join, stay-awake (screen on for v1)
- Same `hello` / `ready` as the Go worker
- Android: NDK + Vulkan after iOS, or skip

**Exit criteria**

- [ ] iPhone appears in the host peer list
- [ ] A generation uses the phone (Activity / Instruments shows Metal)
- [ ] Background/lock: documented limitation (v1 = screen on)

---

## 9. Week 1 calendar

| Day | Work | Done when |
|---|---|---|
| 1 | Repo, submodule, `build-llama.sh`, Phase 0 | `hello` over RPC |
| 2 | Phase 0 on real second machine + notes | tok/s recorded |
| 3 | `swarmd host` / `join`, handshake | peer list prints |
| 4 | `proc` + `ggml-rpc-server` spawn | worker `ready` |
| 5 | `deal` + `llama-server` spawn | curl chat |
| 6 | Heartbeat, version mismatch, teardown | kill-worker is clean |
| 7 | Buffer / fix Phase 0–1 bugs | Phase 1 checklist complete |

---

## 10. What we write vs what we only exec

| Write | Exec only |
|---|---|
| Room code, join, heartbeat | `ggml-rpc-server` |
| VRAM → `--tensor-split` | `llama-server` / `llama-cli` |
| Child process lifecycle | ggml CUDA / Metal / Vulkan |
| Local UI + API proxy | (later) GGUF download helper |

No `.cu`, no WGSL, no “hidden state packer” in v1.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| RPC version skew | Pin submodule; send `llama_sha` in `hello` |
| `ggml-rpc-server` on WAN | LAN-only docs; bind LAN IP; no UPnP |
| Disk: 27B on every node | v1 accept it; shard fetch is post-Phase 3 |
| iPhone thermal / background | Phase 4; screen-on worker |
| Solo is faster when model fits | Document; do not market “faster” |
| llama.cpp RPC flags change | Pin commit; wrap flags in `proc` only |

---

## 12. First commands (day 1)

```bash
mkdir -p ~/Desktop/swarm-native && cd ~/Desktop/swarm-native
git init
git submodule add https://github.com/ggml-org/llama.cpp.git llamacpp/llama.cpp
# pin: cd llamacpp/llama.cpp && git checkout <sha> && cd ../..
# then implement scripts/build-llama.sh and run Phase 0
```

First code after the submodule: `scripts/build-llama.sh` and `cmd/swarmd/main.go`
(`host` / `join` stubs). Not the UI. Not iOS.

---

## 13. Decision log

| Decision | Choice | Revisit when |
|---|---|---|
| Engine | llama.cpp RPC | Never for v1 |
| Room language | Go | If Tauri needs a Rust core later |
| UI v1 | Local HTML | After Phase 2 works |
| Transport for activations | llama.cpp TCP RPC | Cross-network / custom speculation |
| Phone | Last | Phase 1 green |
| 27B | After small Qwen | Phase 1 green |

---

## 14. Definition of done (v1 product)

- Two native devices, one room code, one GGUF
- Host UI or curl chat
- Same llama.cpp commit on all nodes
- README with build flags, ports, LAN-only warning, measured tok/s
- No SwarmLLM files in the tree
