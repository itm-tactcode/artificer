# SCROLL — How Inference Engines Work (and How **Artificer** Implements Them)

**Artificer** is this repository’s local LLM inference engine (CLI binary: `artificer`).  
Paths below are relative to the repository root.

This document has two jobs:

1. **Part I** — teach what a local LLM **inference engine** is, assuming you are comfortable programming but may never have implemented transformers.  
2. **Part II** — map every important source file in this repo to that mental model.

You do **not** need a machine-learning degree. You do need curiosity about “what happens between my prompt string and the model’s reply.”

---

# Part I — A gentle but serious introduction

## 0. What problem are we solving?

People talk about “ChatGPT” or “running Llama locally” as if the model were a chatbot app. Under the hood there are two very different things:

| Piece | What it is | Who builds it |
|-------|------------|----------------|
| **Model weights** | Huge tables of numbers produced by **training** on text (and sometimes code, images, …) | Labs / open-source projects; we usually **download** them |
| **Inference engine** | Software that **loads** those weights and **runs** them step by step to produce new text | This repository (Artificer), llama.cpp, Ollama’s runtime, vLLM, … |

**Training** is the expensive process that *creates* the weights (weeks of GPU time, trillions of tokens).  
**Inference** is the process that *uses* the weights to answer a prompt (milliseconds to seconds per reply on your machine).

Artificer is **inference only**. We never update the weights. We load a file, accept text, emit text.

### 0.1 Why “engine”?

An engine is the piece that:

1. Understands the on-disk **format** of the model  
2. Turns human text into **token IDs** the model knows  
3. Runs the **math graph** (transformer, etc.) efficiently on your hardware  
4. Turns model outputs back into text  
5. Optionally **serves** that capability over HTTP or agent protocols  

Without an engine, a `.gguf` file is just a large binary blob.

---

## 1. Tokens: the model does not “see” words

### 1.1 Vocabulary

A language model’s true input and output alphabet is a fixed list of **tokens** — short chunks of text (sometimes whole words, sometimes pieces, sometimes single bytes). Each token has an integer **ID**.

Example (illustrative, not a real vocab):

| ID | Token piece (text) |
|----|---------------------|
| 15 | `"What"` |
| 88 | `" color"` |
| 201 | `" is"` |
| 44 | `" the"` |
| 903 | `" sky"` |
| 7 | `"?"` |

The set of all pieces is the **vocabulary**, size often written **`n_vocab`** (e.g. 32 000 or 128 000).  
The model never receives the string `"sky"`; it receives the integer `903`.

### 1.2 Encode and decode

| Direction | Name | Job |
|-----------|------|-----|
| text → IDs | **tokenize / encode** | Split text into pieces that exist in the vocab |
| IDs → text | **detokenize / decode** | Concatenate pieces back into a string |

Common algorithms:

- **BPE (Byte Pair Encoding)** — start from characters/bytes; repeatedly merge frequent pairs using a **merge table** (classic GPT style). GGUF may store `tokenizer.ggml.merges`.  
- **SentencePiece unigram** — each piece has a **score**; dynamic programming chooses a segmentation (classic Llama style). GGUF stores `tokenizer.ggml.tokens` + `scores`.

Artificer implements both:

- **Unigram / SPM-style** (`tokenizer.ggml.model = llama`) — dynamic programming over scored pieces.  
- **GPT-2 BPE** (`tokenizer.ggml.model = gpt2` + `merges`) — pre-split words, GPT-2 byte map, priority-queue merges. Pre-types include classic GPT-2, Llama-3 / **pixtral**, and Qwen2-style digit handling.

### 1.3 Special tokens

Some IDs are not “normal words”:

- **BOS** — beginning of sequence (often prepended so the model knows a new document started)  
- **EOS** — end of sequence (when the model emits this, generation usually **stops**)  
- Padding, unknowns, tool markers, etc.

**Chat roles** (`user`, `assistant`, `system`) are **not** magic hardware features. They are ordinary text arranged by a **chat template** so the model “recognizes” a dialogue format it saw in training.

### 1.4 Context length (`n_ctx`)

The model was designed (and often trained) to attend over at most **`n_ctx`** tokens in one sequence (e.g. 2048, 8192, 128k).  
That number sizes the **KV cache** (below) and is a hard limit for “how long can this conversation + reply be.”

GGUF metadata may advertise huge contexts (even 1M). On a 16 GB machine you often **must cap** usable context so RAM does not explode — Artificer does this automatically.

---

## 2. What the model computes (the forward pass)

### 2.1 The one-sentence definition

A (decoder-only) LLM is a function:

> **Given the token IDs so far, produce a score for every possible next token ID.**

Those scores are called **logits** — a vector of length `n_vocab`. Higher logit ≈ model thinks that token is a better continuation.  
We then **sample** (or greedily pick) one ID, append it, and repeat. That loop is **autoregressive generation**.

### 2.2 Embeddings: IDs become vectors

Computers do linear algebra on real numbers, not on “token 903.”  
The first step is an **embedding table**: a matrix of shape roughly `[n_embd × n_vocab]`. Looking up ID `903` returns a vector of length **`n_embd`** (the **hidden size** / model dimension — e.g. 2048, 4096).

That vector is the model’s internal representation of “this token, in this position’s stream.”

### 2.3 Residual stream (a useful mental picture)

Think of a running vector **`x`** of length `n_embd` as a **scratchpad** that starts as the embedding and is refined by each layer:

```
x ← Embedding(token_id)
for layer in 1..n_layer:
    x ← x + AttentionBlock(x)   # residual connection
    x ← x + FeedForward(x)      # residual connection
logits ← Unembed(Normalize(x))  # back to vocabulary scores
```

**Residual connection** means “add the block’s output to what came in” rather than replace it. That architectural trick makes deep networks trainable and is universal in modern transformers.

### 2.4 One transformer layer (dense)

Most open models Artificer runs today are **dense decoder-only transformers**. One layer, simplified:

```
# Attention half
residual = x
x = Normalize(x)                 # RMSNorm or LayerNorm
Q, K, V = Linear_q(x), Linear_k(x), Linear_v(x)
Q, K = RoPE(Q, K, position)      # inject position information
store K, V into KV cache at this position
attn = Attention(Q, all cached K/V)   # “what to read from the past”
x = residual + Linear_out(attn)

# Feed-forward half
residual = x
x = Normalize(x)
x = residual + FFN(x)            # expands to n_ff, non-linearity, projects back
```

**Vocabulary of dimensions:**

| Symbol | Meaning |
|--------|---------|
| `n_embd` | Width of residual stream / hidden size |
| `n_layer` | How many transformer blocks are stacked |
| `n_head` | Number of attention **query** heads |
| `n_head_kv` | Number of **key/value** heads (may be smaller — GQA) |
| `head_dim` | Size of one head; often `n_embd / n_head` |
| `n_ff` | Inner width of the feed-forward network (often 2.5–4× `n_embd`) |
| `n_rot` | How many dims RoPE rotates (sometimes &lt; head_dim) |

### 2.5 Attention in plain language

At the current position `t`, the model builds:

- **Query (Q)** — “what am I looking for right now?”  
- **Key (K)** — “what does each past position offer as a label?”  
- **Value (V)** — “what content should I copy if I attend there?”

It scores Q against every past K, turns scores into weights (softmax), and mixes the V’s.  
**Causal** attention means position `t` may only look at positions `≤ t` — it cannot cheat by reading the future.

Without a cache, you would recompute K and V for the entire prompt on every new token. That is wasteful.

### 2.6 The KV cache (the decode speed secret)

When generating token by token:

1. **Prefill** — process the whole prompt once; write K and V for every prompt position.  
2. **Decode** — for each new token, compute only that position’s Q/K/V, **append** K/V to the cache, attend over all cached keys.

Memory rough order of magnitude (f32 cache):

```
bytes ≈ n_layer × n_ctx × n_embd_kv × 2 × 4
```

where `n_embd_kv = n_head_kv × head_dim`, and the `× 2` is keys + values.

This is why huge `n_ctx` hurts RAM even if weights are quantized and memory-mapped.

**GQA (grouped-query attention):** fewer KV heads than Q heads. Several queries share one KV head → smaller cache, still good quality. Very common (Llama 3, Qwen2, …).

### 2.7 Feed-forward flavors

After attention, each position’s vector is mixed through a small MLP. Variants:

| Name | Sketch | Examples |
|------|--------|----------|
| SwiGLU | `down(silu(gate(x)) * up(x))` | Llama, Qwen |
| GeGLU | `down(gelu(gate(x)) * up(x))` | Gemma |
| ReLU² sequential | `down(relu(up(x))²)` | classic dense Nemotron |

### 2.8 Normalization and RoPE

- **RMSNorm** — scale features by inverse root-mean-square (no mean subtraction). Default for Llama-like models.  
- **Gemma RMS** — same idea with weight interpreted as `(1 + w)`.  
- **LayerNorm** — subtract mean, divide by std, then scale/bias (classic Nemotron dense).  
- **RoPE (rotary position embedding)** — rotate pairs of Q/K coordinates by an angle that depends on **position** and `rope_freq_base`. This is how the model knows “token 5 vs token 50” without adding a separate position vector in many modern designs.

### 2.9 Architectures beyond “one big stack of identical layers”

Artificer today targets **dense** transformers (every layer is attention + FFN).

Other families you will meet in GGUF land:

- **MoE (mixture of experts)** — FFN replaced by many experts; a router picks a few per token. Much more code.  
- **Hybrid Mamba / SSM** (`nemotron_h`) — many layers are **state-space** blocks (constant-time steps, conv + SSM cache), with only a few attention layers and interleaved FFNs. **Supported** in Artificer: see `ssm.rs` and hybrid `LayerType`.  
- **MLA**, multimodal towers, MoE hybrids — further graphs.

The engine’s job is to **dispatch** on `general.architecture` in GGUF and run the matching graph.

---

## 3. Quantization: why files say `Q4_K_M`

### 3.1 The bandwidth problem

A 7B model in float16 is roughly `7e9 × 2 bytes ≈ 14 GB` of weights alone.  
On CPU, generating one token often requires **touching a large fraction of those weights** (the big linear layers). You are limited less by “FLOPs” and more by **how fast RAM can stream bytes** into the CPU.

### 3.2 What quantization does

Store each weight as a small integer plus a **scale** (and sometimes a min) per block:

```
approximate_weight ≈ scale * quantized_int - min
```

Types you will see in GGUF / GGML:

| Type | Idea |
|------|------|
| Q4_0 / Q5_0 / Q8_0 | Classic small blocks (32 values) |
| Q4_K / Q5_K / Q6_K | “K-quants” — larger 256-value super-blocks, better quality/size tradeoff |
| F16 / BF16 / F32 | Little or no compression |

A filename like `…-Q4_K_M.gguf` usually means a **mix**: most tensors Q4_K, some important ones higher precision (e.g. Q6_K).

### 3.3 Fused dequant-dot

Naive approach:

1. Dequantize a whole weight matrix row to f32  
2. Dot with the activation vector  

Better approach (what Artificer does on the hot path):

1. Read quant block  
2. Multiply/add into the accumulator **without** writing a full f32 row to memory  

That is a **fused** kernel. Artificer adds **AVX2** paths so those inner loops use 8-wide SIMD when the CPU supports it.

### 3.4 mmap

Weights live in a file. **Memory-mapping** (`mmap`) lets the OS page bytes in on demand into the process address space instead of `read()`-ing multi-GB into a giant heap allocation up front. Artificer’s GGUF loader defaults to mmap.

---

## 4. The generation loop (runtime)

```
user text
  → optional chat template (roles → one string the model expects)
  → encode → list of token IDs (the prompt)
  → PREFILL: for pos, token in enumerate(prompt):
        forward(token, pos)   # fills KV cache
  → LOOP up to max_tokens:
        logits = last forward’s vocabulary scores
        next_id = sample(logits)   # greedy or temperature / top-k / top-p
        if next_id is EOS or stop rule: break
        decode piece; stream to user
        forward(next_id, pos+1)
  → done
```

### 4.1 Prefill vs decode

| Phase | Work | Feels like |
|-------|------|------------|
| Prefill | One forward per prompt token | Time-to-first-token; long prompts hurt |
| Decode | One forward per new token | Tokens per second you quote as “speed” |

### 4.2 Sampling

Logits are not probabilities until you apply softmax (sometimes after temperature).

- **Temperature = 0** — greedy: always argmax (deterministic).  
- **Temperature &gt; 0** — soften/sharpen the distribution, then sample.  
- **Top-k** — keep only the k largest logits, then sample.  
- **Top-p (nucleus)** — keep the smallest set of tokens whose cumulative probability ≥ p.

**Stop conditions:** EOS token, max new tokens, max context, or **stop strings** in decoded text (e.g. stop if the model starts hallucinating the next `USER:` turn).

---

## 5. GGUF: one file to load them all

**GGUF** (GGML Unified Format) packages:

1. Header (magic `GGUF`, version)  
2. Metadata key/values (architecture name, sizes, tokenizer, rope base, …)  
3. Tensor directory (name, shape, type, offset)  
4. Raw weight bytes (aligned)

The string `general.architecture` (e.g. `llama`, `qwen2`, `nemotron`, `nemotron_h`) selects which metadata keys and which **compute graph** to use.

---

## 6. Serving: OpenAI API, tools, MCP

Once you can generate, product usefulness comes from **protocols**:

### 6.1 OpenAI-compatible HTTP

`POST /v1/chat/completions` with JSON messages is the de-facto integration surface for apps, agents, and frontends.  
Streaming uses **SSE** (`text/event-stream`) with per-token `delta` chunks.

### 6.2 Tool calling

The model does not magically run Python. The protocol is:

1. Client sends **tool definitions** (JSON schemas).  
2. Engine injects instructions + may parse model text into structured **`tool_calls`**.  
3. **Client** executes the tool and returns `role: "tool"` results.  
4. Model continues.

### 6.3 MCP (Model Context Protocol)

A JSON-RPC way for **agent hosts** to discover and call tools. Artificer exposes MCP so a host can drive the **local model** (`chat`, `complete`) over stdio or `POST /mcp`.

---

## 7. Performance intuition (CPU, ~16 GB machines)

1. **Decode tok/s** ≈ limited by weight bandwidth and matmul kernels.  
2. **Quantization + fused SIMD** are the main levers on CPU; **CPU speed tuning is not a current focus** (see §9).  
3. **KV cache** grows with context; auto-cap context and optional **KV f16** trade quality/RAM.  
4. **Threads** help when parallelizing independent output rows of a matmul.  
5. Measure with `artificer bench` (tok/s, peak RSS).  
6. **GPU** is optional (`--features cuda`, `--backend cuda`); CPU remains the supported default without a CUDA toolchain.

---

## 8. End-to-end picture in Artificer

```
┌──────────────────┐     ┌────────────────┐     ┌─────────────────┐
│ CLI / HTTP / MCP │────▶│ Runtime Engine │────▶│ Core forward    │
│ prompts & APIs   │     │ prefill+decode │     │ + Backend matmul│
└──────────────────┘     └───────┬────────┘     └────────▲────────┘
                                 │                       │ mmap weights
                         ┌───────▼────────┐       ┌──────┴───────┐
                         │ Tokenizer +    │       │ GGUF loader  │
                         │ chat template  │       │ metadata+tens│
                         └────────────────┘       └──────────────┘
```

| Crate | Responsibility |
|-------|----------------|
| `gguf` | Parse/mmap GGUF |
| `artificer-core` | Config, tokenizer, quant+SIMD, transformer, KV, sampling, optional CUDA `Backend` |
| `artificer-runtime` | Prefill/decode loop, load options (ctx budget, KV f16) |
| `artificer-server` | OpenAI HTTP + tools + MCP + metrics |
| `artificer-cli` | Binary `artificer`: inspect / run / bench / serve / mcp |

---

## 9. What this engine is not (yet) — honesty walls

These are the engine's current limits. GPT-2 BPE (§1.2) and hybrid Mamba (`nemotron_h`, §2.9) are already in the tree.

| Wall | Status |
|------|--------|
| **CPU default** | Always works. Default builds need **no** CUDA toolkit. `cargo test` / `artificer run` without `--features cuda` stay green. |
| **GPU (CUDA)** | **Optional** (`--features cuda`, `--backend cuda` / `ARTIFICER_BACKEND`). Runtime `dlopen` only (no build-time CUDA): `libcudart`/`libcublas` (GEMM/F32 GEMV), **`libnvrtc.so.13`** (compiles `backend/cuda_kernels.cu` to PTX for the device's `compute_XY` at `init_device`) and the driver **`libcuda.so.1`** (`cuModuleLoadData` / `cuLaunchKernel`). **Quantized-resident weights (2026-10-16):** GGUF blocks (Q4_0/1, Q5_0/1, Q8_0, Q2_K–Q6_K, F16/BF16) are uploaded **as-is** and dequantized **inside** the matvec kernel (one warp per output row, 32-element chunks), so VRAM ≈ GGUF size; F32 tensors keep cuBLAS Sgemv. **Attention:** QKᵀ + AV via cuBLAS on device pointers with the **stable max-subtract softmax (+scale/softcap) kernel on device** between them; only the KV gather stays on host. Free memory is checked (`cudaMemGetInfo`, 384 MiB headroom) before upload; any CUDA/NVRTC/OOM failure is `CoreError::Backend` — never a silent CPU fallback. CPU tokens may differ late in long greedy runs because the CPU path quantizes activations to Q8_K while the GPU uses f32 activations. CPU speed tuning is not a current focus. |
| **CPU speed** | **Not a current focus** — no blind CPU opts / governor changes. Historical speed figures are not live targets. |
| **MoE** | **Experimental, CPU only, behind `--experimental-moe`.** OLMoE (`olmoe`) and Qwen1.5-MoE (`qwen2moe`) loader + top-k router (`forward/moe.rs`; softmax over all experts, top-k, no renormalisation). `qwen2moe` adds a shared expert per layer: a dense SwiGLU FFN (`ffn_{gate,up,down}_shexp`) scaled by `sigmoid(ffn_gate_inp_shexp · x)` and added to the routed sum. Refused at load without the flag (per-family `MOE_FAMILIES` table in `config/mod.rs`, all experimental for now); CUDA refuses MoE with or without it. Other MoE arches (`qwen3moe`, `nemotron_h_moe`, `deepseek2`, …) rejected. |
| Other gaps | Multi-model registry, continuous batching, MLA, IQ quants — still later. |

No public ship / pricing / waitlist framing in this repo. Local-only inference engine.

The **skeleton** is already the same pipeline Ollama-class systems implement.

---

# Part II — Code map (every Rust file in use)

Paths are relative to the repo root. Each note ties the file to **Part I**.

---

## Workspace root

### `Cargo.toml`

Workspace membership (`gguf`, `core`, `runtime`, `server`, `cli`), shared dependency versions (e.g. `memmap2`, `axum`, `minijinja`), and release profile flags (`lto`, `opt-level`). This is not inference logic—it defines how the crates that *do* inference are built and linked.

---

## Crate `gguf` — loading the model package (§6)

### `crates/gguf/src/lib.rs`

Public surface of the loader crate: re-exports `GgufModel`, types, metadata helpers. This is the entry “module index” for everything in §6 (open GGUF → usable metadata + tensor bytes).

### `crates/gguf/src/types.rs`

Core GGUF/GGML vocabulary of types: magic constants, `TensorType` (F32, Q4_K, Q6_K, …), metadata value enums, `TensorInfo`, alignment helpers, and **block/type sizes** used to compute tensor byte lengths. This is the schema language of the container described in §4 and §6.

### `crates/gguf/src/error.rs`

Typed errors for bad magic, unsupported version, OOB tensors, UTF-8 failures, etc. Keeps load failures explicit (strict `Result` path for corrupt or truncated files).

### `crates/gguf/src/reader.rs`

Little-endian binary cursor over a byte slice: read `u32`/`u64`/strings/arrays/metadata values. Implements the low-level walk of the GGUF header and KV stream (§6 structure).

### `crates/gguf/src/metadata.rs`

In-memory map of metadata keys with typed getters (`architecture`, `context_length`, tokenizer specials, …). This is how §2 (tokenizer ids) and §3 (hyperparameters) are discovered without hardcoding every file layout.

### `crates/gguf/src/model.rs`

**Main loader:** open file → mmap or buffer → parse header/metadata/tensor infos → validate tensor bounds → expose `tensor_data(name)` zero-copy slices and `summary()`. This is §6 end-to-end: from path on disk to weight bytes ready for §3/§4.

### `crates/gguf/src/tests.rs`

Synthetic GGUF writer + unit tests (magic, mmap, arrays). Ensures the container parser stays correct without multi‑GB fixtures.

---

## Crate `artificer-core` — brain of inference (§2–§5)

### `crates/core/src/lib.rs`

Crate root: modules and public re-exports (`LlamaModel`, tokenizer, sampling, chat templates, …). External crates (`runtime`, `cli`) depend on this API as “the model that can forward and sample.”

### `crates/core/src/error.rs`

Core-level errors (missing tensors, bad metadata, unsupported architecture, context exceeded). Separates neural/runtime failures from pure GGUF parse errors.

### `crates/core/src/config/arch.rs`

Architecture classification (`Llama`, `Qwen`, `Gemma`, `DeepSeek`, `Nemotron`) and graph flavors: FFN activation type, norm kind, whether embeddings are scaled. This is the switchboard for §3.3–§3.4 multi-arch dense graphs.

### `crates/core/src/config/mod.rs`

`ModelConfig`: reads GGUF metadata into concrete hyperparameters (`n_embd`, `n_layer`, `n_head`, `n_head_kv`, rope base, softcaps, BOS/EOS, …). Rejects MoE DeepSeek for now. Instantiates the numbers that size every buffer in §3 and §5.

### `crates/core/src/tokenizer/` (mod, bpe, gpt2_bytes, pre, utf8_stream)

§2 implementation:

- **`mod.rs`** — `Tokenizer` facade: load GGUF vocab; dispatch **Unigram** vs **BPE**; BOS/EOS; chat_template; streaming decode.  
- **`bpe.rs`** — GPT-2 BPE merge engine (ranked bigrams).  
- **`gpt2_bytes.rs`** — byte ↔ unicode map (Ġ for space, …).  
- **`pre.rs`** — pre-tokenizers (gpt2 / llama3-pixtral / qwen2).  
- **`utf8_stream.rs`** — `Utf8StreamDecoder`: incremental UTF-8 for streamed detokenization. One character can span tokens (`<0xE4><0xBD><0xA0>` = 你, byte-level BPE pieces), so streaming feeds each token's raw bytes (`Tokenizer::decode_token_bytes`) and gets back only complete characters; an incomplete tail is held until the next token, and flushed lossily (U+FFFD) at end of stream. Streamed text then equals a one-shot `decode`; per-token `decode_token` alone would print one U+FFFD per partial piece.  

Without this crate area, weights alone cannot talk text.

### `crates/core/src/chat_template.rs`

§2 / §5 prompt shaping: MiniJinja render of GGUF `tokenizer.chat_template` when present; otherwise Vicuna-style USER/ASSISTANT fallback. Turns chat messages into the raw string that encode() will tokenize—critical for instruction quality.

### `crates/core/src/model.rs`

Binds a loaded `GgufModel` to named weights: `token_embd`, per-block `attn_*` / `ffn_*`, `output_norm`, `output`, optional biases. This is the bridge from §6 tensor names to the §3 graph structure.

### `crates/core/src/backend/` (mod, cuda, cuda_quant, cuda_kernels.cu, cuda_attention)

Pluggable compute backend (§7 / §9):

- **`mod.rs`** — `Backend` trait (`matmul_vec`, optional `as_cuda`), `BackendKind`, `CpuBackend`, `create_backend` / `resolve_backend_kind` / `ARTIFICER_BACKEND`. CPU default.  
- **`cuda.rs`** (`--features cuda`): `dlopen` libcudart + libcublas; device GEMV/GEMM; weight cache (raw quantized blocks, f32 only for F32 tensors); `check_fits` VRAM preflight; counters `gemv/gemm/quant_matvec/softmax_launch_count`, `resident_weight_bytes`; **no silent host fallback**. Batched-prefill matmul path switch `ARTIFICER_PREFILL_MATMUL=old|mmq` (`prefill_matmul_mode()`; unset/unknown follow the single constant `PREFILL_MATMUL_DEFAULT`, currently `mmq`): `old` = dequant-to-f16 tiles + cuBLAS GemmEx, `mmq` = int8 MMQ kernels `mmq_<type>` on Q8 activations for types listed in `mmq_kernel_name` (others stay `old`); counter `mmq_launch_count`.  
- **`cuda_quant.rs`** — NVRTC compile + driver-API load/launch of the custom kernels; `device_quant_layout`, `DEVICE_QUANT_TYPES`; per-type device-vs-host-dequant tests + softmax test.  
- **`cuda_kernels.cu`** — CUDA C source (embedded via `include_str!`): `mv_<quant>` matvec kernels that dequantize GGUF blocks in-register, `mv_f16` (f16/bf16), `softmax_rows`.  
- **`cuda_attention.rs`** — decode attention per head via `CudaBackend::attn_head_device` (cuBLAS QKᵀ → device softmax kernel → cuBLAS AV); host KV pack only; wired from `forward/layer.rs` only when `backend.as_cuda()` is `Some`.

Forward / SSM call `Backend::matmul_vec` (not a hard-coded CPU-only path when kind is Cuda).

### `crates/core/src/profile.rs`

Optional decode-step wall timers. Enable with env `ARTIFICER_PROFILE=1`. Buckets cover matmul quant types, weight roles, SSM/attention non-matmul, norms, and embd. Prefill is muted in the runtime so dumps reflect decode. `artificer bench` resets after warmup and dumps at the end.

### `crates/core/src/kv.rs`

§2.6 KV cache: allocate per-layer K/V slabs; write row at `pos`; read for attention. Supports **f32** (default) and **f16** storage (`KvElem`) to cut RAM ~in half. Enables efficient decode in §4.


### `crates/core/src/quant/q8k.rs` / `pool.rs` (2026-09-29)

- **q8k.rs** — ggml-compatible Q8_K activation quantization and integer `vec_dot_q4_K_q8_K` (scalar / AVX2 / AVX-512 VNNI). Quantize activations once per matvec; weight rows stay Q4_K.
- **pool.rs** — persistent worker threads for Q4_K×Q8_K row chunks (avoids `thread::scope` spawn churn on every projection).
- Wired from `matmul.rs` when `TensorType::Q4_K` and `k % 256 == 0`. Feature flags from `simd.rs` (`avx512_vnni_enabled`, etc.).

### `crates/core/src/ssm.rs`

Mamba-2 single-token step + `SsmCache`. Static SSM tensors (conv/A/D/dt_b/norm) are **pre-dequantized once** at engine init (`SsmCache::from_model`) so decode does not alloc/dequant every token.

### `crates/core/src/sysmem.rs`

§1.4 / §7 memory: detect host RAM (`/proc/meminfo`), estimate KV bytes, clamp context, report process RSS / peak RSS for benches.

### `crates/core/src/perplexity.rs`

Model-quality measurement (not part of generation). Reproduces llama.cpp `llama-perplexity`'s default algorithm at 8642541b: tokenize the whole file once (BOS per GGUF flag, specials as plain text), split into `n_ctx` chunks (`--chunks` caps the count), replace each chunk's first token with BOS when the model adds one, evaluate each chunk from an empty KV cache and score only the second half (logits at positions `n_ctx/2 ..= n_ctx-2` against the next token, `n_ctx/2 - 1` tokens per chunk for even `n_ctx`). Log-softmax with an f32 max and an f64 sum; final `PPL = exp(mean nll)` ± `sqrt(var/(N-1))·PPL`. Pure helpers (`PplPlan`, `chunk_input`, `token_nll`, `PplAccum`) are unit-tested; `run_perplexity` drives `prefill` + `forward`. CLI: `artificer perplexity`.

### `crates/core/src/ops.rs`

Elementwise primitives of §3: RMSNorm / Gemma RMS / LayerNorm, SiLU, GELU, ReLU², softmax, RoPE, softcap, residual add/mul. The “small ops” between big matmuls.

### `crates/core/src/quant/mod.rs`

Module index for quantization (§3): re-exports `dequantize`, `matmul_vec`, half helpers.

### `crates/core/src/quant/half.rs`

f16 ↔ f32 conversion used by quant scales, F16 tensors, and optional KV f16 storage (§3 / §2.6).

### `crates/core/src/quant/blocks.rs`

Block sizes and Q4_K / Q5_K scale packing helpers—layout constants for GGML blocks (§3).

### `crates/core/src/quant/dequant.rs`

Expand quantized (or float) weight segments into f32: F32/F16/BF16, Q4_0, **Q5_0**, Q8_0, Q4_K, **Q5_K**, Q6_K (§3).

### `crates/core/src/quant/simd.rs`

Optional **AVX2/FMA** helpers (runtime-detected) for wide f32 dots and **Q4_K** nibble·x sums used by fused matmul (§7).

### `crates/core/src/quant/matmul.rs`

Hot path of §2: `matmul_vec` = multi-threaded matrix×vector (up to 16 threads) with **fused** dequant-dot for Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K/F32; Q4_K uses AVX2/FMA when available. Almost every Linear layer spends most of its time here.

### `crates/core/src/quant/q50.rs`

Q5_0×Q8_0 and Q8_0×Q8_0 integer vec_dot (scalar, AVX2, AVX-512 VNNI). Activations are quantized to an internal `BlockQ80` (f32 scale); GGUF weights stay packed Q5_0 / Q8_0.

### `crates/core/src/quant/q50_repack.rs`

Q5_0×4 weight interleave for decode GEMV: groups of four rows packed so the same K-block is contiguous instead of striding one row at a time.

### `crates/core/src/quant/q4k_repack.rs`

Q4_K×8 nibble-stripe repack (1152-byte `block_q4_Kx8`) for an optional vendor GEMV. Off unless `ARTIFICER_CPU_REPACK=q4k`.

### `crates/core/src/quant/q6k.rs`

Q6_K×Q8_K integer vec_dot (scalar / AVX2) for mid-size Q6_K projections. Large Q6 language-model heads stay on the fused f32 path in `matmul.rs`.

### `crates/core/src/quant/ggml_ffi.rs`

Optional `dlopen` of vendor ggml CPU vec_dot / GEMV symbols (`ARTIFICER_GGML_FFI`; `ARTIFICER_CPU_REPACK` selects the Q4_K×8 gemv). Default build does not link ggml. Library search (only when one of those env vars asks for it): `ARTIFICER_GGML_LIB_DIR` (a directory holding `libggml-base.so.0` + `libggml-cpu-zen4.so`, e.g. an Ollama install's `lib/ollama`) first, then the system-wide `/usr/lib/ollama`; if neither loads it warns once on stderr and keeps the native kernels.

### `crates/core/src/forward/mod.rs`

Orchestrates one token’s forward pass: embed → layers → final norm → optional LM head (`compute_logits`; prefill skips vocab matmul until the last prompt token). Dense: attention+FFN; hybrid: **Ssm / Attention / Ffn**. `forward(..., ssm, scratch, compute_logits, backend)` — matmuls go through the pluggable `Backend`.

### `crates/core/src/forward/prefill.rs`

Whole-prompt prefill: `LlamaModel::prefill` runs CUDA batched prefill (`backend/cuda_forward.rs::prefill_batched`, GEMM chunks of up to 1024 prompt rows) when supported, otherwise one `forward` per token with logits only for the last. `prefill_cancellable` takes an optional Stop check (`PrefillCancel`): it is polled before the first chunk and between chunks (after a stream sync on CUDA, so the host can't run ahead of the GPU; before every token on the token-by-token path) and returns `PrefillStatus::Cancelled { processed }` with the cache covering whole chunks only. Without a check the path is exactly `prefill`'s, and an untriggered check gives bit-identical logits.

### `crates/core/src/forward/scratch.rs`

Reusable f32 workspaces (`x`, `q`, `k`, `v`, `att`, FFN temps, `logits`) so each decode step doesn’t thrash the allocator (§5 performance hygiene).

### `crates/core/src/forward/weights.rs`

Helpers to load norm vectors, apply bias, and call **`Backend::matmul_vec`** (via `matmul_wt`) against named weight tensors. Glue between `model.rs` bindings and the active CPU/CUDA backend.

### `crates/core/src/forward/layer.rs`

Decode attention uses a **time-outer** loop so f16 KV rows are dequantized once per position (GQA-safe), with SIMD Q·K (`dot_f32_fast`) and V accumulate (`axpy_f32_fast`).

One transformer block: attention path (projections, RoPE, GQA attention over KV cache, output proj) + FFN path (SiLU/GeGLU/ReLU²). Body of the loop in §3.

### `crates/core/src/sample.rs`

§5.2 sampling: greedy argmax; temperature + top-k + top-p; small PRNG. Converts logits into the next token id.

---

## Crate `artificer-runtime` — the generation loop (§5)

### `crates/runtime/src/lib.rs`

Owns `Engine` (model + scratch + KV cache) and `EngineLoadOptions` (auto context budget, max ctx, KV f16). Implements:

- prefill over prompt tokens (`prefill_cancellable`; the `is_cancelled` check of `generate_tokens_cancellable` is polled between prefill chunks too, so a Stop during a long prompt returns `FinishReason::Cancelled` with no tokens within one chunk)  
- decode loop with sampling  
- stop on EOS / stop strings / max tokens  
- `generate_with_callback` for streaming (`StreamEvent::Token`); pieces go through one `Utf8StreamDecoder` per generation, so a token that ends mid-character streams nothing until the character is complete (its bytes are flushed lossily at the end only if still incomplete). `GenerateOutput::text` is built from the same pieces, so CLI `run`, server SSE / non-stream and MCP all get whole characters.  

This is the executable form of §4; CLI and server call into it rather than reimplementing the loop.

---

## Crate `artificer-server` — serving (§7)

### `crates/server/src/lib.rs`

Boots Tokio/Axum: load model, bind host/port, install body limits, optional CORS, route table (`/health`, `/v1/*`, `/mcp`). Also exports stdio MCP entry (`serve_mcp_stdio_blocking`). Process entry for `artificer serve` / `artificer mcp` (§7).

### `crates/server/src/config.rs`

Bind address, model path/id, `RequestLimits` (max tokens, body size, message counts)—operational knobs and DoS guards around §7. Also the security settings (`allowed_hosts`, `allowed_origins`, `api_key`, `allow_insecure_bind`) and the non-loopback bind refusal (`check_bind_security`).

### `crates/server/src/security.rs`

Middleware in front of every route, outermost first: `Host` / `Origin` allowlist against the port actually bound (403; guards against DNS rebinding and cross-site pages), optional `Authorization: Bearer` API key with a constant-time compare (401; `GET /health` stays open), and a JSON `Content-Type` on every POST (415), which forces a browser CORS preflight.

### `crates/server/src/error.rs`

HTTP-facing error type and JSON error responses for bad requests / failures.

### `crates/server/src/api_types.rs`

Serde structs for OpenAI-compatible chat requests/responses: messages, `stop`, **`tools` / `tool_choice` / `tool_calls`**, legacy `functions` / `function_call`. Defines the wire format of §7.

### `crates/server/src/tools.rs`

Tool-calling core (§7.1): validate tool defs, resolve `tool_choice`, inject Hermes-style system instructions, parse model text into OpenAI `tool_calls`, format tool results / history for prompts. Caps tool count and schema size.

### `crates/server/src/prompt.rs`

Adapts API messages (including tool history) to `artificer_core::render_chat` (templates from §2/§5); prepends tools system message when active.

### `crates/server/src/routes.rs`

Handlers: `/health`, `/v1/models`, `/v1/chat/completions`. Non-stream path returns full JSON (with optional `tool_calls`); stream path runs generation on a blocking thread and emits SSE token deltas—or buffered tool_calls when tools are active (§5 + §7). Serializes access with a mutex (single active generation). `POST /v1/models/load` answers one message for a missing / non-file / non-`.gguf` path and a generic message for damaged files, passing through only the engine's own refusals (MoE gate, CUDA scope, unsupported architecture, named only when it is a known one); the full error goes to the log. `POST /v1/chat/cancel` and client hang-ups set the running completion's cancel flag: it stops within one token while decoding, or within one prefill chunk while the prompt is being prefilled (final chunk `finish_reason: "user"`, zero tokens).

### `crates/server/src/mcp.rs`

MCP JSON-RPC 2.0 (§7.2): `initialize`, `ping`, `tools/list`, `tools/call` (`chat` / `complete`), `prompts/list` / `prompts/get`, empty `resources/list`. Stdio loop (newline-delimited) and HTTP `POST /mcp` adapter over the same engine mutex. Over HTTP, `chat` / `complete` register with the same cancel flag as chat completions (`POST /v1/chat/cancel` by JSON-RPC id or with no id, and client hang-ups), and both tools apply `max_prompt_chars` (over all `chat` messages combined) and the `max_tokens` cap.

### `crates/server/src/metrics.rs`

Counters: requests started/completed/failed, prompt/completion tokens, in-flight and peak concurrency. Visibility into §7 under load.

---

## Crate `artificer-cli` — operator interface

### `crates/cli/src/main.rs`

Binary **`artificer`** (the Artificer app):

- **`inspect`** — dump GGUF summary/metadata/tensors (§5 debugging)  
- **`run`** — load model, apply chat template (unless `--raw`), generate with sampling/stop/`--ctx`/`--kv-f16` (§4)  
- **`bench`** — timed generate + tok/s + peak RSS (§7 measurement)  
- **`serve`** — start the HTTP server + MCP HTTP endpoint (§6)  
- **`mcp`** — MCP server over stdio for agent hosts (§6.3)  
- **`perplexity`** — perplexity of a text file, same method as llama.cpp `llama-perplexity` default mode (`-f`, `-c`, `--chunks`, `-t`); see `crates/core/src/perplexity.rs`  
- **`--experimental-moe`** (`run` / `bench` / `serve` / `mcp` / `perplexity`) — load an MoE family whose status is experimental (one-line warning); without it such a model is refused with an error naming the flag  

Tracing always goes to **stderr** so stdout stays free for MCP protocol.

---

## Mental model: “where do I look if…?”

| Question | Start here |
|----------|------------|
| File won’t load / wrong magic | `gguf` reader + model |
| Wrong architecture / dims | `core/config` |
| Gibberish detokenization | `core/tokenizer/` (unigram vs BPE) |
| Bad chat behavior / loops | `core/chat_template`, stop strings in `runtime` / `server` |
| Slow tok/s | `core/quant/matmul`, `quant/simd`, forward layers |
| RAM / context too large | `core/sysmem`, `EngineLoadOptions`, `--ctx` / `--kv-f16` |
| Wrong attention / GQA | `core/forward/layer`, `kv` |
| Sampling too random/deterministic | `core/sample`, CLI/API temperature |
| API / SSE issues | `server/routes` |
| Tool calling / parse | `server/tools`, `server/prompt` |
| MCP JSON-RPC | `server/mcp` |
| Benchmarks | `artificer bench`, `sysmem` RSS helpers |
| CLI flags | `cli/src/main.rs` |

---

## Closing

An inference engine is not one algorithm—it is a **pipeline**:

**package (GGUF) → tokens → transformer steps with KV cache → sample → stream/serve.**

This repo implements each stage in small crates and modules so you can read Part I once, then open any file in Part II and know which stage of the pipeline you are standing in.
