# LLM Inference Engine

A from-scratch Large Language Model inference engine built in C++20 with zero external dependencies. This project bypasses frameworks like PyTorch and llama.cpp: every matrix multiply, attention head, and dequantization kernel is implemented from the ground up.

Runs TinyLlama 1.1B at **29.4 tok/s** on an M2 MacBook Air, a **49x speedup** over the naive baseline through ARM NEON SIMD, multi-threaded matmul, and weight quantization.

## Demo: Naive vs Optimized Inference
Side-by-side comparison of the slowest configuration (F16, naive backend, 1 thread) 
against the fastest (Q4_0, NEON SIMD, 4 threads), demonstrating the 49x speedup 
in real-time token generation.

https://github.com/user-attachments/assets/701615f4-ca66-4c5c-9a79-d64db419a4d4



https://github.com/user-attachments/assets/fd1b9500-1410-4f6e-8a7b-a4936691816f




## Why It Matters

Every LLM query costs compute time, memory, and energy. As companies deploy billions of inference requests per day, the difference between a naive implementation and an optimized one translates directly to hardware costs and user-facing latency. This project demonstrates those optimizations from first principles by using SIMD to accelerate math, multi-threading to parallelize work, and quantization to shrink the model. Together, these techniques achieve a 49x speedup on commodity hardware.

The gap between calling `model.generate()` and understanding what actually happens inside an LLM is enormous. This project opens that black box, combining disciplines usually taught separately:

| Domain | What's Involved |
|--------|-----------------|
| **Systems Programming** | Memory mapping, binary parsing, RAII, multi-threading |
| **Computer Architecture** | Memory hierarchy, SIMD vectorization, cache behavior |
| **Linear Algebra** | Matrix-vector multiplication, FP16/FP32 precision |
| **Deep Learning** | Transformers, attention, positional encodings |
| **Optimization** | Quantization, KV caching, thread scaling |

## Performance

All benchmarks measured on Apple M2 MacBook Air (8GB RAM) with a custom benchmarking harness using median of multiple trials with warmup isolation.

### Peak Configuration

| Metric | Value |
|--------|-------|
| Best throughput | **29.4 tok/s** (Q4_0, NEON, 4 threads) |
| Best SIMD speedup | **23.2x** (F16, NEON vs naive) |
| Best thread scaling | **4.51x** at 8 threads (F16 naive) |
| Best compression | **3.4x** (F16 → Q4_0, 2.2 GB → 637 MB) |
| Combined speedup | **49x** (F16 naive 1T → Q4_0 NEON 4T) |

### NEON SIMD Speedup (1 thread)

| Format | Naive (ms/tok) | NEON (ms/tok) | Speedup |
|--------|---------------|---------------|---------|
| F16    | 1,697         | 73            | **23.2x** |
| Q4_0   | 568           | 97            | **5.9x** |
| Q8_0   | 300           | 124           | **2.4x** |

### Thread Scaling (F16 naive)

| Threads | ms/tok | Speedup | Efficiency |
|---------|--------|---------|------------|
| 1       | 1,697  | —       | —          |
| 2       | 903    | 1.88x   | 94%        |
| 4       | 476    | 3.57x   | 89%        |
| 8       | 376    | 4.51x   | 56%        |

### Quantization Comparison (NEON, 4 threads)

| Format | Model Size | ms/tok | tok/s |
|--------|-----------|--------|-------|
| F16    | 2.2 GB    | 39     | 25.6  |
| Q8_0   | 1.2 GB    | 41     | 24.3  |
| Q4_0   | 637 MB    | 34     | **29.4** |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GGUF Model File                          │
│  ┌─────────┐  ┌──────────────┐  ┌─────────────────────────────┐ │
│  │ Header  │  │   Metadata   │  │        Tensor Data          │ │
│  │ (magic, │  │ (config,     │  │  (F16 / Q8_0 / Q4_0 /       │ │
│  │ version)│  │  vocab)      │  │   Q6_K weights)             │ │
│  └─────────┘  └──────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
        │                │                       │
        ▼                ▼                       ▼
   ┌─────────┐    ┌─────────────┐        ┌─────────────┐
   │ Parser  │    │  Tokenizer  │        │   Tensors   │
   │         │    │  (BPE, 32K) │        │  (mmap'd)   │
   └─────────┘    └─────────────┘        └─────────────┘
        │                │                       │
        └────────────────┼───────────────────────┘
                         ▼
                 ┌───────────────┐
                 │   GGUFModel   │
                 └───────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │         Forward Pass          │
         │  ┌─────────┐  ┌───────────┐   │
         │  │ RMSNorm │  │ RoPE      │   │
         │  │         │  │           │   │
         │  └─────────┘  └───────────┘   │
         │  ┌─────────────────────────┐  │
         │  │  Grouped Query Attention│  │
         │  │  (with KV Cache)        │  │
         │  └─────────────────────────┘  │
         │  ┌─────────────────────────┐  │
         │  │  SwiGLU FFN             │  │
         │  └─────────────────────────┘  │
         │         × 22 layers           │
         └───────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │          Generation           │
         │  Sampling: argmax / temp /    │
         │  top-p nucleus                │
         └───────────────────────────────┘
```

## Features

### Model Loading
- Memory-mapped file I/O (`mmap`) for zero-copy weight access
- Complete GGUF header, metadata, and tensor parsing
- Support for F16, F32, Q8_0, Q4_0, and Q6_K weight formats

### Transformer Implementation
- Full Llama architecture across 22 transformer layers
- Grouped Query Attention (GQA) with 32 query heads and 4 key-value heads
- Rotary Positional Encoding (RoPE)
- RMSNorm pre-normalization
- SwiGLU feed-forward network
- KV cache for autoregressive generation

### Tokenizer
- BPE tokenization with greedy longest-match algorithm
- SentencePiece-compatible space handling (▁ character)
- 32,000 token vocabulary loaded from GGUF metadata
- Encode/decode with round-trip verification

### NEON SIMD Optimization
- Vectorized matmul for F16, Q8_0, Q4_0, and Q6_K formats
- 4 independent accumulators processing 16 floats per iteration
- Hardware FP16→FP32 conversion (`vcvt_f32_f16`)
- Fused multiply-add (`vfmaq_f32`)
- NEON-optimized RMSNorm, element-wise multiply, and vector add
- Runtime backend selection (`--backend naive` or `--backend neon`)

### Multi-Threading
- Row-level parallel matmul via `parallel_for` template
- Configurable thread count (`--threads N`)
- Optimal at 4 threads on M2 (4 performance cores)

### Sampling Strategies
- Argmax (greedy decoding)
- Temperature scaling
- Top-p (nucleus) sampling

### Benchmarking
- Separate prefill and decode timing
- Configurable warmup, trials, and decode token count
- Per-trial ms/tok computation (handles early EOS correctly)
- Statistical reporting: median, mean, stddev, range

## Project Structure

```
inference-engine/
├── src/
│   ├── main.cpp           # Entry point, CLI argument parsing
│   ├── gguf.h             # GGUF types, quantization block structs
│   ├── gguf_model.h       # GGUFModel class declaration
│   ├── gguf_model.cpp     # GGUF parser, mmap, tensor loading
│   ├── tokenizer.h        # Tokenizer class declaration
│   ├── tokenizer.cpp      # BPE encode/decode implementation
│   ├── ops.h              # Operation interfaces, backend selection
│   ├── ops.cpp            # Naive + NEON implementations of all ops
│   ├── forward.h          # Forward pass declaration
│   ├── forward.cpp        # 22-layer transformer forward pass
│   ├── generate.h         # Generation loop declaration
│   ├── generate.cpp       # Prefill + decode with sampling
│   ├── run_state.h        # Runtime memory allocation (KV cache, scratch)
│   ├── benchmark.cpp      # Benchmarking harness
│   └── test_ops.cpp       # Unit tests for tensor operations
├── models/                # GGUF model files (not in repo)
├── Makefile
└── README.md
```

## Building & Running

```bash
# Build the engine
make

# Run inference
./engine models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --backend neon --threads 4

# Run with custom prompt
./engine models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --backend neon --threads 4 \
    --prompt "What is the meaning of life?"

# Build and run benchmarks
make benchmark
./benchmark models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --backend neon --threads 4

# Run unit tests
make test
```

### CLI Options

| Option | Description | Default | Range |
|--------|-------------|---------|-------|
| `--backend naive\|neon` | Computation backend | `neon` on ARM | — |
| `--threads N` | Number of threads for matmul | `1` | 1+ |
| `--prompt "text"` | Input prompt (auto-wrapped in chat template) | Default demo | — |
| `--temperature F` | Sampling temperature (0 = greedy) | `0.7` | >= 0 |
| `--top_p F` | Nucleus sampling threshold | `0.9` | (0, 1.0] |
| `--max_tokens N` | Maximum tokens to generate | `2048` | [1, 2048] |

## Model Files

This project supports TinyLlama 1.1B in multiple quantization formats:

| Format | File | Size | Download |
|--------|------|------|----------|
| F16 | `tinyllama-1.1b-chat-v1.0.f16.gguf` | 2.2 GB | [HuggingFace](https://huggingface.co/pbatra/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.f16.gguf) |
| Q8_0 | `tinyllama-1.1b-chat-v1.0.Q8_0.gguf` | 1.2 GB | [HuggingFace](https://huggingface.co/pbatra/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf) |
| Q4_0 | `tinyllama-1.1b-chat-v1.0.Q4_0.gguf` | 637 MB | [HuggingFace](https://huggingface.co/pbatra/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf) |
| Q6_K | `tinyllama-1.1b-chat-v1.0.Q6_K.gguf` | 903 MB | [HuggingFace](https://huggingface.co/pbatra/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q6_K.gguf) |

Place model files in the `models/` directory.

## Hardware

- **Tested on**: Apple MacBook Air M2 (8GB RAM)
- **Architecture**: ARM64 with NEON SIMD
- **Optimal config**: 4 threads (M2 has 4 performance + 4 efficiency cores)

## Learning Resources

- [GGUF Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [Llama 3 Architecture](https://github.com/meta-llama/llama3)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Reference implementation

## License

MIT License - Feel free to learn from and build upon this code.
