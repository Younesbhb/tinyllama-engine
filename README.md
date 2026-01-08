# LLM Inference Engine

A minimal Large Language Model inference engine built from scratch in C++20, without relying on frameworks like PyTorch or llama.cpp.

## Why Build This?

The gap between calling `model.generate()` and understanding what actually happens inside an LLM is enormous. This project opens that black box.

LLM inference is memory-bandwidth bound, the CPU spends most of its time **waiting** for weights to arrive from RAM, not computing. Once you understand this, optimization strategies become obvious: quantization (8× less data to move), memory mapping (zero-copy file access), KV caching (don't recompute), and cache-aware algorithms (keep data in L1). This project applies these principles to run a 1.1B parameter model on an M2 MacBook with 8GB RAM.

Building this requires combining disciplines usually taught separately:

| Domain | What's Involved |
|--------|-----------------|
| **Systems Programming** | Memory mapping, binary parsing, RAII |
| **Computer Architecture** | Memory hierarchy, SIMD vectorization |
| **Linear Algebra** | Matrix multiplication, FP16/FP32 precision |
| **Deep Learning** | Transformers, attention, positional encodings |
| **Optimization** | Quantization, KV caching, PagedAttention |


## Project Goals

- **Educational**: Understand every byte between a model file and generated text
- **Portfolio**: Build something that shows genuine systems understanding
- **Minimal**: Zero ML framework dependencies, just standard C++20

## Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        GGUF Model File                          │
│  ┌─────────┐  ┌──────────────┐  ┌─────────────────────────────┐ │
│  │ Header  │  │   Metadata   │  │        Tensor Data          │ │
│  │ (magic, │  │ (config,     │  │  (weights in F16/F32)       │ │
│  │ version)│  │  vocab)      │  │                             │ │
│  └─────────┘  └──────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
        │                │                       │
        ▼                ▼                       ▼
   ┌─────────┐    ┌─────────────┐        ┌─────────────┐
   │ Parser  │    │  Tokenizer  │        │   Tensors   │
   │         │    │             │        │  (mmap'd)   │
   └─────────┘    └─────────────┘        └─────────────┘
        │                │                       │
        └────────────────┼───────────────────────┘
                         ▼
                 ┌───────────────┐
                 │   GGUFModel   │
                 │  (main API)   │
                 └───────────────┘
```

## Completed Phases

### Phase 2: GGUF Parser
- Memory-mapped file I/O for efficient model loading
- Complete GGUF header and metadata parsing
- Tensor index with offset computation and bounds checking
- Support for F16 and F32 data types

### Phase 2.5: Model Configuration
- Extract all Llama architecture parameters from metadata
- Support for Grouped Query Attention (GQA) configuration
- RoPE positional encoding parameters

### Phase 3: Tokenizer
- BPE tokenization with greedy longest-match algorithm
- SentencePiece-compatible space handling (▁ character)
- Encode/decode with round-trip verification
- 32,000 token vocabulary from GGUF metadata

## Upcoming Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 4 | Matrix multiplication | Next |
| 5 | Transformer components (RMSNorm, RoPE, Attention) | Planned |
| 6 | Forward pass & text generation | Planned |
| 7 | KV Cache | Planned |
| 8 | SIMD optimization (ARM NEON) | Planned |
| 9 | Quantization optimization | Planned |
| 10 | PagedAttention optimization | Planned |


## Technical Details

### Target Model
- **Model**: TinyLlama-1.1B-Chat-v1.0
- **Format**: GGUF (F16 weights)
- **Architecture**: Llama (22 layers, 2048 hidden dim, 32 heads)

### Key Techniques
| Component | Technique |
|-----------|-----------|
| File I/O | Memory mapping (`mmap`) for zero-copy access |
| Tokenization | Greedy longest-match BPE |
| Memory safety | RAII wrappers, bounds checking |
| Unicode | UTF-8 handling for SentencePiece compatibility |

### Hardware
- Apple MacBook Air M2 (8GB RAM)
- ARM64 architecture (NEON SIMD planned)

## Project Structure
```
inference-engine/
├── src/
│   ├── main.cpp        # Entry point, GGUFModel class
│   ├── gguf.h          # GGUF types and config structs
│   ├── tokenizer.h     # Tokenizer class declaration
│   ├── tokenizer.cpp   # Tokenizer implementation
│   ├── ops.h           # Tensor operations (coming soon)
│   └── ops.cpp         # MatMul implementation (coming soon)
├── models/             # GGUF model files (not in repo)
├── Makefile
└── README.md
```

## Requirements

This project requires a GGUF model file. This model has been used for implementation:

- [tinyllama-1.1b-chat-v1.0.f16.gguf](https://huggingface.co/pbatra/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.f16.gguf)

Place it in the `models/` directory.


## Building & Running
```bash
# Build
make

# Run
./engine models/tinyllama-1.1b-chat-v1.0.f16.gguf

# Clean
make clean
```


## Learning Resources

- [GGUF Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [Llama 3 Architecture](https://github.com/meta-llama/llama3)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Reference implementation

## License

MIT License - Feel free to learn from and build upon this code.
