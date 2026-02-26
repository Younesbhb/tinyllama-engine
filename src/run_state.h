#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include "gguf.h"

// -------------------- RunState --------------------
//
// Pre-allocated scratch buffers for the forward pass.
//
// Why a single struct? During inference, every operation needs temporary
// storage. Instead of allocating/freeing memory hundreds of times per
// token, we allocate everything once here and reuse the same buffers
// across all 22 layers.
//
// Think of it as the "workbench" — set it up once, reuse for every layer.
//
// Memory layout for TinyLlama 1.1B:
//   Scratch buffers: ~300 KB  (small, fits in L2 cache)
//   KV cache:        ~22 MB   (big, lives in RAM)
//   Logits:          ~125 KB
//   Total:           ~23 MB
//
// The KV cache is by far the largest allocation. This is why
// PagedAttention (Phase 10) matters — it optimizes this memory.

struct RunState {

    // ---- Scratch buffers (reused every layer) ----

    // x: The "running" hidden state. This is the main signal that flows
    // through all 22 layers. Starts as the token embedding, gets
    // transformed by each layer, ends up as the final representation.
    // Size: [n_embd] = [2048]
    float* x = nullptr;

    // xb: Buffer for the output of RMSNorm. We can't normalize in-place
    // because attention/FFN need the original x for the residual connection.
    // Size: [n_embd] = [2048]
    float* xb = nullptr;

    // xb2: Buffer for intermediate results in attention and FFN.
    // Used for: attention output projection, FFN down projection.
    // Size: [n_embd] = [2048]
    float* xb2 = nullptr;

    // q: Query vector after projecting xb through W_q.
    // Size: [n_embd] = [2048] (32 heads × 64 dims per head)
    float* q = nullptr;

    // k: Key vector after projecting xb through W_k.
    // Smaller than q because of GQA — only 4 KV heads instead of 32.
    // Size: [n_head_kv * head_dim] = [4 * 64] = [256]
    float* k = nullptr;

    // v: Value vector after projecting xb through W_v.
    // Same size as k.
    // Size: [n_head_kv * head_dim] = [4 * 64] = [256]
    float* v = nullptr;

    // att: Attention scores for one head.
    // For each head, we compute a score against every previous token.
    // Size: [n_ctx] = [2048] (worst case: attending to max context)
    float* att = nullptr;

    // hb: FFN gate projection buffer. After matmul with W_gate, then silu.
    // Size: [n_ff] = [5632]
    float* hb = nullptr;

    // hb2: FFN up projection buffer. After matmul with W_up.
    // Gets element-wise multiplied with hb (the SwiGLU gating).
    // Size: [n_ff] = [5632]
    float* hb2 = nullptr;


    // ---- Output ----

    // logits: Final output — one score per vocab word.
    // Softmax turns these into probabilities for sampling.
    // Size: [n_vocab] = [32000]
    float* logits = nullptr;


    // ---- KV Cache (persists across tokens) ----
    //
    // Unlike the scratch buffers above, the KV cache is NOT overwritten
    // each layer. It accumulates: every token's keys and values are
    // stored here permanently (until context is full).
    //
    // When generating token #50, the attention mechanism needs to look
    // back at keys/values from tokens #0-#49. That's what this cache holds.
    //
    // Layout: [layer][position][head][dim]
    // Flattened: [n_layers * n_ctx * n_head_kv * head_dim]
    //
    // For TinyLlama: 22 layers × 2048 positions × 4 heads × 64 dims
    //              = 22 × 2048 × 256 = 11,534,336 floats = ~44 MB total
    //              (key_cache ~22 MB + value_cache ~22 MB)

    float* key_cache = nullptr;
    float* value_cache = nullptr;


    // ---- Bookkeeping ----

    // How many tokens have been processed so far.
    // This tells us where to write in the KV cache and how many
    // previous tokens to attend to.
    int pos = 0;


    // ---- Lifecycle ----

    // Allocate all buffers based on model config
    void allocate(const llama_config_t& cfg) {
        std::uint32_t hd = cfg.head_dim();
        std::uint32_t kv_dim = cfg.n_head_kv * hd;

        // Scratch buffers
        x      = new float[cfg.n_embd];
        xb     = new float[cfg.n_embd];
        xb2    = new float[cfg.n_embd];
        q      = new float[cfg.n_embd];      // n_head * head_dim = n_embd
        k      = new float[kv_dim];           // n_head_kv * head_dim
        v      = new float[kv_dim];           // n_head_kv * head_dim
        att    = new float[cfg.n_ctx];        // max sequence length
        hb     = new float[cfg.n_ff];
        hb2    = new float[cfg.n_ff];

        // Output
        logits = new float[cfg.n_vocab];

        // KV cache: [n_layers * n_ctx * kv_dim]
        std::size_t cache_size = static_cast<std::size_t>(cfg.n_layers)
                               * cfg.n_ctx
                               * kv_dim;
        key_cache   = new float[cache_size];
        value_cache = new float[cache_size];

        // Zero everything out
        std::memset(x,      0, sizeof(float) * cfg.n_embd);
        std::memset(xb,     0, sizeof(float) * cfg.n_embd);
        std::memset(xb2,    0, sizeof(float) * cfg.n_embd);
        std::memset(q,      0, sizeof(float) * cfg.n_embd);
        std::memset(k,      0, sizeof(float) * kv_dim);
        std::memset(v,      0, sizeof(float) * kv_dim);
        std::memset(att,    0, sizeof(float) * cfg.n_ctx);
        std::memset(hb,     0, sizeof(float) * cfg.n_ff);
        std::memset(hb2,    0, sizeof(float) * cfg.n_ff);
        std::memset(logits, 0, sizeof(float) * cfg.n_vocab);

        std::memset(key_cache,   0, sizeof(float) * cache_size);
        std::memset(value_cache, 0, sizeof(float) * cache_size);
    }

    // Reset for a new conversation: zero KV cache, reset position.
    // Scratch buffers don't need zeroing — they get overwritten every forward pass.
    // But the KV cache accumulates across tokens, so it must be cleared.
    void reset(const llama_config_t& cfg) {
        std::uint32_t kv_dim = cfg.n_head_kv * cfg.head_dim();
        std::size_t cache_size = static_cast<std::size_t>(cfg.n_layers)
                               * cfg.n_ctx
                               * kv_dim;
        std::memset(key_cache,   0, sizeof(float) * cache_size);
        std::memset(value_cache, 0, sizeof(float) * cache_size);
        pos = 0;
    }

    // Free all buffers
    void free_buffers() {
        delete[] x;           x = nullptr;
        delete[] xb;          xb = nullptr;
        delete[] xb2;         xb2 = nullptr;
        delete[] q;           q = nullptr;
        delete[] k;           k = nullptr;
        delete[] v;           v = nullptr;
        delete[] att;         att = nullptr;
        delete[] hb;          hb = nullptr;
        delete[] hb2;         hb2 = nullptr;
        delete[] logits;      logits = nullptr;
        delete[] key_cache;   key_cache = nullptr;
        delete[] value_cache; value_cache = nullptr;
    }

    // No copying (these buffers are big, accidental copies would be bad)
    RunState(const RunState&) = delete;
    RunState& operator=(const RunState&) = delete;

    // Allow default construction (buffers start as nullptr)
    RunState() = default;

    // Cleanup on destruction
    ~RunState() {
        free_buffers();
    }
};