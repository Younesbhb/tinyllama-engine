#pragma once
#include <cstdint>
#include "gguf.h"

// -------------------- Tensor Operations --------------------
// These are the building blocks for the Transformer forward pass.
// All runtime computation happens in F32. Weights may be stored as F16
// in the GGUF file and are converted on the fly during matmul.
//
// Phase 4: Naive implementations (correct but slow)
// Phase 8: Will be optimized with ARM NEON SIMD
// -------------------- ----------------------------------------


// -------------------- Matrix-Vector Multiply --------------------
// The core operation of LLM inference.
// Computes: out[i] = sum_j( W[i,j] * x[j] )  for i in [0, rows)
//
// This is memory-bandwidth bound: the CPU finishes the math before
// the next row of weights arrives from RAM. That's why quantization
// helps — less data to move.
//
// Parameters:
//   out    - output vector  [rows]          (F32, caller-allocated)
//   W      - weight matrix  [rows x cols]   (raw bytes, F16 or F32)
//   x      - input vector   [cols]          (F32)
//   rows   - number of output features (outer dimension)
//   cols   - number of input features  (inner dimension)
//   type   - ggml_type of the weight matrix (GGML_TYPE_F16 or GGML_TYPE_F32)

void matmul(float* out, const void* W, const float* x,
            int rows, int cols, ggml_type type);

// Convenience wrappers when you know the type at call site
void matmul_f32(float* out, const float* W, const float* x,
                int rows, int cols);

void matmul_f16(float* out, const std::uint16_t* W, const float* x,
                int rows, int cols);


// -------------------- RMSNorm --------------------
// Llama uses RMSNorm instead of LayerNorm (cheaper: no mean calculation).
// Formula: x_norm = (x / sqrt(mean(x^2) + eps)) * weight
//
// Parameters:
//   out    - output vector  [size]   (F32, can alias input for in-place)
//   x      - input vector   [size]   (F32)
//   weight - learned scale  [size]   (F32, from "attn_norm.weight" etc.)
//   size   - vector dimension
//   eps    - epsilon for numerical stability (rms_norm_eps from config)

void rmsnorm(float* out, const float* x, const float* weight,
             int size, float eps);


// -------------------- Softmax --------------------
// Converts logits to probabilities: P_i = exp(x_i) / sum(exp(x_j))
// Uses the max-subtraction trick for numerical stability.
//
// Operates in-place on x.
//   x      - input/output vector [size]
//   size   - number of elements

void softmax(float* x, int size);


// -------------------- SiLU (Swish) Activation --------------------
// Used inside SwiGLU in the FFN.
// Formula: silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
//
// Operates in-place.
//   x      - input/output vector [size]
//   size   - number of elements

void silu(float* x, int size);


// -------------------- Element-wise Multiply --------------------
// Used in SwiGLU: output = silu(gate) * up
// Formula: out[i] = a[i] * b[i]
//
// Operates in-place on a: a[i] *= b[i]
//   a      - first vector (modified in-place)  [size]
//   b      - second vector                     [size]
//   size   - number of elements

void elementwise_mul(float* a, const float* b, int size);


// -------------------- Vector Add (Residual Connection) --------------------
// The residual (skip) connection: out = out + x
// This is what makes deep networks trainable — gradients flow through
// the skip connection even if the layer's contribution is small.
//
// Operates in-place on out: out[i] += x[i]
//   out    - accumulator vector (modified in-place) [size]
//   x      - vector to add                          [size]
//   size   - number of elements

void vec_add(float* out, const float* x, int size);


// -------------------- RoPE (Rotary Positional Embeddings) --------------------
// Encodes word order by rotating Q and K vectors based on position.
//
// Without RoPE, "dog bites man" and "man bites dog" look the same to
// attention because the same words produce the same Q/K vectors.
// RoPE fixes this by rotating each vector differently based on where
// the word appears in the sequence.
//
// How it works:
//   - Groups the 64 dimensions of each head into 32 pairs
//   - Rotates each pair by an angle that depends on:
//     (a) the token's position in the sequence (pos)
//     (b) which pair it is (earlier pairs rotate faster)
//   - The rotation formula for pair i at position pos:
//       angle = pos / (10000 ^ (2i / head_dim))
//       new_x = x * cos(angle) - y * sin(angle)
//       new_y = x * sin(angle) + y * cos(angle)
//
// Operates in-place on q and k.
//
// Parameters:
//   q         - query vector  [n_embd]              (all heads concatenated)
//   k         - key vector    [n_head_kv * head_dim] (KV heads only)
//   pos       - position of this token in the sequence (0, 1, 2, ...)
//   head_dim  - dimension per head (64 for TinyLlama)
//   n_head    - number of query heads (32)
//   n_head_kv - number of KV heads (4)
//   freq_base - base frequency for angle calculation (10000.0)

void rope(float* q, float* k, int pos, int head_dim,
          int n_head, int n_head_kv, float freq_base);


// -------------------- Multi-Head GQA Attention --------------------
// The core mechanism that lets the model "look back" at previous tokens.
//
// What happens:
//   1. Store current token's k and v into the KV cache
//   2. For each query head:
//      a. Find which KV head it shares (GQA: every 8 query heads share 1 KV head)
//      b. Compute dot product of q against all cached keys → scores
//      c. Scale scores by 1/sqrt(head_dim) to prevent overflow
//      d. Softmax → attention weights (percentages)
//      e. Weighted sum of cached values → this head's output (64 floats)
//   3. All 32 heads' outputs are concatenated → 2048 floats
//
// Parameters:
//   out         - output vector [n_embd]  (all heads concatenated)
//   q           - query vector  [n_embd]  (already rotated by RoPE)
//   k           - key vector    [kv_dim]  (already rotated by RoPE)
//   v           - value vector  [kv_dim]
//   key_cache   - cached keys   [n_layers * n_ctx * kv_dim]
//   value_cache - cached values [n_layers * n_ctx * kv_dim]
//   att         - scratch buffer for attention scores [n_ctx]
//   layer       - which layer (0-21), for indexing into KV cache
//   pos         - current token position
//   cfg         - model config (dimensions, head counts, etc.)

void attention(float* out, const float* q, const float* k, const float* v,
               float* key_cache, float* value_cache, float* att,
               int layer, int pos, const llama_config_t& cfg);


// -------------------- SwiGLU Feed-Forward Network --------------------
// The "thinking" part of each transformer block.
// Attention gathers context, FFN processes it.
//
// Flow:
//   1. gate = matmul(W_gate, input)   [n_embd → n_ff]   (expand)
//   2. up   = matmul(W_up,   input)   [n_embd → n_ff]   (expand)
//   3. silu(gate)                                         (activation)
//   4. gate = gate ⊙ up                                  (gating)
//   5. out  = matmul(W_down, gate)    [n_ff → n_embd]   (shrink back)
//
// Parameters:
//   out     - output vector [n_embd]                (result goes here)
//   input   - input vector  [n_embd]                (normalized hidden state)
//   hb      - scratch buffer [n_ff]                 (gate projection)
//   hb2     - scratch buffer [n_ff]                 (up projection)
//   W_gate  - gate weight matrix [n_ff × n_embd]   (raw bytes from GGUF)
//   W_up    - up weight matrix   [n_ff × n_embd]   (raw bytes from GGUF)
//   W_down  - down weight matrix [n_embd × n_ff]   (raw bytes from GGUF)
//   n_ff    - FFN intermediate dimension (5632)
//   n_embd  - hidden dimension (2048)
//   type    - ggml_type of the weight matrices

void ffn_swiglu(float* out, const float* input,
                float* hb, float* hb2,
                const void* W_gate, const void* W_up, const void* W_down,
                int n_ff, int n_embd, ggml_type type);