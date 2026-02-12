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
