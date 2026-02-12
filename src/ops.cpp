#include "ops.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <limits>

// -------------------- FP16 → F32 conversion --------------------
// Duplicated from main.cpp for now. When you refactor later, move this
// to a shared utility header. Keeping it here avoids coupling ops.cpp
// to the GGUFModel internals.

static float fp16_to_f32(std::uint16_t h) {
    std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    std::uint32_t exp  = (static_cast<std::uint32_t>(h) >> 10) & 0x1Fu;
    std::uint32_t mant =  static_cast<std::uint32_t>(h) & 0x3FFu;

    std::uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;  // ±0
        } else {
            // Subnormal: normalize it
            int e = -14;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x3FFu;
            std::uint32_t fexp  = static_cast<std::uint32_t>(e + 127);
            std::uint32_t fmant = mant << 13;
            bits = sign | (fexp << 23) | fmant;
        }
    } else if (exp == 31) {
        // Inf or NaN
        bits = sign | 0x7F800000u | (mant << 13);
        if (mant != 0) bits |= 0x00400000u;  // quiet NaN
    } else {
        // Normal number
        std::uint32_t fexp  = exp + (127 - 15);
        std::uint32_t fmant = mant << 13;
        bits = sign | (fexp << 23) | fmant;
    }

    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}


// -------------------- Matrix-Vector Multiply (F32 weights) --------------------
//
// For each output row i:
//   out[i] = dot(W[i, :], x)  = sum over j of W[i*cols + j] * x[j]
//
// Memory layout: W is row-major, so W[i][j] = W[i * cols + j]
// This means we access W sequentially (good for cache).
//
// Complexity: O(rows * cols) multiplications
// Bottleneck: Loading W from RAM. For a 2048x2048 F32 matrix = 16MB.
//             At ~100 GB/s bandwidth, that's ~160μs just to load.
//             The actual multiplications take far less time.

void matmul_f32(float* out, const float* W, const float* x,
                int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        // Pointer to the start of row i in the weight matrix
        const float* row = W + i * cols;
        for (int j = 0; j < cols; j++) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}


// -------------------- Matrix-Vector Multiply (F16 weights) --------------------
//
// Same as above, but weights are stored as 16-bit floats.
// We convert each weight to F32 on the fly before multiplying.
//
// This is the common case for TinyLlama F16 GGUF files.
// The conversion overhead is negligible because the CPU is already
// waiting for the next cache line of weights to arrive from RAM.

void matmul_f16(float* out, const std::uint16_t* W, const float* x,
                int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        // Pointer to the start of row i (each element is 2 bytes)
        const std::uint16_t* row = W + i * cols;
        for (int j = 0; j < cols; j++) {
            float w = fp16_to_f32(row[j]);
            sum += w * x[j];
        }
        out[i] = sum;
    }
}


// -------------------- Matrix-Vector Multiply (type dispatch) --------------------
//
// This is the function you'll call in the forward pass.
// It dispatches to the correct typed implementation based on the
// ggml_type stored in the tensor info.

void matmul(float* out, const void* W, const float* x,
            int rows, int cols, ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            matmul_f32(out, static_cast<const float*>(W), x, rows, cols);
            break;
        case GGML_TYPE_F16:
            matmul_f16(out, static_cast<const std::uint16_t*>(W), x, rows, cols);
            break;
        default:
            throw std::runtime_error(
                "matmul: unsupported weight type (only F16/F32 for now). "
                "Quantized types (Q4_0, Q8_0) will be added in Phase 9.");
    }
}


// -------------------- RMSNorm --------------------
//
// RMSNorm is simpler than LayerNorm:
//   - LayerNorm: subtract mean, divide by stddev, scale + shift
//   - RMSNorm:   divide by RMS, scale only (no mean, no shift)
//
// Steps:
//   1. Compute mean of squares: ms = (1/n) * sum(x_i^2)
//   2. Compute normalization factor: rsqrt = 1 / sqrt(ms + eps)
//   3. Scale: out_i = x_i * rsqrt * weight_i
//
// The epsilon (eps) prevents division by zero when the vector is all zeros.
// TinyLlama uses eps = 1e-5 (from config.rms_norm_eps).

void rmsnorm(float* out, const float* x, const float* weight,
             int size, float eps) {
    // Step 1: sum of squares
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }

    // Step 2: 1 / sqrt(mean_of_squares + eps)
    float rms = 1.0f / std::sqrt(ss / static_cast<float>(size) + eps);

    // Step 3: normalize and scale
    for (int i = 0; i < size; i++) {
        out[i] = x[i] * rms * weight[i];
    }
}


// -------------------- Softmax --------------------
//
// Converts raw scores (logits) into a probability distribution.
// P_i = exp(x_i - max) / sum(exp(x_j - max))
//
// The max-subtraction trick is essential for numerical stability:
//   - Without it: exp(1000) = Inf, and Inf/Inf = NaN
//   - With it: exp(1000 - 1000) = exp(0) = 1, perfectly fine
//
// Used in two places:
//   1. Attention: softmax over attention scores (per-head, per-query)
//   2. Generation: softmax over final logits to get token probabilities

void softmax(float* x, int size) {
    // Step 1: find max for numerical stability
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    // Step 2: exp(x_i - max) and accumulate sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }

    // Step 3: normalize so all values sum to 1.0
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}


// -------------------- SiLU (Swish) Activation --------------------
//
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// This is used in the FFN's SwiGLU mechanism:
//   FFN output = W_down * (silu(W_gate * x) ⊙ (W_up * x))
//
// SiLU is smoother than ReLU near zero, which gives better gradients.
// It's the activation that made Llama/PaLM models work better than
// older architectures using plain ReLU.

void silu(float* x, int size) {
    for (int i = 0; i < size; i++) {
        float val = x[i];
        // sigmoid(x) = 1 / (1 + exp(-x))
        float sigmoid = 1.0f / (1.0f + std::exp(-val));
        // silu(x) = x * sigmoid(x)
        x[i] = val * sigmoid;
    }
}


// -------------------- Element-wise Multiply --------------------
//
// a[i] *= b[i]
//
// Used in SwiGLU after applying silu to the gate projection:
//   hidden = silu(gate) ⊙ up
//
// The ⊙ symbol means Hadamard product (element-wise multiply).
// This "gating" mechanism lets the network learn to selectively
// pass or block information through the FFN.

void elementwise_mul(float* a, const float* b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] *= b[i];
    }
}


// -------------------- Vector Add (Residual Connection) --------------------
//
// out[i] += x[i]
//
// The residual connection is arguably the most important architectural
// idea in deep learning. Without it, gradients vanish in deep networks.
//
// In the Transformer block:
//   x = x + attention(rmsnorm(x))    ← first residual
//   x = x + ffn(rmsnorm(x))          ← second residual
//
// The original signal always flows through unchanged; the layers only
// need to learn the "delta" (what to add/modify).

void vec_add(float* out, const float* x, int size) {
    for (int i = 0; i < size; i++) {
        out[i] += x[i];
    }
}
