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


// -------------------- RoPE (Rotary Positional Embeddings) --------------------
//
// Let's walk through what happens with a concrete example.
//
// Say we have head_dim = 64, so there are 32 pairs: (dim0,dim1), (dim2,dim3), ...
// Say this word is at position 5 in the sequence (pos = 5).
//
// For pair i=0 (the fastest-rotating pair):
//   freq = 1.0 / (10000^(0/64)) = 1.0 / 1.0 = 1.0
//   angle = 5 * 1.0 = 5.0 radians
//   Rotate (dim0, dim1) by 5.0 radians
//
// For pair i=1:
//   freq = 1.0 / (10000^(2/64)) = 1.0 / 1.9307 = 0.518
//   angle = 5 * 0.518 = 2.59 radians
//   Rotate (dim2, dim3) by 2.59 radians
//
// For pair i=31 (the slowest-rotating pair):
//   freq = 1.0 / (10000^(62/64)) = very small number
//   angle = 5 * tiny = almost 0
//   Barely rotate (dim62, dim63)
//
// The result: early pairs carry fine-grained position info (they rotate
// a lot between adjacent positions), late pairs carry coarse position
// info (they barely change). This is similar to how a clock has a fast
// second hand and a slow hour hand — together they precisely tell time.
//
// We apply the SAME rotation logic to both Q and K vectors. This way,
// when attention computes dot(Q, K), the result naturally depends on
// the relative distance between the two tokens.

void rope(float* q, float* k, int pos, int head_dim,
          int n_head, int n_head_kv, float freq_base) {

    // Number of pairs per head (each pair = 2 dimensions)
    int n_pairs = head_dim / 2;

    // Rotate all query heads
    for (int h = 0; h < n_head; h++) {
        // Pointer to this head's slice of the q vector
        // Head 0 starts at q[0], head 1 at q[64], head 2 at q[128], etc.
        float* head_q = q + h * head_dim;

        for (int i = 0; i < n_pairs; i++) {
            // Compute the rotation angle for this pair at this position
            //
            // freq = 1 / (10000 ^ (2i / head_dim))
            // angle = pos * freq
            //
            // The pow() computes 10000^(2i/64). For i=0 this is 1.0,
            // for i=31 this is nearly 10000. So early pairs rotate
            // fast and late pairs rotate slow.
            float freq = 1.0f / std::pow(freq_base,
                                         static_cast<float>(2 * i) / static_cast<float>(head_dim));
            float angle = static_cast<float>(pos) * freq;

            // Precompute cos and sin (used for the 2D rotation)
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);

            // The two elements of this pair
            float x0 = head_q[2 * i];      // "x coordinate"
            float x1 = head_q[2 * i + 1];  // "y coordinate"

            // Apply 2D rotation
            head_q[2 * i]     = x0 * cos_a - x1 * sin_a;
            head_q[2 * i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }

    // Rotate all KV heads (same logic, fewer heads)
    for (int h = 0; h < n_head_kv; h++) {
        float* head_k = k + h * head_dim;

        for (int i = 0; i < n_pairs; i++) {
            float freq = 1.0f / std::pow(freq_base,
                                         static_cast<float>(2 * i) / static_cast<float>(head_dim));
            float angle = static_cast<float>(pos) * freq;

            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);

            float x0 = head_k[2 * i];
            float x1 = head_k[2 * i + 1];

            head_k[2 * i]     = x0 * cos_a - x1 * sin_a;
            head_k[2 * i + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}


// -------------------- Multi-Head GQA Attention --------------------
//
// This is the most complex operation in the entire engine.
// Let's trace through it carefully.
//
// KV Cache Layout:
//   The cache is a flat array: [n_layers * n_ctx * kv_dim]
//   To find where layer L, position P, KV head H starts:
//     offset = L * (n_ctx * kv_dim) + P * kv_dim + H * head_dim
//
//   For TinyLlama:
//     kv_dim = 4 heads * 64 dims = 256
//     Layer 0, position 0: offset = 0
//     Layer 0, position 1: offset = 256
//     Layer 0, position 5: offset = 1280
//     Layer 1, position 0: offset = 2048 * 256 = 524288
//
// GQA Mapping:
//   32 query heads share 4 KV heads. The mapping is:
//     Query heads 0-7   → KV head 0
//     Query heads 8-15  → KV head 1
//     Query heads 16-23 → KV head 2
//     Query heads 24-31 → KV head 3
//   Formula: kv_head = query_head / (n_head / n_head_kv)
//   For TinyLlama: kv_head = query_head / 8

void attention(float* out, const float* q, const float* k, const float* v,
               float* key_cache, float* value_cache, float* att,
               int layer, int pos, const llama_config_t& cfg) {

    int head_dim = static_cast<int>(cfg.head_dim());
    int n_head = static_cast<int>(cfg.n_head);
    int n_head_kv = static_cast<int>(cfg.n_head_kv);
    int n_ctx = static_cast<int>(cfg.n_ctx);
    int kv_dim = n_head_kv * head_dim;

    // How many query heads share each KV head
    // For TinyLlama: 32 / 4 = 8
    int gqa_ratio = n_head / n_head_kv;

    // Scale factor for attention scores: 1 / sqrt(head_dim)
    // Without this, dot products grow with head_dim and softmax saturates
    // (all attention goes to one token, ignoring everything else).
    // For head_dim=64: scale = 1/8 = 0.125
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // ---- Step 1: Store current token's k and v in the KV cache ----
    //
    // The cache pointer for this layer starts at:
    //   layer * n_ctx * kv_dim
    // Within that, position pos starts at:
    //   pos * kv_dim
    //
    // We copy all kv_dim values (4 heads × 64 dims = 256 floats)

    int layer_offset = layer * n_ctx * kv_dim;

    float* k_cache_pos = key_cache + layer_offset + pos * kv_dim;
    float* v_cache_pos = value_cache + layer_offset + pos * kv_dim;

    for (int i = 0; i < kv_dim; i++) {
        k_cache_pos[i] = k[i];
        v_cache_pos[i] = v[i];
    }

    // ---- Step 2: For each query head, compute attention ----

    for (int qh = 0; qh < n_head; qh++) {

        // Pointer to this query head's 64-dim slice
        const float* q_head = q + qh * head_dim;

        // Which KV head does this query head use?
        int kvh = qh / gqa_ratio;

        // ---- Step 2a: Compute attention scores ----
        //
        // For each cached position (0 through pos), compute:
        //   score = dot(q_head, cached_key) * scale
        //
        // This tells us "how relevant is position t to the current token?"

        for (int t = 0; t <= pos; t++) {
            // Pointer to the cached key at position t, KV head kvh
            const float* k_cached = key_cache + layer_offset + t * kv_dim + kvh * head_dim;

            // Dot product of q_head and k_cached (both are 64 floats)
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_head[d] * k_cached[d];
            }

            att[t] = score * scale;
        }

        // ---- Step 2b: Softmax the scores into attention weights ----
        //
        // After this, att[0..pos] contains percentages that sum to 1.0
        // For example: [0.04, 0.52, 0.18, 0.03, 0.23]
        // meaning "pay 52% attention to position 1, 18% to position 2, etc."

        softmax(att, pos + 1);

        // ---- Step 2c: Weighted sum of cached values ----
        //
        // For each of the 64 output dimensions:
        //   out[d] = sum over all positions t of: att[t] * v_cache[t][d]
        //
        // This blends the value vectors based on attention weights.
        // If "cat" got 52% attention, its value vector contributes 52%
        // to the output.

        // Pointer to where this head's output goes
        // Head 0 → out[0..63], head 1 → out[64..127], etc.
        float* out_head = out + qh * head_dim;

        // Zero out first (we're accumulating a sum)
        for (int d = 0; d < head_dim; d++) {
            out_head[d] = 0.0f;
        }

        // Accumulate weighted values
        for (int t = 0; t <= pos; t++) {
            // Pointer to cached value at position t, KV head kvh
            const float* v_cached = value_cache + layer_offset + t * kv_dim + kvh * head_dim;

            float weight = att[t];
            for (int d = 0; d < head_dim; d++) {
                out_head[d] += weight * v_cached[d];
            }
        }
    }
    // At this point, out[0..2047] contains the concatenated output of all 32 heads.
    // Next step (outside this function): multiply by W_o to project back.
}