#pragma once
#include <cstdint>
#include <string>
#include <vector>

enum gguf_type : std::uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT
};

enum ggml_type : std::uint32_t {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,   // not implemented, listed for correct numbering
    // gap: 4-6 are other types we don't use
    GGML_TYPE_Q8_0 = 8,
    // K-quant types (super-block quantization, block size 256)
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_COUNT = 40
};

// -------------------- Quantization Block Structures --------------------
//
// Quantized weights are stored in blocks of 32 elements.
// Each block has a scale factor (F16) and compressed integer values.
// To dequantize: float_value = integer_value * scale
//
// This is how the model shrinks from 2.2GB (F16) to 1.17GB (Q8_0)
// or 637MB (Q4_0) — the weights are stored as small integers instead
// of 16-bit floats.

// Q8_0: 8-bit quantization
// Each block: 2 bytes (scale) + 32 bytes (32 × int8) = 34 bytes
// Stores 32 weights in 34 bytes vs 64 bytes for F16 → ~1.88× compression
//
// The scale is the maximum absolute value in the block divided by 127.
// Each int8 value is the original weight divided by scale, rounded.
//
// Block size just means how many weights are grouped together and share one scale factor.
// For both Q8_0 and Q4_0, the block size is 32.
static constexpr int QK8_0 = 32;  // block size

struct block_q8_0 {
    /*
    d — The scale factor, stored as F16 (2 bytes). 
    When the model was quantized, whoever made the file took each group of 32 weights,
    found the range, and computed a scale so that dividing every weight by this scale fits
    into -128 to +127. For example, if the 32 weights range from -0.5 to +0.5, 
    the scale might be 0.004 (0.5 / 127 ≈ 0.004).
    */
    std::uint16_t d;       // scale factor stored as F16 (2 bytes)
    /*
    qs[32] — The 32 quantized weights as signed 8-bit integers. 
    Each one ranges from -128 to +127. To get the real float value: float_weight = qs[j] * fp16_to_f32(d).
    For example, if qs[5] is 63 and the scale is 0.004, the real weight is 63 × 0.004 = 0.252.
    */
    std::int8_t   qs[QK8_0]; // quantized values: 32 × int8 (32 bytes)
};
// Total: 34 bytes for 32 weights
static_assert(sizeof(block_q8_0) == 34, "Q8_0 block must be 34 bytes");

// Q4_0: 4-bit quantization
// Each block: 2 bytes (scale) + 16 bytes (32 × int4 packed) = 18 bytes
// Stores 32 weights in 18 bytes vs 64 bytes for F16 → ~3.56× compression
//
// Two 4-bit values are packed into each byte:
//   byte = (high_nibble << 4) | low_nibble
// The raw nibble range is 0-15; subtract 8 to get signed range -8 to +7.
//
// The block size of 32 for Q8_0 and Q4_0 is defined in the llama.cpp source code,
// which is the reference implementation that created the GGML/GGUF format. 
// Specifically in their ggml-common.h:

static constexpr int QK4_0 = 32;  // block size

struct block_q4_0 {
    std::uint16_t d;          // scale factor stored as F16 (2 bytes)
    std::uint8_t  qs[QK4_0 / 2]; // quantized values: 16 bytes (2 values per byte)
};
// Total: 18 bytes for 32 weights
static_assert(sizeof(block_q4_0) == 18, "Q4_0 block must be 18 bytes");


// Q6_K: 6-bit "k-quant" quantization (super-block format)
// Unlike Q4_0 and Q8_0 which use blocks of 32 elements, Q6_K uses
// "super-blocks" of 256 elements. This is the format llama.cpp uses
// for the output.weight tensor even in Q4_0 models, because the final
// vocabulary projection is sensitive to quantization error and benefits
// from the extra precision (6.5625 bits per weight vs 4.5).
//
// Each super-block stores 256 weights as:
//   - ql[128]: Lower 4 bits of each 6-bit quantized value (2 per byte)
//   - qh[64]:  Upper 2 bits of each 6-bit quantized value (4 per byte)
//   - scales[16]: Per-sub-block int8 scale factors (one per 16 elements)
//   - d: Super-block scale (F16), shared across all 256 weights
//
// To dequantize element i:
//   1. Extract 6-bit value: q = (lower_4_bits | (upper_2_bits << 4)) - 32
//   2. float_weight = fp16_to_f32(d) * scales[sub_block] * q
//
// The bit layout is interleaved for SIMD efficiency (not sequential),
// matching the llama.cpp reference implementation in ggml-quants.c.

static constexpr int QK_K = 256;  // super-block size for k-quants

/*
The division isn't extracting the bits — it's calculating how many bytes we need to store them when we pack multiple values per byte.
Each of the 256 weights has a 6-bit quantized value. Those 6 bits are split into two pieces: the lower 4 bits and the upper 2 bits, stored in separate arrays.
ql[QK_K / 2] = ql[128] — lower 4 bits
Each weight contributes 4 bits. Two sets of 4 bits fit in one byte (one in the low nibble, one in the high nibble). So 256 weights ÷ 2 per byte = 128 bytes.
qh[QK_K / 4] = qh[64] — upper 2 bits
Each weight contributes 2 bits. Four sets of 2 bits fit in one byte (bits 0-1, bits 2-3, bits 4-5, bits 6-7). So 256 weights ÷ 4 per byte = 64 bytes.
*/
struct block_q6_K {
    std::uint8_t ql[QK_K / 2];     // lower 4 bits of quants (128 bytes)
    std::uint8_t qh[QK_K / 4];     // upper 2 bits of quants (64 bytes)
    std::int8_t  scales[QK_K / 16]; // sub-block scales (16 bytes)
    std::uint16_t d;                // super-block scale as F16 (2 bytes)
};
// Total: 128 + 64 + 16 + 2 = 210 bytes for 256 weights
// 210 * 8 / 256 = 6.5625 bits per weight
static_assert(sizeof(block_q6_K) == 210, "Q6_K block must be 210 bytes");


// Model configuration extracted from GGUF metadata
struct llama_config_t {
    // Architecture
    std::string architecture;      // "llama", "mistral", etc.
    
    // Core dimensions
    std::uint32_t n_layers = 0;           // llama.block_count/llama.layer_count AKA transformer blocks
    std::uint32_t n_embd = 0;             // llama.embedding_length (hidden dimensions)
    std::uint32_t n_head = 0;             // llama.attention.head_count nb of attention heads (Query heads)
    std::uint32_t n_head_kv = 0;          // llama.attention.head_count_kv (for GQA) AKA Key/value heads, 4 KV heads

    std::uint32_t n_ff = 0;               // llama.feed_forward_length
    
    // RoPE parameters
    std::uint32_t rope_dim = 0;           // llama.rope.dimension_count
    float rope_freq_base = 10000.0f;      // llama.rope.freq_base
    
    // Context
    std::uint32_t n_ctx = 2048;           // llama.context_length AKA max sequence length (context window)
    std::uint32_t n_vocab = 0;            // vocabulary size inferred from token_embd dimensions
    
    // Normalization
    float rms_norm_eps = 1e-5f;           // llama.attention.layer_norm_rms_epsilon
    
    // Computed values (filled after parsing tensors)
    //This calculates the dimension of each attention head.
    // n_embd / n_head
    std::uint32_t head_dim() const { 
        return (n_head > 0) ? (n_embd / n_head) : 0; 
    }
};




struct gguf_tensor_info_t {
    std::string name;
    std::uint32_t n_dimensions = 0;
    std::vector<std::uint64_t> dimensions;
    ggml_type type = GGML_TYPE_F32;
    std::uint64_t offset = 0; // relative to tensor_data_start

    // computed for safety/debug
    std::uint64_t n_elems = 0;
    std::uint64_t byte_size = 0;
    std::uint64_t abs_offset = 0; // absolute file offset
};