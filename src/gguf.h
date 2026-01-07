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
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    // many others exist; we intentionally treat them as "unsupported" for now
    GGML_TYPE_COUNT = 40
};


// Model configuration extracted from GGUF metadata
struct llama_config_t {
    // Architecture
    std::string architecture;      // "llama", "mistral", etc.
    
    // Core dimensions
    std::uint32_t n_layers = 0;           // llama.block_count/llama.layer_count AKA transformer blocks
    std::uint32_t n_embd = 0;             // llama.embedding_length (hidden dimensions)
    std::uint32_t n_head = 0;             // llama.attention.head_count nb of attention heads (Query heads)
    std::uint32_t n_head_kv = 0;          // llama.attention.head_count_kv (for GQA) AKA Key/value heads (GQA: 32/4 = 8 queries share each KV)

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
