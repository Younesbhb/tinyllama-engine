#include "forward.h"


#include "gguf_model.h"   // full GGUFModel definition (needed to call methods)
#include "run_state.h"     // full RunState definition
#include "ops.h"           // matmul, rmsnorm, rope, attention, ffn_swiglu, etc.

#include <string>
#include <stdexcept>
#include <cstring>

// -------------------- Forward Pass Helpers --------------------


// embed_token(out, model, token) — Goes to the embedding table (a big grid of 32,000 rows × 2,048 columns stored in the model file) and 
// copies row number 'token' into 'out'. If token is 15043, it copies the 15,043rd row — 2,048 numbers that represent what "Hello" means to the model.
// These numbers were learned during training.
static void embed_token(float* out, const GGUFModel& model, int token) {
    const auto& t = model.tensor_info("token_embd.weight");
    /*
    This gets a pointer to where the embedding table starts in the memory-mapped file. 
    It comes back as uint8_t* (raw bytes) because at this point we don't know what 
    format the data is in — could be F16, F32, Q8_0, etc.
    */
    const std::uint8_t* data = model.tensor_bytes(t);
    // This is 2048 — the embedding dimension. Each token's embedding is a row of 2048 numbers.
    int n_embd = static_cast<int>(model.config().n_embd);

    if (t.type == GGML_TYPE_F16) {
        // Now we know the type is F16, so we tell the compiler "treat these raw bytes as 16-bit values." 
        // The pointer emb points to the same memory address as data, 
        // but now the compiler knows each element is 2 bytes. 
        const std::uint16_t* emb = reinterpret_cast<const std::uint16_t*>(data);
        /*
        This jumps to the right row in the embedding table. The table is a grid:
        ```
        Row 0     (token "<unk>"):  [2048 F16 values]
        Row 1     (token "<s>"):    [2048 F16 values]
        Row 2     (token "</s>"):   [2048 F16 values]
        ...
        Row 15043 (token "Hello"):  [2048 F16 values]   ← if token = 15043
        ...
        Row 31999:                  [2048 F16 values]
        */
        const std::uint16_t* row = emb + static_cast<std::size_t>(token) * n_embd;
        for (int i = 0; i < n_embd; i++) {
            out[i] = fp16_to_f32(row[i]);
        }
    } else if (t.type == GGML_TYPE_F32) {
        const float* emb = reinterpret_cast<const float*>(data);
        const float* row = emb + static_cast<std::size_t>(token) * n_embd;
        for (int i = 0; i < n_embd; i++) {
            out[i] = row[i];
        }
    } else if (t.type == GGML_TYPE_Q8_0) {
        // Each row is (n_embd / 32) blocks, each block is 34 bytes
        // To find row 'token': skip token * blocks_per_row blocks
        int blocks_per_row = n_embd / QK8_0;
        const block_q8_0* blocks = reinterpret_cast<const block_q8_0*>(data);
        const block_q8_0* row = blocks + static_cast<std::size_t>(token) * blocks_per_row;

        // Dequantize: for each block, multiply int8 values by the block's scale
        // For each block in a certain row 
        for (int b = 0; b < blocks_per_row; b++) {
            float scale = fp16_to_f32(row[b].d);
            // For each weight loop
            for (int j = 0; j < QK8_0; j++) {
                out[b * QK8_0 + j] = static_cast<float>(row[b].qs[j]) * scale;
            }
        }
    } else if (t.type == GGML_TYPE_Q4_0) {
        // Each row is (n_embd / 32) blocks, each block is 18 bytes
        int blocks_per_row = n_embd / QK4_0;
        const block_q4_0* blocks = reinterpret_cast<const block_q4_0*>(data);
        const block_q4_0* row = blocks + static_cast<std::size_t>(token) * blocks_per_row;

        for (int b = 0; b < blocks_per_row; b++) {
            float scale = fp16_to_f32(row[b].d);
            int base_idx = b * QK4_0;
            for (int j = 0; j < QK4_0 / 2; j++) {
                // Low nibble → element j (first 16 of the block)
                int v0 = static_cast<int>(row[b].qs[j] & 0x0F) - 8;
                // High nibble → element j + 16 (second 16 of the block)
                int v1 = static_cast<int>(row[b].qs[j] >> 4) - 8;

                out[base_idx + j]              = static_cast<float>(v0) * scale;
                out[base_idx + j + QK4_0 / 2]  = static_cast<float>(v1) * scale;
            }
        }
    } else {
        throw std::runtime_error("Unsupported embedding type");
    }
}

// Looks up a small array of 2,048 numbers from the model file by name. 
// These are the learned weights for RMSNorm 
static const float* get_norm_weight(const GGUFModel& model, const std::string& name) {
    const auto& t = model.tensor_info(name);
    if (t.type != GGML_TYPE_F32) {
        throw std::runtime_error("Expected F32 norm weight: " + name);
    }
    return reinterpret_cast<const float*>(model.tensor_bytes(t));
}

struct WeightRef {
    const void* data;
    ggml_type type;
};

// Looks up a weight matrix from the model file by name. 
// Returns both the pointer to the data and what format it's in (F16 or F32), 
// so matmul knows how to read it.
static WeightRef get_weight(const GGUFModel& model, const std::string& name) {
    const auto& t = model.tensor_info(name);
    return { model.tensor_bytes(t), t.type };
}

// -------------------- Forward Pass --------------------
// model — The model file. Contains all the weight matrices (the "knowledge" the model learned during training). Read-only, never changes.
// state — Scratch space. Contains temporary buffers (x, xb, q, k, v, etc.) and the KV cache. Gets overwritten every call.
// token — The token ID to process (like 15043 for "Hello").
// pos — Where this token sits in the sequence (0, 1, 2, ...). Used by RoPE to encode word order.

void forward(GGUFModel& model, RunState& state, int token, int pos) {
    const auto& cfg = model.config();
    int n_embd   = static_cast<int>(cfg.n_embd);
    int n_layers = static_cast<int>(cfg.n_layers);
    int n_ff     = static_cast<int>(cfg.n_ff);
    int kv_dim   = static_cast<int>(cfg.n_head_kv * cfg.head_dim());

    // Convert the token ID into a vector of 2,048 numbers
    embed_token(state.x, model, token);

    // Run for 22 layers
    for (int l = 0; l < n_layers; l++) {
        std::string prefix = "blk." + std::to_string(l) + ".";

        const float* attn_norm_w = get_norm_weight(model, prefix + "attn_norm.weight");
        rmsnorm(state.xb, state.x, attn_norm_w, n_embd, cfg.rms_norm_eps);

        auto wq = get_weight(model, prefix + "attn_q.weight");
        auto wk = get_weight(model, prefix + "attn_k.weight");
        auto wv = get_weight(model, prefix + "attn_v.weight");

        matmul(state.q, wq.data, state.xb, n_embd, n_embd, wq.type);    // fills state.q
        matmul(state.k, wk.data, state.xb, kv_dim,  n_embd, wk.type);   // fills state.k
        matmul(state.v, wv.data, state.xb, kv_dim,  n_embd, wv.type);   // fills state.v

        rope(state.q, state.k, pos,
             static_cast<int>(cfg.head_dim()),
             static_cast<int>(cfg.n_head),
             static_cast<int>(cfg.n_head_kv),
             cfg.rope_freq_base);

        // Stores this token's key and value into the KV cache at position pos in layer l
        // Compares this token's query against every previous token's cached key (dot product)
        // to compute relevance scores.
        // Converts scores to percentages with softmax
        // Blends the cached values using those percentages
        // The result goes into state.xb2 — 2,048 numbers representing "what I learned by looking at all previous tokens."
        // state.att is scratch space the attention function uses internally for the scores.

        attention(state.xb2, state.q, state.k, state.v,
                  state.key_cache, state.value_cache, state.att,
                  l, pos, cfg);

        auto wo = get_weight(model, prefix + "attn_output.weight");
        matmul(state.xb, wo.data, state.xb2, n_embd, n_embd, wo.type);

        vec_add(state.x, state.xb, n_embd);

        // --------------------- FFN (thinking) ---------------------

        const float* ffn_norm_w = get_norm_weight(model, prefix + "ffn_norm.weight");
        rmsnorm(state.xb, state.x, ffn_norm_w, n_embd, cfg.rms_norm_eps);

        auto wgate = get_weight(model, prefix + "ffn_gate.weight");
        auto wup   = get_weight(model, prefix + "ffn_up.weight");
        auto wdown = get_weight(model, prefix + "ffn_down.weight");

        ffn_swiglu(state.xb2, state.xb,
                   state.hb, state.hb2,
                   wgate.data, wup.data, wdown.data,
                   n_ff, n_embd, wgate.type);

        vec_add(state.x, state.xb2, n_embd);
    }
    // After the loop ends, state.x holds the model's full understanding — 2048 numbers encoding everything the model knows about what should come next. But these numbers are in the model's "internal language" — they're meaningful to the model's math but not to us.   
    // After the forward function ends, state.logits holds 32000 human-interpretable scores — one per word in the vocabulary. "the" gets 2.1, "on" gets 8.7, "quietly" gets 5.3, etc.
    // The final norm + matmul is just a translation step. It converts the 2048-number internal representation into 32000 concrete word predictions. 
    // state.x doesn't change in a meaningful way after the loop — it just gets normalized (cleaned up) and then multiplied by the output matrix to produce state.logits. The "thinking" is done. The last step is just reading out the answer.

    // ---- Final Output: Convert hidden state to word predictions ----
    // All 22 layers are done. state.x now holds the final refined 2048-number
    // representation of the current token, enriched with context from all 
    // previous tokens. 
    // 
    // We normalize one last time, then multiply by the output matrix 
    // (32000 × 2048) to produce 32000 scores (logits) — one per word 
    // in the vocabulary. The highest score = the model's best guess 
    // for the next word.
    //
    // TinyLlama uses "tied embeddings": no separate output.weight exists,
    // so we reuse token_embd.weight (the same matrix used to convert 
    // token IDs to vectors at the start).
    const float* final_norm_w = get_norm_weight(model, "output_norm.weight");
    rmsnorm(state.x, state.x, final_norm_w, n_embd, cfg.rms_norm_eps);

    WeightRef w_out;
    bool found_output = false;
    try {
        w_out = get_weight(model, "output.weight");
        found_output = true;
    } catch (...) {}
    if (!found_output) {
        w_out = get_weight(model, "token_embd.weight");
    }

    // By the end, state.x contains the original embedding plus 22 attention contributions plus 22 FFN contributions. 
    // Every layer's work is preserved in that sum.
    matmul(state.logits, w_out.data, state.x,
           static_cast<int>(cfg.n_vocab), n_embd, w_out.type);
}