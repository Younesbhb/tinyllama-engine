#include "gguf_model.h"
#include "run_state.h"
#include "generate.h"
#include "ops.h"

#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>


// Some definitions
// n_embd (2048) — The "width" of the model. Every token gets represented as 2,048 numbers. 
// This vector flows through the entire model.

// n_layers (22) — How many times the token gets processed. Each layer refines the representation.

// n_ff (5632): "feed-forward dimension." The FFN temporarily expands from 2048 to 5632 (bigger workspace to "think" in), then shrinks back.

// n_ctx (2048): "context length." Maximum number of tokens the model can see at once — prompt plus generated text combined.

// n_vocab (32000): "vocabulary size." How many words/tokens the model knows.

// n_layers (22): How many transformer blocks the signal passes through.

// kv_dim (256) — The size of the key and value vectors. Smaller than 2048 because of 
// GQA (4 KV heads × 64 dimensions per head = 256 instead of 32 × 64 = 2048).

// Input embedding matrix (token_embd.weight): Size 32,000 × 2,048. Used at the very start of the forward pass. Converts a token ID into a 2,048-number vector. Row 15043 is the vector for "Hello." We used this in embed_token().

// Output matrix (output.weight): Size 32,000 × 2,048. Used at the very end. Converts the final 2,048-number hidden state into 32,000 scores — one per vocabulary word.

// n_head (32): The number of query heads. The model runs attention 32 times in parallel, each focusing on different aspects of the input. One head might focus on grammar, another on meaning, another on nearby words, etc.

// n_head_kv (4): The number of key/value heads. Instead of each query head having its own key and value (which would need 32 KV heads), multiple query heads share KV heads. 32 query heads share 4 KV heads — that's 8 query heads per KV head. This saves memory.

// kv_dim (256): The total size of the key and value vectors. It's n_head_kv × head_dim = 4 × 64 = 256. This isn't stored in the config — we compute it in the forward pass because it's derived from other values.

// head_dim (64): The size of each head's vector. It's calculated as n_embd / n_head = 2048 / 32 = 64. Each head works on a 64-dimensional slice. All 32 heads together: 32 × 64 = 2048 (the full hidden state).

// kv_dim (256): is the total size of all KV heads combined. It's how many floats make up one position's key (or value) in the cache.

// What is normalizing?
// Imagine you have the numbers [1000, 2000, 3000]. After multiplying through several layers, they might become [5000000, 10000000, 15000000]. After a few more layers, they could overflow to infinity.
// Normalizing scales them back to a reasonable range. RMSNorm takes those giant numbers and rescales them so the average magnitude is around 1.0:

//Do we have a KV cache for each layer? A query cache?
// KV cache: yes, one per layer. The cache stores keys and values for all 22 layers, each with space for 2048 positions × 256 floats.
// Query cache: no. Queries are never cached. A query is only used once — at the moment the token is processed — to compute attention scores against all cached keys. After that, the query is thrown away.


// -------------------- main --------------------

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage:\n"
                      << "  ./engine <model.gguf> \"prompt text\"\n"
                      << "  ./engine <model.gguf>                    (default prompt)\n"
                      << "  ./engine <model.gguf> dump <tensor> [n]  (dump tensor)\n"
                      << "\nOptions:\n"
                      << "  --backend naive   Use naive (unoptimized) implementations\n"
                      << "  --backend neon    Use ARM NEON SIMD implementations (default on ARM)\n";
            return 1;
        }

        // Parse --backend flag (can appear anywhere in args)
        for (int i = 1; i < argc - 1; i++) {
            if (std::strcmp(argv[i], "--backend") == 0) {
                std::string val = argv[i + 1];
                if (val == "naive") {
                    set_backend(Backend::NAIVE);
                } else if (val == "neon") {
                    set_backend(Backend::NEON);
                } else {
                    std::cerr << "Unknown backend: " << val << " (use 'naive' or 'neon')\n";
                    return 1;
                }
                // Shift remaining args to remove --backend and its value
                for (int j = i; j < argc - 2; j++) {
                    argv[j] = argv[j + 2];
                }
                argc -= 2;
                break;
            }
        }

        std::cout << "Backend: " << (get_backend() == Backend::NEON ? "NEON" : "naive") << "\n";

        std::string path = argv[1];
        GGUFModel model(path);

        std::cout << "Loaded " << model.config().architecture
                  << " (" << model.config().n_layers << " layers, "
                  << model.config().n_embd << " dim, "
                  << model.config().n_vocab << " vocab)\n";

        // Handle dump mode (keep existing functionality)
        if (argc >= 4 && std::string(argv[2]) == "dump") {
            std::string name = argv[3];
            std::size_t n = 10;
            if (argc >= 5) {
                n = static_cast<std::size_t>(std::strtoull(argv[4], nullptr, 10));
                if (n == 0) n = 10;
            }
            model.dump_tensor(name, n);
            return 0;
        }

        // Allocate runtime state
        RunState state;
        state.allocate(model.config());

        // Get prompt from command line or use default
        std::string prompt;
        if (argc >= 3) {
            prompt = argv[2];
        } else {
            // TinyLlama chat template format
            prompt = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWhat is the meaning of life?</s>\n<|assistant|>\n";

        }

        std::cout << "Prompt: \"" << prompt << "\"\n";
        std::cout << "Generating...\n\n";

        generate(model, state, prompt);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}