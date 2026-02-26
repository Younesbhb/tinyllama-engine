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

// Is each position aka token represented in 4 different ways per layer, bc we have 4kv heads ?
// Yes. Each KV head captures different aspects of the token at that position. One head might encode syntactic relationships, another semantic meaning, another positional patterns, etc. The 256 floats per position break down as:
// pos 30, layer 5:  [head0: 64 floats] [head1: 64 floats] [head2: 64 floats] [head3: 64 floats]
// And remember, during attention, each KV head is shared by 8 query heads (32 query heads ÷ 4 KV heads = 8). So query heads 0–7 all attend against KV head 0's keys and values, query heads 8–15 against KV head 1, and so on. The 8 query heads within each group ask different questions but search through the same set of keys and values. That's the GQA tradeoff — 8× less memory in the KV cache compared to full multi-head attention, with minimal quality loss.


// -------------------- Helper: parse and remove a flag --------------------
// Scans argv for "--flag value", removes both from argv, returns the value.
// Returns empty string if flag not found.
static std::string consume_flag(int& argc, char** argv, const char* flag) {
    for (int i = 1; i < argc - 1; i++) {
        if (std::strcmp(argv[i], flag) == 0) {
            std::string val = argv[i + 1];
            // Shift remaining args left by 2
            for (int j = i; j < argc - 2; j++) {
                argv[j] = argv[j + 2];
            }
            argc -= 2;
            return val;
        }
    }
    return "";
}

// Scans argv for a bare "--flag" (no value), removes it.
// Returns true if found.
static bool consume_bare_flag(int& argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], flag) == 0) {
            // Shift remaining args left by 1
            for (int j = i; j < argc - 1; j++) {
                argv[j] = argv[j + 1];
            }
            argc -= 1;
            return true;
        }
    }
    return false;
}


// -------------------- Interactive chat loop --------------------
//
// Wraps user input in TinyLlama's chat template and calls generate()
// repeatedly. The KV cache accumulates the entire conversation so the
// model sees full history.
//
// TinyLlama chat template:
//   <|system|>\nYou are a helpful assistant.</s>\n
//   <|user|>\n{message}</s>\n
//   <|assistant|>\n
//
// The system prompt is sent once on the first turn. Each subsequent
// turn only adds the user/assistant markers.

static void chat_loop(GGUFModel& model, RunState& state,
                      int max_tokens, float temperature, float top_p) {
    const auto& cfg = model.config();
    int n_ctx = static_cast<int>(cfg.n_ctx);

    std::cout << "\n";
    std::cout << "=== Interactive Chat ===\n";
    std::cout << "Type your message and press Enter.\n";
    std::cout << "Commands:  /quit  /reset\n";
    std::cout << "\n";

    bool first_turn = true;

    while (true) {
        std::cout << "> ";
        std::string user_input;
        if (!std::getline(std::cin, user_input)) {
            // EOF (Ctrl+D)
            std::cout << "\n";
            break;
        }

        // Trim whitespace
        // The npos check handles the case where the input is entirely whitespace — like the user just hit spacebar a few times and pressed Enter
        // find_first_not_of returns npos (a special "not found" value) because there's no non-whitespace character.
        size_t start = user_input.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) continue;  // empty line
        user_input = user_input.substr(start);

        // Handle commands
        if (user_input == "/quit" || user_input == "/exit") {
            break;
        }
        if (user_input == "/reset") {
            state.reset(cfg);
            first_turn = true;
            std::cout << "[conversation reset]\n\n";
            continue;
        }

        // Build the prompt for this turn
        std::string prompt;
        if (first_turn) {
            // First turn: include system prompt
            prompt = "<|system|>\nYou are a helpful assistant.</s>\n"
                     "<|user|>\n" + user_input + "</s>\n"
                     "<|assistant|>\n";
            first_turn = false;
        } else {
            // Subsequent turns: just user + assistant markers
            prompt = "<|user|>\n" + user_input + "</s>\n"
                     "<|assistant|>\n";
        }

        // Generate response
        std::cout << "\n";
        generate(model, state, prompt, max_tokens, temperature, top_p);

        // Display KV cache usage
        std::cout << "\n[KV cache: " << state.pos << " / " << n_ctx << " tokens]\n\n";

        // Check if context is nearly full (less than 100 tokens remaining)
        int remaining = n_ctx - state.pos;
        if (remaining < 100) {
            std::cout << "[warning: only " << remaining
                      << " tokens remaining — type /reset to start a new conversation]\n\n";
        }
    }
}

// -------------------- main --------------------

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage:\n"
                      << "  ./engine <model.gguf> \"prompt\"       (single-shot generation)\n"
                      << "  ./engine <model.gguf> --chat           (interactive chat)\n"
                      << "  ./engine <model.gguf> dump <tensor>    (dump tensor values)\n"
                      << "\nOptions:\n"
                      << "  --backend naive|neon    Compute backend (default: neon on ARM)\n"
                      << "  --threads N             Number of threads (default: 1)\n"
                      << "  --temp T                Temperature (default: 0.7)\n"
                      << "  --top-p P               Nucleus sampling threshold (default: 0.9)\n"
                      << "  --max-tokens N          Max tokens to generate (default: 2048)\n";
            return 1;
        }

        // ---- Parse flags (order-independent, consumed from argv) ----

        // --backend
        std::string backend_val = consume_flag(argc, argv, "--backend");
        if (!backend_val.empty()) {
            if (backend_val == "naive") {
                set_backend(Backend::NAIVE);
            } else if (backend_val == "neon") {
                set_backend(Backend::NEON);
            } else {
                std::cerr << "Unknown backend: " << backend_val << " (use 'naive' or 'neon')\n";
                return 1;
            }
        }

        // --threads
        std::string threads_val = consume_flag(argc, argv, "--threads");
        if (!threads_val.empty()) {
            int n = std::atoi(threads_val.c_str());
            if (n < 1) n = 1;
            set_num_threads(n);
        }

        // --temp
        float temperature = 0.7f;
        std::string temp_val = consume_flag(argc, argv, "--temp");
        if (!temp_val.empty()) {
            temperature = std::atof(temp_val.c_str());
        }

        // --top-p
        float top_p = 0.9f;
        std::string top_p_val = consume_flag(argc, argv, "--top-p");
        if (!top_p_val.empty()) {
            top_p = std::atof(top_p_val.c_str());
        }

        // --max-tokens
        int max_tokens = 2048;
        std::string max_tok_val = consume_flag(argc, argv, "--max-tokens");
        if (!max_tok_val.empty()) {
            max_tokens = std::atoi(max_tok_val.c_str());
            if (max_tokens < 1) max_tokens = 2048;
        }

        // --chat (bare flag, no value)
        bool chat_mode = consume_bare_flag(argc, argv, "--chat");

        // ---- Load model ----
        std::cout << "Backend: " << (get_backend() == Backend::NEON ? "NEON" : "naive")
                  << " | Threads: " << get_num_threads() << "\n";

        std::string path = argv[1];
        GGUFModel model(path);

        std::cout << "Loaded " << model.config().architecture
                  << " (" << model.config().n_layers << " layers, "
                  << model.config().n_embd << " dim, "
                  << model.config().n_vocab << " vocab)\n";

        // Handle dump mode
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

        // ---- Chat mode ----
        if (chat_mode) {
            chat_loop(model, state, max_tokens, temperature, top_p);
            return 0;
        }

        // ---- Single-shot mode ----
        std::string prompt;
        if (argc >= 3) {
            prompt = argv[2];
        } else {
            prompt = "<|system|>\nYou are a helpful assistant.</s>\n"
                     "<|user|>\nWhat is the meaning of life?</s>\n"
                     "<|assistant|>\n";
        }

        std::cout << "Prompt: \"" << prompt << "\"\n";
        std::cout << "Generating...\n\n";

        generate(model, state, prompt, max_tokens, temperature, top_p);

        std::cout << "\n[KV cache: " << state.pos << " / "
                  << model.config().n_ctx << " tokens]\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}