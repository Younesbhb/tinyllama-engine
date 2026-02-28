#include "generate.h"

#include "gguf_model.h"
#include "run_state.h"
#include "forward.h"
#include "ops.h"          // sample_argmax, sample_top_p

#include <iostream>
#include <vector>
#include <cstdint>




// -------------------- Generation Loop --------------------
//
// This is where the engine comes alive. The loop:
//   1. Encode the prompt into token IDs
//   2. Prepend BOS token
//   3. Run forward pass for each prompt token (fills KV cache)
//   4. Sample next token from the logits
//   5. Feed that token back in, run forward again
//   6. Repeat until EOS or max length
//   7. Print each generated token as it appears
//
// Prompt processing (steps 2-3) is called "prefill" — we're not
// generating yet, just letting the model "read" the prompt.
//
// Generation (steps 4-6) is called "decode" — one new token per
// forward pass.

void generate(GGUFModel& model, RunState& state,
                     const std::string& prompt,
                     int max_tokens,
                     float temperature,
                     float top_p) {

    const auto& cfg = model.config();
    const auto& tok = model.tokenizer();
    int n_vocab = static_cast<int>(cfg.n_vocab);

    // ---- Step 1: Encode the prompt ----
    std::vector<uint32_t> prompt_tokens = tok.encode(prompt);

    // Prepend BOS token (the model expects this at the start)
    prompt_tokens.insert(prompt_tokens.begin(), tok.bos_token());

    std::cout << "Prompt tokens: " << prompt_tokens.size() << " [";
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        std::cout << prompt_tokens[i];
        if (i + 1 < prompt_tokens.size()) std::cout << ", ";
    }
    std::cout << "]\n\n";

    // ---- Step 2: Prefill — process prompt tokens ----
    // Run forward pass for each prompt token to fill KV cache.
    // We only care about the logits after the LAST prompt token.
    state.pos = 0;
    int next_token = 0;

    // Let the model read the prompt
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        forward(model, state, static_cast<int>(prompt_tokens[i]), state.pos);
        state.pos++;
    }

    // Sample first generated token from logits after prefill
    if (temperature <= 0.0f) {
        next_token = sample_argmax(state.logits, n_vocab);
    } else {
        next_token = sample_top_p(state.logits, n_vocab, temperature, top_p);
    }

    // Print the first generated token
    // flush() forces the output to appear on screen right away instead of waiting in a buffer. 
    // This is what creates the "streaming" effect where you see words appear one by one.
    std::string token_str = tok.decode_stripped({static_cast<uint32_t>(next_token)});
    std::cout << token_str;
    std::cout.flush();  // Print immediately, don't buffer

    // ---- Step 3: Generate — one token at a time ----
    for (int i = 1; i < max_tokens; i++) {
        forward(model, state, next_token, state.pos);
        state.pos++;

        // Sample next token
        if (temperature <= 0.0f) {
            next_token = sample_argmax(state.logits, n_vocab);
        } else {
            next_token = sample_top_p(state.logits, n_vocab, temperature, top_p);
        }

        // Check for end of sequence
        if (static_cast<uint32_t>(next_token) == tok.eos_token()) {
            break;
        }

        // Check context window limit
        if (state.pos >= static_cast<int>(cfg.n_ctx)) {
            std::cout << "\n[reached max context length]\n";
            break;
        }

        // Print the token
        token_str = tok.decode({static_cast<uint32_t>(next_token)});
        std::cout << token_str;
        std::cout.flush();
    }

    std::cout << "\n";
}