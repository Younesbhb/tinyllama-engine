#include "gguf_model.h"
#include "run_state.h"
#include "forward.h"
#include "ops.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <numeric>
#include <algorithm>
#include <cmath>

// -------------------- Benchmark Harness --------------------
//
// Measures two phases of LLM inference independently:
//
//   1. Prefill  — Processing the prompt tokens (fills KV cache).
//      Metric: tokens/sec (throughput). Higher is better.
//      This is mostly matmul-bound and benefits from NEON + threads.
//
//   2. Decode   — Generating new tokens one at a time.
//      Metric: ms/token and tokens/sec. Lower ms is better.
//      This is memory-bandwidth-bound (loading weights from RAM).
//
// Each measurement is repeated multiple times and we report
// median, mean, min, max, and stddev to catch variance.
//
// Usage:
//   ./benchmark <model.gguf> [options]
//
// Options:
//   --backend naive|neon       Force backend (default: neon on ARM)
//   --threads N                Number of threads (default: 1)
//   --warmup N                 Warmup iterations (default: 2)
//   --decode-tokens N          Tokens to generate per trial (default: 20)
//   --trials N                 Number of timed trials (default: 5)
//   --prompt "text"            Custom prompt (default: chat template)
//   --csv                      Output results as CSV row (for scripting)
//   --label "text"             Label for CSV output

struct BenchConfig {
    std::string model_path;
    std::string prompt;
    std::string label;
    int warmup_iters   = 2;
    int decode_tokens  = 20;
    int trials         = 5;
    bool csv_output    = false;
};

// -------------------- Statistics --------------------

struct Stats {
    double mean;
    double median;
    double min_val;
    double max_val;
    double stddev;
};

Stats compute_stats(std::vector<double>& values) {
    Stats s{};
    if (values.empty()) return s;

    std::sort(values.begin(), values.end());
    s.min_val = values.front();
    s.max_val = values.back();

    // Calculate average
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    s.mean = sum / static_cast<double>(values.size());

    size_t n = values.size();
    if (n % 2 == 0) {
        s.median = (values[n/2 - 1] + values[n/2]) / 2.0;
    } else {
        s.median = values[n/2];
    }

    double sq_sum = 0.0;
    for (double v : values) {
        double diff = v - s.mean;
        sq_sum += diff * diff;
    }
    s.stddev = std::sqrt(sq_sum / static_cast<double>(n));

    return s;
}

// -------------------- Timer Helper --------------------

using Clock = std::chrono::high_resolution_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// -------------------- Benchmark Core --------------------
//
// Runs a single trial: prefill the prompt, then decode N tokens.
// Returns {prefill_ms, decode_total_ms, num_prompt_tokens, num_decode_tokens}.

struct TrialResult {
    double prefill_ms;
    double decode_ms;
    int    prompt_tokens;
    int    decode_tokens;
};

TrialResult run_trial(GGUFModel& model, const BenchConfig& cfg) {
    const auto& mcfg = model.config();
    const auto& tok  = model.tokenizer();
    int n_vocab = static_cast<int>(mcfg.n_vocab);

    // Allocate fresh state each trial (clean KV cache)
    RunState state;
    state.allocate(mcfg);

    // Encode prompt
    std::vector<uint32_t> prompt_tokens = tok.encode(cfg.prompt);
    prompt_tokens.insert(prompt_tokens.begin(), tok.bos_token());
    int n_prompt = static_cast<int>(prompt_tokens.size());

    // ---- Prefill: process all prompt tokens ----
    state.pos = 0;
    auto t_prefill_start = Clock::now();

    for (int i = 0; i < n_prompt; i++) {
        forward(model, state, static_cast<int>(prompt_tokens[i]), state.pos);
        state.pos++;
    }

    auto t_prefill_end = Clock::now();

    // Sample first token (not timed as part of prefill or decode)
    int next_token = sample_argmax(state.logits, n_vocab);

    // ---- Decode: generate tokens one at a time ----
    int decoded = 0;
    auto t_decode_start = Clock::now();

    for (int i = 0; i < cfg.decode_tokens; i++) {
        forward(model, state, next_token, state.pos);
        state.pos++;

        next_token = sample_argmax(state.logits, n_vocab);
        decoded++;

        // Stop at EOS or context limit
        if (static_cast<uint32_t>(next_token) == tok.eos_token()) break;
        if (state.pos >= static_cast<int>(mcfg.n_ctx)) break;
    }

    auto t_decode_end = Clock::now();

    return {
        elapsed_ms(t_prefill_start, t_prefill_end),
        elapsed_ms(t_decode_start, t_decode_end),
        n_prompt,
        decoded
    };
}

// -------------------- Print Results --------------------

void print_results(const BenchConfig& cfg,
                   const std::string& backend_name,
                   int threads,
                   const std::string& quant_type,
                   int prompt_tokens,
                   Stats& prefill_stats,
                   Stats& decode_stats,
                   int decode_tokens) {

    if (cfg.csv_output) {
        // CSV header (printed once by the shell script):
        // label,backend,threads,quant,prompt_toks,decode_toks,
        // prefill_median_ms,prefill_tok_s,decode_median_ms_tok,decode_tok_s

        double prefill_tok_s = static_cast<double>(prompt_tokens) /
                               (prefill_stats.median / 1000.0);
        double decode_ms_tok = decode_stats.median / static_cast<double>(decode_tokens);
        double decode_tok_s  = static_cast<double>(decode_tokens) /
                               (decode_stats.median / 1000.0);

        std::cout << cfg.label << ","
                  << backend_name << ","
                  << threads << ","
                  << quant_type << ","
                  << prompt_tokens << ","
                  << decode_tokens << ","
                  << std::fixed << std::setprecision(1)
                  << prefill_stats.median << ","
                  << std::setprecision(1) << prefill_tok_s << ","
                  << std::setprecision(2) << decode_ms_tok << ","
                  << std::setprecision(1) << decode_tok_s
                  << "\n";
        return;
    }

    // Human-readable output
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║  BENCHMARK RESULTS";
    if (!cfg.label.empty()) {
        std::cout << " — " << cfg.label;
    }
    std::cout << "\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Backend:      " << backend_name << "\n";
    std::cout << "║  Threads:      " << threads << "\n";
    std::cout << "║  Quant:        " << quant_type << "\n";
    std::cout << "║  Trials:       " << cfg.trials << "\n";
    std::cout << "║  Prompt toks:  " << prompt_tokens << "\n";
    std::cout << "║  Decode toks:  " << decode_tokens << "\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";

    // Prefill results
    double prefill_tok_s = static_cast<double>(prompt_tokens) /
                           (prefill_stats.median / 1000.0);
    std::cout << "║  PREFILL\n";
    std::cout << std::fixed;
    std::cout << "║    Median:     " << std::setprecision(1) << prefill_stats.median << " ms"
              << "  (" << std::setprecision(1) << prefill_tok_s << " tok/s)\n";
    std::cout << "║    Mean:       " << std::setprecision(1) << prefill_stats.mean << " ms"
              << "  ± " << std::setprecision(1) << prefill_stats.stddev << " ms\n";
    std::cout << "║    Range:      [" << std::setprecision(1) << prefill_stats.min_val
              << " — " << std::setprecision(1) << prefill_stats.max_val << "] ms\n";

    // Decode results
    double decode_ms_tok = decode_stats.median / static_cast<double>(decode_tokens);
    double decode_tok_s  = static_cast<double>(decode_tokens) /
                           (decode_stats.median / 1000.0);
    std::cout << "║  DECODE\n";
    std::cout << "║    Median:     " << std::setprecision(1) << decode_stats.median << " ms total"
              << "  (" << std::setprecision(2) << decode_ms_tok << " ms/tok)\n";
    std::cout << "║    Throughput: " << std::setprecision(1) << decode_tok_s << " tok/s\n";
    std::cout << "║    Mean:       " << std::setprecision(1) << decode_stats.mean << " ms"
              << "  ± " << std::setprecision(1) << decode_stats.stddev << " ms\n";
    std::cout << "║    Range:      [" << std::setprecision(1) << decode_stats.min_val
              << " — " << std::setprecision(1) << decode_stats.max_val << "] ms\n";

    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
}

// -------------------- Quant Type Name --------------------

std::string detect_quant_type(GGUFModel& model) {
    // Check the type of a representative weight tensor (first layer Q weight)
    try {
        const auto& t = model.tensor_info("blk.0.attn_q.weight");
        switch (t.type) {
            case GGML_TYPE_F32:  return "F32";
            case GGML_TYPE_F16:  return "F16";
            case GGML_TYPE_Q8_0: return "Q8_0";
            case GGML_TYPE_Q4_0: return "Q4_0";
            case GGML_TYPE_Q6_K: return "Q6_K";
            default:             return "unknown";
        }
    } catch (...) {
        return "unknown";
    }
}

// -------------------- Main --------------------

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: ./benchmark <model.gguf> [options]\n"
                      << "\nOptions:\n"
                      << "  --backend naive|neon   Backend (default: neon on ARM)\n"
                      << "  --threads N            Thread count (default: 1)\n"
                      << "  --warmup N             Warmup iterations (default: 2)\n"
                      << "  --decode-tokens N      Tokens to decode per trial (default: 20)\n"
                      << "  --trials N             Number of timed trials (default: 5)\n"
                      << "  --prompt \"text\"       Custom prompt\n"
                      << "  --csv                  CSV output mode\n"
                      << "  --label \"text\"        Label for CSV row\n";
            return 1;
        }

        BenchConfig cfg;
        cfg.model_path = argv[1];

        // Default prompt (TinyLlama chat template)
        cfg.prompt = "<|system|>\nYou are a helpful assistant.</s>\n"
                     "<|user|>\nWhat is the meaning of life</s>\n"
                     "<|assistant|>\n";

        // Parse options
        std::string backend_str;
        int thread_count = 1;

        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--backend" && i + 1 < argc) {
                backend_str = argv[++i];
            } else if (arg == "--threads" && i + 1 < argc) {
                thread_count = std::atoi(argv[++i]);
                if (thread_count < 1) thread_count = 1;
            } else if (arg == "--warmup" && i + 1 < argc) {
                cfg.warmup_iters = std::atoi(argv[++i]);
            } else if (arg == "--decode-tokens" && i + 1 < argc) {
                cfg.decode_tokens = std::atoi(argv[++i]);
            } else if (arg == "--trials" && i + 1 < argc) {
                cfg.trials = std::atoi(argv[++i]);
            } else if (arg == "--prompt" && i + 1 < argc) {
                cfg.prompt = argv[++i];
            } else if (arg == "--csv") {
                cfg.csv_output = true;
            } else if (arg == "--label" && i + 1 < argc) {
                cfg.label = argv[++i];
            }
        }

        // Set backend
        if (backend_str == "naive") {
            set_backend(Backend::NAIVE);
        } else if (backend_str == "neon") {
            set_backend(Backend::NEON);
        }
        // else: keep default (NEON on ARM, NAIVE on x86)

        set_num_threads(thread_count);

        std::string backend_name = (get_backend() == Backend::NEON) ? "NEON" : "naive";

        // Load model
        if (!cfg.csv_output) {
            std::cerr << "Loading model: " << cfg.model_path << "\n";
        }
        GGUFModel model(cfg.model_path);

        std::string quant_type = detect_quant_type(model);

        if (!cfg.csv_output) {
            std::cerr << "Model: " << model.config().architecture
                      << " (" << model.config().n_layers << "L, "
                      << model.config().n_embd << "D, "
                      << quant_type << ")\n";
            std::cerr << "Config: backend=" << backend_name
                      << " threads=" << thread_count
                      << " warmup=" << cfg.warmup_iters
                      << " trials=" << cfg.trials
                      << " decode_tokens=" << cfg.decode_tokens << "\n";
        }

        // ---- Warmup ----
        if (!cfg.csv_output) {
            std::cerr << "Warming up (" << cfg.warmup_iters << " iterations)...\n";
        }
        for (int w = 0; w < cfg.warmup_iters; w++) {
            run_trial(model, cfg);
        }

        // ---- Timed trials ----
        if (!cfg.csv_output) {
            std::cerr << "Running " << cfg.trials << " trials...\n";
        }

        std::vector<double> prefill_times;
        std::vector<double> decode_times;
        int prompt_tokens = 0;
        int actual_decode_tokens = 0;

        for (int t = 0; t < cfg.trials; t++) {
            TrialResult result = run_trial(model, cfg);
            prefill_times.push_back(result.prefill_ms);
            decode_times.push_back(result.decode_ms);
            // We seem to be only using the prompt_tokens and actual_decode_tokens of the last run_trial
            prompt_tokens = result.prompt_tokens;
            actual_decode_tokens = result.decode_tokens;

            if (!cfg.csv_output) {
                std::cerr << "  Trial " << (t + 1) << "/" << cfg.trials
                          << ": prefill=" << std::fixed << std::setprecision(1)
                          << result.prefill_ms << "ms"
                          << "  decode=" << result.decode_ms << "ms"
                          << " (" << result.decode_tokens << " tokens)\n";
            }
        }

        // ---- Compute and print stats ----
        Stats prefill_stats = compute_stats(prefill_times);
        Stats decode_stats  = compute_stats(decode_times);

        print_results(cfg, backend_name, thread_count, quant_type,
                      prompt_tokens, prefill_stats, decode_stats,
                      actual_decode_tokens);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}