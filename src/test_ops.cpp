#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include "ops.h"
#include "gguf.h"

// -------------------- Test Helpers --------------------

static int tests_passed = 0;
static int tests_failed = 0;

static bool approx_equal(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) < tol;
}

static void check(const char* name, bool condition) {
    if (condition) {
        std::cout << "  ✓ " << name << "\n";
        tests_passed++;
    } else {
        std::cout << "  ✗ " << name << " FAILED\n";
        tests_failed++;
    }
}

static void print_vec(const char* label, const float* v, int n) {
    std::cout << "    " << label << ": [";
    for (int i = 0; i < n; i++) {
        std::cout << std::setprecision(6) << v[i];
        if (i + 1 < n) std::cout << ", ";
    }
    std::cout << "]\n";
}

// -------------------- Tests --------------------

void test_matmul_f32() {
    std::cout << "\n=== matmul_f32 ===\n";

    // Simple 2x3 matrix times 3-vector
    // W = [[1, 2, 3],    x = [1, 1, 1]
    //      [4, 5, 6]]
    // out = [1+2+3, 4+5+6] = [6, 15]
    float W[] = {1, 2, 3, 4, 5, 6};
    float x[] = {1, 1, 1};
    float out[2] = {0};

    matmul(out, W, x, 2, 3, GGML_TYPE_F32);
    print_vec("out", out, 2);
    check("row 0 = 6.0", approx_equal(out[0], 6.0f));
    check("row 1 = 15.0", approx_equal(out[1], 15.0f));

    // Identity matrix test: I * x = x
    float I4[] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float x4[] = {3.14f, 2.71f, 1.41f, 1.73f};
    float out4[4] = {0};

    matmul(out4, I4, x4, 4, 4, GGML_TYPE_F32);
    bool identity_ok = true;
    for (int i = 0; i < 4; i++) {
        if (!approx_equal(out4[i], x4[i])) identity_ok = false;
    }
    check("identity matrix: I*x = x", identity_ok);
}

void test_matmul_f16() {
    std::cout << "\n=== matmul_f16 ===\n";

    // Convert known F32 values to F16 bit patterns, then test matmul
    // We'll use a simple 2x2 case: W = [[2, 0], [0, 3]], x = [5, 7]
    // Expected: out = [10, 21]

    // F16 bit patterns for common values:
    // 2.0 in F16 = 0x4000
    // 3.0 in F16 = 0x4200
    // 0.0 in F16 = 0x0000
    std::uint16_t W_f16[] = {0x4000, 0x0000, 0x0000, 0x4200};
    float x[] = {5.0f, 7.0f};
    float out[2] = {0};

    matmul(out, W_f16, x, 2, 2, GGML_TYPE_F16);
    print_vec("out", out, 2);
    check("F16 matmul row 0 = 10.0", approx_equal(out[0], 10.0f));
    check("F16 matmul row 1 = 21.0", approx_equal(out[1], 21.0f));
}

void test_rmsnorm() {
    std::cout << "\n=== rmsnorm ===\n";

    // Input: [1, 1, 1, 1], weight: [1, 1, 1, 1], eps = 1e-5
    // mean(x^2) = (1+1+1+1)/4 = 1.0
    // rms = 1/sqrt(1.0 + 1e-5) ≈ 1.0
    // out = x * rms * weight ≈ [1, 1, 1, 1]
    float x1[] = {1, 1, 1, 1};
    float w1[] = {1, 1, 1, 1};
    float out1[4];

    rmsnorm(out1, x1, w1, 4, 1e-5f);
    print_vec("uniform input", out1, 4);
    check("uniform input ≈ 1.0", approx_equal(out1[0], 1.0f, 1e-3f));

    // Input: [3, 4], weight: [1, 1], eps = 0
    // mean(x^2) = (9+16)/2 = 12.5
    // rms = 1/sqrt(12.5) = 0.28284...
    // out = [3*0.28284, 4*0.28284] = [0.8485, 1.1314]
    float x2[] = {3.0f, 4.0f};
    float w2[] = {1.0f, 1.0f};
    float out2[2];

    rmsnorm(out2, x2, w2, 2, 0.0f);
    print_vec("[3,4] normalized", out2, 2);
    float expected_rms = 1.0f / std::sqrt(12.5f);
    check("[3,4] element 0", approx_equal(out2[0], 3.0f * expected_rms));
    check("[3,4] element 1", approx_equal(out2[1], 4.0f * expected_rms));

    // Verify that weight scaling works
    float w3[] = {2.0f, 0.5f};
    float out3[2];
    rmsnorm(out3, x2, w3, 2, 0.0f);
    print_vec("[3,4] with weights [2, 0.5]", out3, 2);
    check("weight scaling elem 0", approx_equal(out3[0], 3.0f * expected_rms * 2.0f));
    check("weight scaling elem 1", approx_equal(out3[1], 4.0f * expected_rms * 0.5f));
}

void test_softmax() {
    std::cout << "\n=== softmax ===\n";

    // Equal logits → uniform distribution
    float x1[] = {1.0f, 1.0f, 1.0f, 1.0f};
    softmax(x1, 4);
    print_vec("uniform logits", x1, 4);
    check("uniform → 0.25 each", approx_equal(x1[0], 0.25f));

    // One dominant logit
    float x2[] = {10.0f, 0.0f, 0.0f};
    softmax(x2, 3);
    print_vec("dominant logit", x2, 3);
    check("dominant ≈ 1.0", x2[0] > 0.99f);
    check("others ≈ 0.0", x2[1] < 0.01f && x2[2] < 0.01f);

    // Probabilities sum to 1
    float x3[] = {2.0f, 1.0f, 0.1f, -1.0f, 3.0f};
    softmax(x3, 5);
    float sum = 0;
    for (int i = 0; i < 5; i++) sum += x3[i];
    check("probabilities sum to 1.0", approx_equal(sum, 1.0f, 1e-5f));

    // Numerical stability: large values shouldn't produce NaN/Inf
    float x4[] = {1000.0f, 1000.0f, 999.0f};
    softmax(x4, 3);
    bool no_nan = true;
    for (int i = 0; i < 3; i++) {
        if (std::isnan(x4[i]) || std::isinf(x4[i])) no_nan = false;
    }
    print_vec("large logits", x4, 3);
    check("large logits: no NaN/Inf", no_nan);
}

void test_silu() {
    std::cout << "\n=== silu ===\n";

    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    float x1[] = {0.0f};
    silu(x1, 1);
    check("silu(0) = 0", approx_equal(x1[0], 0.0f));

    // silu(x) for large positive x ≈ x (sigmoid → 1)
    float x2[] = {10.0f};
    silu(x2, 1);
    check("silu(10) ≈ 10", approx_equal(x2[0], 10.0f, 0.01f));

    // silu(x) for large negative x ≈ 0 (sigmoid → 0)
    float x3[] = {-10.0f};
    silu(x3, 1);
    check("silu(-10) ≈ 0", approx_equal(x3[0], 0.0f, 0.01f));

    // silu(1) = 1 * sigmoid(1) = 1/(1+e^-1) ≈ 0.7311
    float x4[] = {1.0f};
    silu(x4, 1);
    float expected = 1.0f / (1.0f + std::exp(-1.0f));
    check("silu(1) ≈ 0.7311", approx_equal(x4[0], expected));
}

void test_elementwise_mul() {
    std::cout << "\n=== elementwise_mul ===\n";

    float a[] = {2.0f, 3.0f, 4.0f};
    float b[] = {5.0f, 0.5f, -1.0f};
    elementwise_mul(a, b, 3);
    print_vec("result", a, 3);
    check("[2,3,4] ⊙ [5,0.5,-1] = [10, 1.5, -4]",
          approx_equal(a[0], 10.0f) &&
          approx_equal(a[1], 1.5f) &&
          approx_equal(a[2], -4.0f));
}

void test_vec_add() {
    std::cout << "\n=== vec_add ===\n";

    float out[] = {1.0f, 2.0f, 3.0f};
    float x[] = {10.0f, 20.0f, 30.0f};
    vec_add(out, x, 3);
    print_vec("result", out, 3);
    check("[1,2,3] + [10,20,30] = [11, 22, 33]",
          approx_equal(out[0], 11.0f) &&
          approx_equal(out[1], 22.0f) &&
          approx_equal(out[2], 33.0f));
}

void test_ffn_swiglu() {
    std::cout << "\n=== ffn_swiglu ===\n";

    // Use small dimensions to verify by hand
    // n_embd = 2, n_ff = 3
    int n_embd = 2;
    int n_ff = 3;

    float input[] = {1.0f, 2.0f};
    float hb[3] = {0};
    float hb2[3] = {0};
    float out[2] = {0};

    // W_gate [3x2]: [[1,0],[0,1],[1,1]]
    // W_up   [3x2]: [[1,1],[0,1],[1,0]]
    // W_down [2x3]: [[1,0,1],[0,1,0]]
    float W_gate[] = {1,0, 0,1, 1,1};
    float W_up[]   = {1,1, 0,1, 1,0};
    float W_down[] = {1,0,1, 0,1,0};

    // Trace through manually:
    // Step 1: gate = W_gate * [1,2] = [1*1+0*2, 0*1+1*2, 1*1+1*2] = [1, 2, 3]
    // Step 2: up   = W_up   * [1,2] = [1*1+1*2, 0*1+1*2, 1*1+0*2] = [3, 2, 1]
    // Step 3: silu(gate) = [silu(1), silu(2), silu(3)]
    //         silu(1) = 1*sigmoid(1) = 0.7311
    //         silu(2) = 2*sigmoid(2) = 1.7616
    //         silu(3) = 3*sigmoid(3) = 2.8577
    // Step 4: gate ⊙ up = [0.7311*3, 1.7616*2, 2.8577*1]
    //                    = [2.1933, 3.5232, 2.8577]
    // Step 5: out = W_down * [2.1933, 3.5232, 2.8577]
    //         out[0] = 1*2.1933 + 0*3.5232 + 1*2.8577 = 5.0510
    //         out[1] = 0*2.1933 + 1*3.5232 + 0*2.8577 = 3.5232

    ffn_swiglu(out, input, hb, hb2, W_gate, W_up, W_down,
               n_ff, n_embd, GGML_TYPE_F32);

    float silu_1 = 1.0f / (1.0f + std::exp(-1.0f));        // 0.7311
    float silu_2 = 2.0f / (1.0f + std::exp(-2.0f));        // 1.7616
    float silu_3 = 3.0f / (1.0f + std::exp(-3.0f));        // 2.8577

    float expected_0 = silu_1 * 3.0f + silu_3 * 1.0f;      // 5.0510
    float expected_1 = silu_2 * 2.0f;                       // 3.5232

    print_vec("out", out, 2);
    std::cout << "    expected: [" << expected_0 << ", " << expected_1 << "]\n";
    check("FFN output[0] correct", approx_equal(out[0], expected_0, 1e-3f));
    check("FFN output[1] correct", approx_equal(out[1], expected_1, 1e-3f));

    // --- Test 2: Zero input → zero output ---
    // silu(0) = 0, so gate is all zeros, so gate ⊙ up = all zeros
    float zero_input[] = {0.0f, 0.0f};
    ffn_swiglu(out, zero_input, hb, hb2, W_gate, W_up, W_down,
               n_ff, n_embd, GGML_TYPE_F32);
    check("zero input → zero output", approx_equal(out[0], 0.0f) && approx_equal(out[1], 0.0f));

    // --- Test 3: No NaN/Inf with realistic dimensions ---
    {
        int big_embd = 64;
        int big_ff = 128;
        std::vector<float> big_input(big_embd, 0.5f);
        std::vector<float> big_hb(big_ff);
        std::vector<float> big_hb2(big_ff);
        std::vector<float> big_out(big_embd);
        std::vector<float> big_gate(big_ff * big_embd, 0.01f);
        std::vector<float> big_up(big_ff * big_embd, 0.01f);
        std::vector<float> big_down(big_embd * big_ff, 0.01f);

        ffn_swiglu(big_out.data(), big_input.data(),
                   big_hb.data(), big_hb2.data(),
                   big_gate.data(), big_up.data(), big_down.data(),
                   big_ff, big_embd, GGML_TYPE_F32);

        bool no_nan = true;
        for (int i = 0; i < big_embd; i++) {
            if (std::isnan(big_out[i]) || std::isinf(big_out[i])) { no_nan = false; break; }
        }
        check("larger dimensions: no NaN/Inf", no_nan);
    }
}

void test_attention() {
    std::cout << "\n=== attention ===\n";

    // Use TinyLlama dimensions
    llama_config_t cfg;
    cfg.n_layers = 2;       // just 2 layers for testing
    cfg.n_embd = 2048;
    cfg.n_head = 32;
    cfg.n_head_kv = 4;
    cfg.n_ff = 5632;
    cfg.n_vocab = 32000;
    cfg.n_ctx = 64;         // small context for testing

    int head_dim = static_cast<int>(cfg.head_dim());  // 64
    int kv_dim = cfg.n_head_kv * head_dim;             // 256
    int n_embd = static_cast<int>(cfg.n_embd);         // 2048

    // Allocate buffers
    std::vector<float> out(n_embd, 0.0f);
    std::vector<float> q(n_embd, 0.0f);
    std::vector<float> k(kv_dim, 0.0f);
    std::vector<float> v(kv_dim, 0.0f);
    std::vector<float> att(cfg.n_ctx, 0.0f);

    size_t cache_size = cfg.n_layers * cfg.n_ctx * kv_dim;
    std::vector<float> key_cache(cache_size, 0.0f);
    std::vector<float> value_cache(cache_size, 0.0f);

    // --- Test 1: Single token (pos=0) → attention to self must be 100% ---
    // When there's only one token, softmax over a single element = 1.0
    // So the output should be exactly the value vector.
    {
        // Set q to some values
        for (int i = 0; i < n_embd; i++) q[i] = 1.0f;
        // Set k to some values
        for (int i = 0; i < kv_dim; i++) k[i] = 1.0f;
        // Set v to a recognizable pattern: v[i] = i for each head
        for (int i = 0; i < kv_dim; i++) v[i] = static_cast<float>(i % head_dim);

        attention(out.data(), q.data(), k.data(), v.data(),
                  key_cache.data(), value_cache.data(), att.data(),
                  0, 0, cfg);

        // Head 0 should output exactly v[0..63] since it's the only token
        // Head 0 uses KV head 0, and v for KV head 0 is v[0..63]
        bool self_attention_ok = true;
        for (int d = 0; d < head_dim; d++) {
            float expected = static_cast<float>(d);  // v[d] = d
            if (!approx_equal(out[d], expected, 1e-3f)) {
                self_attention_ok = false;
                std::cout << "    mismatch at d=" << d << ": got " << out[d]
                          << " expected " << expected << "\n";
                break;
            }
        }
        check("single token: output = value (100% self-attention)", self_attention_ok);
    }

    // --- Test 2: Two tokens with identical keys → uniform attention (50/50) ---
    {
        // Reset caches
        std::fill(key_cache.begin(), key_cache.end(), 0.0f);
        std::fill(value_cache.begin(), value_cache.end(), 0.0f);

        // Token 0: v = all 1.0
        for (int i = 0; i < kv_dim; i++) { k[i] = 1.0f; v[i] = 1.0f; }
        for (int i = 0; i < n_embd; i++) q[i] = 1.0f;
        attention(out.data(), q.data(), k.data(), v.data(),
                  key_cache.data(), value_cache.data(), att.data(),
                  0, 0, cfg);

        // Token 1: same k (so scores are equal), v = all 3.0
        for (int i = 0; i < kv_dim; i++) { k[i] = 1.0f; v[i] = 3.0f; }
        for (int i = 0; i < n_embd; i++) q[i] = 1.0f;
        attention(out.data(), q.data(), k.data(), v.data(),
                  key_cache.data(), value_cache.data(), att.data(),
                  0, 1, cfg);

        // With equal keys, attention is 50/50
        // Output should be 0.5 * 1.0 + 0.5 * 3.0 = 2.0 for head 0
        check("uniform attention: 50/50 blend → 2.0", approx_equal(out[0], 2.0f, 0.01f));
    }

    // --- Test 3: Output has no NaN or Inf ---
    {
        std::fill(key_cache.begin(), key_cache.end(), 0.0f);
        std::fill(value_cache.begin(), value_cache.end(), 0.0f);

        // Process 10 tokens with varied values
        for (int p = 0; p < 10; p++) {
            for (int i = 0; i < n_embd; i++) q[i] = static_cast<float>((i + p) % 11) - 5.0f;
            for (int i = 0; i < kv_dim; i++) {
                k[i] = static_cast<float>((i + p * 3) % 7) - 3.0f;
                v[i] = static_cast<float>((i + p * 5) % 9) - 4.0f;
            }
            attention(out.data(), q.data(), k.data(), v.data(),
                      key_cache.data(), value_cache.data(), att.data(),
                      0, p, cfg);
        }

        bool no_nan = true;
        for (int i = 0; i < n_embd; i++) {
            if (std::isnan(out[i]) || std::isinf(out[i])) { no_nan = false; break; }
        }
        check("10 tokens: no NaN/Inf in output", no_nan);
    }

    // --- Test 4: GQA head mapping works correctly ---
    // Verify that query heads 0 and 8 use different KV heads
    // when KV heads have different values
    {
        std::fill(key_cache.begin(), key_cache.end(), 0.0f);
        std::fill(value_cache.begin(), value_cache.end(), 0.0f);

        // Set up k with same values across all KV heads (equal scores)
        for (int i = 0; i < kv_dim; i++) k[i] = 1.0f;
        for (int i = 0; i < n_embd; i++) q[i] = 1.0f;

        // But set v so KV head 0 = all 10.0, KV head 1 = all 20.0
        for (int i = 0; i < kv_dim; i++) v[i] = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            v[0 * head_dim + d] = 10.0f;   // KV head 0
            v[1 * head_dim + d] = 20.0f;   // KV head 1
        }

        attention(out.data(), q.data(), k.data(), v.data(),
                  key_cache.data(), value_cache.data(), att.data(),
                  0, 0, cfg);

        // Query head 0 uses KV head 0 → output should be 10.0
        // Query head 8 uses KV head 1 → output should be 20.0
        float head0_val = out[0];                    // first dim of query head 0
        float head8_val = out[8 * head_dim];         // first dim of query head 8
        check("GQA: head 0 uses KV head 0 (val=10)", approx_equal(head0_val, 10.0f, 0.01f));
        check("GQA: head 8 uses KV head 1 (val=20)", approx_equal(head8_val, 20.0f, 0.01f));
    }
}

void test_rope() {
    std::cout << "\n=== rope ===\n";

    // TinyLlama dimensions
    const int head_dim = 64;
    const int n_head = 32;
    const int n_head_kv = 4;
    const float freq_base = 10000.0f;

    // --- Test 1: Position 0 should not change anything ---
    // Rotating by angle 0 means cos(0)=1, sin(0)=0
    // So new_x = x*1 - y*0 = x, new_y = x*0 + y*1 = y (unchanged)
    {
        // Fill q with known values
        std::vector<float> q(n_head * head_dim, 1.0f);
        std::vector<float> k(n_head_kv * head_dim, 1.0f);
        std::vector<float> q_orig = q;
        std::vector<float> k_orig = k;

        rope(q.data(), k.data(), 0, head_dim, n_head, n_head_kv, freq_base);

        bool q_unchanged = true, k_unchanged = true;
        for (size_t i = 0; i < q.size(); i++) {
            if (!approx_equal(q[i], q_orig[i])) { q_unchanged = false; break; }
        }
        for (size_t i = 0; i < k.size(); i++) {
            if (!approx_equal(k[i], k_orig[i])) { k_unchanged = false; break; }
        }
        check("position 0: Q unchanged", q_unchanged);
        check("position 0: K unchanged", k_unchanged);
    }

    // --- Test 2: Rotation preserves vector length ---
    // A 2D rotation never stretches or shrinks a vector.
    // So the length (norm) of each head should be the same before and after.
    {
        std::vector<float> q(n_head * head_dim);
        std::vector<float> k(n_head_kv * head_dim);
        // Fill with varied values
        for (int i = 0; i < n_head * head_dim; i++) q[i] = static_cast<float>(i % 7) - 3.0f;
        for (int i = 0; i < n_head_kv * head_dim; i++) k[i] = static_cast<float>(i % 5) - 2.0f;

        // Compute norm of head 0 before
        float norm_before = 0.0f;
        for (int i = 0; i < head_dim; i++) norm_before += q[i] * q[i];
        norm_before = std::sqrt(norm_before);

        rope(q.data(), k.data(), 42, head_dim, n_head, n_head_kv, freq_base);

        // Compute norm of head 0 after
        float norm_after = 0.0f;
        for (int i = 0; i < head_dim; i++) norm_after += q[i] * q[i];
        norm_after = std::sqrt(norm_after);

        check("rotation preserves vector length", approx_equal(norm_before, norm_after, 1e-3f));
    }

    // --- Test 3: Different positions give different results ---
    {
        std::vector<float> q1(n_head * head_dim, 1.0f);
        std::vector<float> k1(n_head_kv * head_dim, 1.0f);
        std::vector<float> q2(n_head * head_dim, 1.0f);
        std::vector<float> k2(n_head_kv * head_dim, 1.0f);

        rope(q1.data(), k1.data(), 5, head_dim, n_head, n_head_kv, freq_base);
        rope(q2.data(), k2.data(), 100, head_dim, n_head, n_head_kv, freq_base);

        bool different = false;
        for (int i = 0; i < head_dim; i++) {
            if (!approx_equal(q1[i], q2[i])) { different = true; break; }
        }
        check("position 5 vs 100 gives different Q", different);
    }

    // --- Test 4: Same position gives same result (deterministic) ---
    {
        std::vector<float> q1(n_head * head_dim, 2.5f);
        std::vector<float> k1(n_head_kv * head_dim, 2.5f);
        std::vector<float> q2(n_head * head_dim, 2.5f);
        std::vector<float> k2(n_head_kv * head_dim, 2.5f);

        rope(q1.data(), k1.data(), 17, head_dim, n_head, n_head_kv, freq_base);
        rope(q2.data(), k2.data(), 17, head_dim, n_head, n_head_kv, freq_base);

        bool same = true;
        for (size_t i = 0; i < q1.size(); i++) {
            if (!approx_equal(q1[i], q2[i])) { same = false; break; }
        }
        check("same position gives identical result", same);
    }

    // --- Test 5: Manual check on a single pair ---
    // For head 0, pair 0, position 1:
    //   freq = 1.0 / (10000^(0/64)) = 1.0
    //   angle = 1 * 1.0 = 1.0 radian
    //   q[0] = 3.0 * cos(1) - 4.0 * sin(1)
    //   q[1] = 3.0 * sin(1) + 4.0 * cos(1)
    {
        std::vector<float> q(n_head * head_dim, 0.0f);
        std::vector<float> k(n_head_kv * head_dim, 0.0f);
        q[0] = 3.0f;
        q[1] = 4.0f;

        rope(q.data(), k.data(), 1, head_dim, n_head, n_head_kv, freq_base);

        float expected_0 = 3.0f * std::cos(1.0f) - 4.0f * std::sin(1.0f);
        float expected_1 = 3.0f * std::sin(1.0f) + 4.0f * std::cos(1.0f);

        std::cout << "    manual check: q[0]=" << q[0] << " expected=" << expected_0 << "\n";
        std::cout << "    manual check: q[1]=" << q[1] << " expected=" << expected_1 << "\n";
        check("manual rotation q[0]", approx_equal(q[0], expected_0));
        check("manual rotation q[1]", approx_equal(q[1], expected_1));
    }
}

void test_matmul_realistic_size() {
    std::cout << "\n=== matmul at realistic dimensions ===\n";

    // Test with dimensions closer to TinyLlama (smaller for speed)
    // 256 x 256 matrix times 256-vector (full TinyLlama is 2048 x 2048)
    const int N = 256;

    std::vector<float> W(N * N);
    std::vector<float> x(N);
    std::vector<float> out(N, 0.0f);

    // Fill with a pattern: W[i][j] = (i == j) ? 1.0 : 0.0 (identity)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            W[i * N + j] = (i == j) ? 1.0f : 0.0f;
        }
        x[i] = static_cast<float>(i);
    }

    matmul(out.data(), W.data(), x.data(), N, N, GGML_TYPE_F32);

    bool identity_ok = true;
    for (int i = 0; i < N; i++) {
        if (!approx_equal(out[i], x[i])) {
            identity_ok = false;
            break;
        }
    }
    check("256x256 identity matmul", identity_ok);

    std::cout << "  (Note: full TinyLlama uses 2048x2048 and 2048x5632 matrices.\n"
              << "   Naive matmul will be slow but correct. NEON optimization comes in Phase 8.)\n";
}


void test_sampling() {
    std::cout << "\n=== sampling ===\n";

    // --- Argmax ---
    {
        float logits[] = {1.0f, 5.0f, 3.0f, 2.0f};
        int result = sample_argmax(logits, 4);
        check("argmax picks index 1 (value 5.0)", result == 1);
    }

    {
        float logits[] = {-1.0f, -5.0f, -0.5f, -2.0f};
        int result = sample_argmax(logits, 4);
        check("argmax with all negatives picks -0.5 (index 2)", result == 2);
    }

    // --- Temperature sampling: temp=0 should behave like argmax ---
    // (We can't easily test random sampling deterministically,
    //  but we can verify the structure works without crashing)
    {
        // Run temperature sampling a few times - should not crash
        bool no_crash = true;
        for (int trial = 0; trial < 10; trial++) {
            float logits_copy[] = {1.0f, 10.0f, 2.0f, 0.5f, 0.1f};
            int result = sample_temperature(logits_copy, 5, 1.0f);
            if (result < 0 || result >= 5) { no_crash = false; break; }
        }
        check("temperature sampling: valid token IDs", no_crash);
    }

    // --- Top-P sampling ---
    {
        bool no_crash = true;
        for (int trial = 0; trial < 10; trial++) {
            float logits_copy[] = {1.0f, 10.0f, 2.0f, 0.5f, 0.1f};
            int result = sample_top_p(logits_copy, 5, 0.8f, 0.9f);
            if (result < 0 || result >= 5) { no_crash = false; break; }
        }
        check("top-p sampling: valid token IDs", no_crash);
    }

    // --- Top-P with very low p should mostly pick the top token ---
    {
        int top_count = 0;
        for (int trial = 0; trial < 20; trial++) {
            float logits_copy[] = {0.0f, 100.0f, 0.0f, 0.0f, 0.0f};
            int result = sample_top_p(logits_copy, 5, 0.5f, 0.1f);
            if (result == 1) top_count++;
        }
        // With logit 100 and low temp+p, should almost always pick token 1
        check("top-p with dominant logit picks it consistently", top_count >= 18);
    }
}


void test_matmul_q8_0() {
    std::cout << "\n=== matmul_q8_0 ===\n";

    // Test 1: Simple case — 1 row, 32 columns (1 block)
    // scale = 1.0 (F16 = 0x3C00), all qs = 2, x = all 1.0
    // Expected: 1.0 * (2*1 + 2*1 + ... 32 times) = 1.0 * 64 = 64.0
    {
        block_q8_0 block;
        block.d = 0x3C00;  // 1.0 in F16
        for (int i = 0; i < QK8_0; i++) block.qs[i] = 2;

        float x[32];
        for (int i = 0; i < 32; i++) x[i] = 1.0f;

        float out = 0.0f;
        matmul_q8_0(&out, &block, x, 1, 32);
        std::cout << "    Q8_0 simple: " << out << " (expected 64.0)\n";
        check("Q8_0 scale=1.0, qs=2, x=1 → 64.0", approx_equal(out, 64.0f, 0.1f));
    }

    // Test 2: scale = 0.5 (F16 = 0x3800), qs = 0,1,2,...,31, x = all 1.0
    // Expected: 0.5 * (0+1+2+...+31) = 0.5 * 496 = 248.0
    {
        block_q8_0 block;
        block.d = 0x3800;  // 0.5 in F16
        for (int i = 0; i < QK8_0; i++) block.qs[i] = static_cast<std::int8_t>(i);

        float x[32];
        for (int i = 0; i < 32; i++) x[i] = 1.0f;

        float out = 0.0f;
        matmul_q8_0(&out, &block, x, 1, 32);
        std::cout << "    Q8_0 sequential: " << out << " (expected 248.0)\n";
        check("Q8_0 scale=0.5, qs=0..31, x=1 → 248.0", approx_equal(out, 248.0f, 0.5f));
    }

    // Test 3: Two rows, 64 columns (2 blocks per row)
    // Row 0: both blocks scale=1.0, qs=1 → dot with x=1 → 32+32 = 64
    // Row 1: both blocks scale=2.0 (0x4000), qs=1 → dot with x=1 → 2*(32)+2*(32) = 128
    {
        block_q8_0 blocks[4]; // 2 rows × 2 blocks
        // Row 0
        blocks[0].d = 0x3C00; // 1.0
        blocks[1].d = 0x3C00;
        for (int i = 0; i < QK8_0; i++) { blocks[0].qs[i] = 1; blocks[1].qs[i] = 1; }
        // Row 1
        blocks[2].d = 0x4000; // 2.0
        blocks[3].d = 0x4000;
        for (int i = 0; i < QK8_0; i++) { blocks[2].qs[i] = 1; blocks[3].qs[i] = 1; }

        float x[64];
        for (int i = 0; i < 64; i++) x[i] = 1.0f;

        float out[2] = {0};
        matmul_q8_0(out, blocks, x, 2, 64);
        std::cout << "    Q8_0 2-row: [" << out[0] << ", " << out[1] << "] (expected [64, 128])\n";
        check("Q8_0 two rows: row 0 = 64", approx_equal(out[0], 64.0f, 0.5f));
        check("Q8_0 two rows: row 1 = 128", approx_equal(out[1], 128.0f, 0.5f));
    }

    // Test 4: Dispatch through matmul() with GGML_TYPE_Q8_0
    {
        block_q8_0 block;
        block.d = 0x3C00; // scale = 1.0
        for (int i = 0; i < QK8_0; i++) block.qs[i] = 3;

        float x[32];
        for (int i = 0; i < 32; i++) x[i] = 1.0f;

        float out = 0.0f;
        matmul(&out, &block, x, 1, 32, GGML_TYPE_Q8_0);
        check("Q8_0 dispatch through matmul()", approx_equal(out, 96.0f, 0.5f));
    }
}


void test_matmul_q4_0() {
    std::cout << "\n=== matmul_q4_0 ===\n";

    // Test 1: Simple case — 1 row, 32 columns (1 block)
    // scale = 1.0 (F16 = 0x3C00)
    // All nibbles = 9, so value = 9 - 8 = 1
    // x = all 1.0
    // Expected: 1.0 * (1*1 × 32 times) = 32.0
    {
        block_q4_0 block;
        block.d = 0x3C00; // 1.0 in F16
        // Each byte: low nibble = 9, high nibble = 9 → byte = 0x99
        for (int i = 0; i < QK4_0 / 2; i++) block.qs[i] = 0x99;

        float x[32];
        for (int i = 0; i < 32; i++) x[i] = 1.0f;

        float out = 0.0f;
        matmul_q4_0(&out, &block, x, 1, 32);
        std::cout << "    Q4_0 simple: " << out << " (expected 32.0)\n";
        check("Q4_0 all nibbles=9 (val=1), x=1 → 32.0", approx_equal(out, 32.0f, 0.5f));
    }

    // Test 2: Mixed values
    // scale = 2.0 (F16 = 0x4000)
    // All nibbles = 8 → value = 8 - 8 = 0
    // x = all 1.0 → expected 0.0
    {
        block_q4_0 block;
        block.d = 0x4000; // 2.0
        for (int i = 0; i < QK4_0 / 2; i++) block.qs[i] = 0x88;

        float x[32];
        for (int i = 0; i < 32; i++) x[i] = 1.0f;

        float out = 0.0f;
        matmul_q4_0(&out, &block, x, 1, 32);
        std::cout << "    Q4_0 zero values: " << out << " (expected 0.0)\n";
        check("Q4_0 all nibbles=8 (val=0) → 0.0", approx_equal(out, 0.0f, 0.5f));
    }

    // Test 3: Dispatch through matmul()
    {
        block_q4_0 block;
        block.d = 0x3C00; // scale = 1.0
        // Low nibble = 10 (val=2), high nibble = 10 (val=2) → byte = 0xAA
        for (int i = 0; i < QK4_0 / 2; i++) block.qs[i] = 0xAA;

        float x[32];
        for (int i = 0; i < 32; i++) x[i] = 1.0f;

        float out = 0.0f;
        matmul(&out, &block, x, 1, 32, GGML_TYPE_Q4_0);
        // 32 elements, each value = 2, x = 1 → 1.0 * (2*32) = 64.0
        std::cout << "    Q4_0 dispatch: " << out << " (expected 64.0)\n";
        check("Q4_0 dispatch through matmul()", approx_equal(out, 64.0f, 0.5f));
    }
}


void test_matmul_q6_k() {
    std::cout << "\n=== matmul_q6_k ===\n";

    // Test 1: All q values = -32 (ql=0, qh=0), scales=1, d=1.0, x=1.0
    // Each of the 256 elements dequantizes to: 1.0 * 1 * (-32) = -32
    // Dot product with x=1.0: 256 * (-32) = -8192.0
    {
        block_q6_K block;
        block.d = 0x3C00; // 1.0 in F16
        std::memset(block.ql, 0x00, sizeof(block.ql));
        std::memset(block.qh, 0x00, sizeof(block.qh));
        for (int i = 0; i < 16; i++) block.scales[i] = 1;

        float x[256];
        for (int i = 0; i < 256; i++) x[i] = 1.0f;

        float out = 0.0f;
        matmul_q6_k(&out, &block, x, 1, 256);
        std::cout << "    Q6_K all q=-32: " << out << " (expected -8192.0)\n";
        check("Q6_K all ql=0,qh=0 -> q=-32, sum=-8192", approx_equal(out, -8192.0f, 1.0f));
    }

    // Test 2: All q values = 0 (ql=0, qh=0xAA to set upper 2 bits to 2)
    // For each qh byte = 0xAA = 10101010:
    //   (qh >> 0) & 3 = 2,  (qh >> 2) & 3 = 2,  (qh >> 4) & 3 = 2,  (qh >> 6) & 3 = 2
    // q = (0 | (2 << 4)) - 32 = 32 - 32 = 0
    {
        block_q6_K block;
        block.d = 0x3C00; // 1.0
        std::memset(block.ql, 0x00, sizeof(block.ql));
        std::memset(block.qh, 0xAA, sizeof(block.qh));
        for (int i = 0; i < 16; i++) block.scales[i] = 1;

        float x[256];
        for (int i = 0; i < 256; i++) x[i] = 1.0f;

        float out = 99.0f;
        matmul_q6_k(&out, &block, x, 1, 256);
        std::cout << "    Q6_K all q=0: " << out << " (expected 0.0)\n";
        check("Q6_K all q=0 -> sum=0", approx_equal(out, 0.0f, 1.0f));
    }

    // Test 3: Dispatch through matmul()
    {
        block_q6_K block;
        block.d = 0x3C00; // 1.0
        std::memset(block.ql, 0x00, sizeof(block.ql));
        std::memset(block.qh, 0x00, sizeof(block.qh));
        for (int i = 0; i < 16; i++) block.scales[i] = 1;

        float x[256];
        for (int i = 0; i < 256; i++) x[i] = 1.0f;

        float out = 0.0f;
        matmul(&out, &block, x, 1, 256, GGML_TYPE_Q6_K);
        std::cout << "    Q6_K dispatch: " << out << " (expected -8192.0)\n";
        check("Q6_K dispatch through matmul()", approx_equal(out, -8192.0f, 1.0f));
    }

    // Test 4: Multi-row with scale = 2.0
    {
        block_q6_K blocks[2];
        for (int r = 0; r < 2; r++) {
            blocks[r].d = 0x4000; // 2.0 in F16
            std::memset(blocks[r].ql, 0x00, sizeof(blocks[r].ql));
            std::memset(blocks[r].qh, 0x00, sizeof(blocks[r].qh));
            for (int i = 0; i < 16; i++) blocks[r].scales[i] = 1;
        }

        float x[256];
        for (int i = 0; i < 256; i++) x[i] = 1.0f;

        float out[2] = {0.0f, 0.0f};
        matmul_q6_k(out, blocks, x, 2, 256);
        std::cout << "    Q6_K multi-row: [" << out[0] << ", " << out[1]
                  << "] (expected [-16384, -16384])\n";
        check("Q6_K 2-row d=2.0", approx_equal(out[0], -16384.0f, 1.0f)
                               && approx_equal(out[1], -16384.0f, 1.0f));
    }
}


void test_multithreaded_matmul() {
    std::cout << "\n=== multithreaded matmul ===\n";

    // Test: Run a larger Q8_0 matmul with 1 thread and 4 threads,
    // verify both produce the same results.
    // 8 rows × 64 columns = 2 blocks per row, 8 rows
    const int rows = 8;
    const int cols = 64;
    const int blocks_per_row = cols / QK8_0;

    // Create weight blocks: each int8 = 1, scale = 0.5
    std::vector<block_q8_0> blocks(rows * blocks_per_row);
    for (auto& b : blocks) {
        b.d = 0x3800; // 0.5 in F16
        for (int j = 0; j < QK8_0; j++) b.qs[j] = 1;
    }

    // Input: all 1.0
    std::vector<float> x(cols, 1.0f);

    // Expected per row: 0.5 * (1*1 × 32) + 0.5 * (1*1 × 32) = 16 + 16 = 32.0

    // Run single-threaded
    std::vector<float> out_1t(rows, 0.0f);
    set_num_threads(1);
    matmul_q8_0(out_1t.data(), blocks.data(), x.data(), rows, cols);

    // Run multi-threaded
    std::vector<float> out_4t(rows, 0.0f);
    set_num_threads(4);
    matmul_q8_0(out_4t.data(), blocks.data(), x.data(), rows, cols);

    // Compare
    bool match = true;
    for (int i = 0; i < rows; i++) {
        if (!approx_equal(out_1t[i], out_4t[i])) {
            match = false;
            std::cout << "    MISMATCH row " << i << ": 1t=" << out_1t[i]
                      << " 4t=" << out_4t[i] << "\n";
        }
    }
    check("Q8_0: 1-thread vs 4-thread results match", match);
    check("Q8_0: row 0 = 32.0", approx_equal(out_4t[0], 32.0f, 0.5f));
    check("Q8_0: row 7 = 32.0", approx_equal(out_4t[7], 32.0f, 0.5f));

    // Also test F32 with threads
    std::vector<float> W_f32(rows * cols, 2.0f); // all weights = 2.0
    std::vector<float> out_f32_1t(rows, 0.0f);
    std::vector<float> out_f32_4t(rows, 0.0f);

    set_num_threads(1);
    matmul_f32(out_f32_1t.data(), W_f32.data(), x.data(), rows, cols);

    set_num_threads(4);
    matmul_f32(out_f32_4t.data(), W_f32.data(), x.data(), rows, cols);

    bool f32_match = true;
    for (int i = 0; i < rows; i++) {
        if (!approx_equal(out_f32_1t[i], out_f32_4t[i])) f32_match = false;
    }
    check("F32: 1-thread vs 4-thread results match", f32_match);
    // Each row: 2.0 * 1.0 × 64 cols = 128.0
    check("F32: row 0 = 128.0", approx_equal(out_f32_4t[0], 128.0f, 0.5f));

    // Reset to single-threaded for remaining tests
    set_num_threads(1);
}


// -------------------- main --------------------

int main() {
    std::cout << "=== Phase 4: Tensor Operations Tests ===\n";
    std::cout << std::fixed << std::setprecision(6);

    test_matmul_f32();
    test_matmul_f16();
    test_matmul_q8_0();
    test_matmul_q4_0();
    test_matmul_q6_k();
    test_rmsnorm();
    test_softmax();
    test_silu();
    test_elementwise_mul();
    test_vec_add();
    test_ffn_swiglu();
    test_attention();
    test_rope();
    test_matmul_realistic_size();
    test_sampling();
    test_multithreaded_matmul();

    std::cout << "\n============================\n";
    std::cout << "Results: " << tests_passed << " passed, "
              << tests_failed << " failed\n";

    if (tests_failed > 0) {
        std::cout << "⚠ Some tests failed!\n";
        return 1;
    } else {
        std::cout << "✓ All tests passed!\n";
        return 0;
    }
}