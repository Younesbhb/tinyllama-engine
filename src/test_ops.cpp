#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <iostream>
#include <iomanip>

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


// -------------------- main --------------------

int main() {
    std::cout << "=== Phase 4: Tensor Operations Tests ===\n";
    std::cout << std::fixed << std::setprecision(6);

    test_matmul_f32();
    test_matmul_f16();
    test_rmsnorm();
    test_softmax();
    test_silu();
    test_elementwise_mul();
    test_vec_add();
    test_rope();
    test_matmul_realistic_size();

    std::cout << "\n============================\n";
    std::cout << "Results: " << tests_passed << " passed, "
              << tests_failed << " failed\n";

    if (tests_failed > 0) {
        std::cout << "⚠ Some tests failed! Fix before moving to Phase 5.\n";
        return 1;
    } else {
        std::cout << "✓ All tests passed! Ready for Phase 5 (Transformer components).\n";
        return 0;
    }
}