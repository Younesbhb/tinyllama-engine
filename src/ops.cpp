#include "ops.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <random>
#include <numeric>
#include <thread>
#include <vector>

// Phase 8: ARM NEON SIMD optimization
// On ARM (your M2 Mac), this header gives us access to NEON intrinsics —
// special C functions that map directly to hardware instructions which
// process 4 floats at once instead of 1.
//
// The #ifdef guard means this code still compiles on non-ARM machines
// (like x86 Linux), it just falls back to the naive loops.
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif


// -------------------- Backend Selection --------------------
// Global flag that controls whether NEON or naive code runs.
// Default to NEON on ARM, naive everywhere else.
// Can be changed at runtime via set_backend().

static Backend g_backend =
#ifdef __ARM_NEON
    Backend::NEON;
#else
    Backend::NAIVE;
#endif

void set_backend(Backend b) { g_backend = b; }
Backend get_backend() { return g_backend; }


// -------------------- Thread Count --------------------
// Number of threads for parallel matmul. Default 1 (no threading).
// Changed via set_num_threads() or --threads CLI flag.

static int g_num_threads = 1;

void set_num_threads(int n) { g_num_threads = (n < 1) ? 1 : n; }
int get_num_threads() { return g_num_threads; }


// -------------------- Parallel For --------------------
//
// Splits a range [0, total) across g_num_threads threads.
// Each thread calls func(start, end) on its chunk.
//
// If g_num_threads == 1, no threads are created — just calls func directly.
// This avoids any overhead for the single-threaded case.
//
// Why not std::async or OpenMP?
//   - std::async has unpredictable scheduling overhead
//   - OpenMP requires compiler flags (-fopenmp) and isn't always available
//   - std::thread gives us full control with minimal complexity
//
// Thread creation overhead is ~10μs per thread on M2. With ~154 matmul
// calls per token and 7 threads each, that's ~10ms overhead per token.
// Since each token takes ~350ms (Q8_0 NEON), the overhead is ~3%.

template<typename Func>
static void parallel_for(int total, Func&& func) {
    int n_threads = g_num_threads;

    // Single-threaded: skip all thread machinery
    if (n_threads <= 1 || total <= 1) {
        func(0, total);
        return;
    }

    // Don't create more threads than there are rows
    if (n_threads > total) n_threads = total;

    // Divide rows into chunks. Last thread gets any remainder.
    // Example with 2048 rows, 8 threads:
    //   chunk = 2048 / 8 = 256
    //   Thread 0: rows 0-255
    //   Thread 1: rows 256-511
    //   ...
    //   Thread 7: rows 1792-2047
    int chunk = total / n_threads;

    std::vector<std::thread> threads;
    threads.reserve(n_threads - 1);

    // Launch n_threads - 1 worker threads
    for (int t = 0; t < n_threads - 1; t++) {
        int start = t * chunk;
        int end = start + chunk;
        // emplace_back(func, start, end) creates a new thread and immediately starts it running func(start, end)
        threads.emplace_back(func, start, end);
    }

    // Current thread handles the last chunk (includes any remainder)
    int last_start = (n_threads - 1) * chunk;
    func(last_start, total);

    // Wait for all worker threads to finish
    for (auto& th : threads) {
        // `join()` means "wait until this thread finishes." We loop through all 7 worker threads and wait for each one.
        // The current thread already finished its own chunk (the `func` call above completed), so now it just waits for any stragglers.
        th.join();
    }
}


// -------------------- FP16 → F32 conversion --------------------
// This manual conversion is still needed for non-matmul uses (e.g. dump_tensor)
// and for the naive matmul_f16 path.
// Inside the NEON matmul_f16, we use the hardware instruction vcvt_f32_f16
// instead, which converts 4 half-floats in a single cycle.

float fp16_to_f32(std::uint16_t h) {
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
//
// NEON optimization: We use 4 accumulator registers to process 16 floats
// per loop iteration. Why 4 accumulators instead of 1?
// The CPU can start the next FMA instruction while the previous one is
// still finishing (instruction pipelining). With 1 accumulator, each FMA
// depends on the previous one's result, so the CPU stalls. With 4
// independent accumulators, the CPU keeps all its execution units busy.

void matmul_f32(float* out, const float* W, const float* x,
                int rows, int cols) {
#ifdef __ARM_NEON
    if (g_backend == Backend::NEON) {
        // Lambdas can capture variables (the [&] part captures out, W, x, cols, blocks_per_row)
        parallel_for(rows, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            const float* row = W + i * cols;

            // 4 independent accumulator registers — each holds 4 partial sums.
            // By using 4 separate accumulators, the CPU can pipeline the FMA
            // instructions without waiting for the previous result.            
            // vdupq_n_f32(val): Create a register with all 4 slots set to val
            float32x4_t sum0 = vdupq_n_f32(0.0f);  // [0, 0, 0, 0]
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);

            int j = 0;

            // Main loop: process 16 floats per iteration (4 registers × 4 floats)
            // For cols=2048, this runs 128 times instead of 2048
            for (; j + 15 < cols; j += 16) {
                // Load 16 weights from the current row
                // vld1q_f32(ptr): Load 4 consecutive floats from memory into a register
                float32x4_t w0 = vld1q_f32(row + j);       // row[j..j+3]
                float32x4_t w1 = vld1q_f32(row + j + 4);   // row[j+4..j+7]
                float32x4_t w2 = vld1q_f32(row + j + 8);   // row[j+8..j+11]
                float32x4_t w3 = vld1q_f32(row + j + 12);  // row[j+12..j+15]

                // Load 16 input values
                float32x4_t x0 = vld1q_f32(x + j);
                float32x4_t x1 = vld1q_f32(x + j + 4);
                float32x4_t x2 = vld1q_f32(x + j + 8);
                float32x4_t x3 = vld1q_f32(x + j + 12);

                // Fused multiply-add: sum += w * x (4 floats at a time)
                // Each of these does 4 multiplications + 4 additions in one instruction
                // vfmaq_f32(acc, a, b): Fused multiply-add: acc + (a * b) element-wise
                sum0 = vfmaq_f32(sum0, w0, x0);
                sum1 = vfmaq_f32(sum1, w1, x1);
                sum2 = vfmaq_f32(sum2, w2, x2);
                sum3 = vfmaq_f32(sum3, w3, x3);
            }

            // Combine the 4 accumulators into one
            // sum0 + sum1 + sum2 + sum3 → one register with 4 partial sums
            // vaddq_f32(a, b): Add element-wise
            float32x4_t total = vaddq_f32(vaddq_f32(sum0, sum1),
                                          vaddq_f32(sum2, sum3));

            // Handle leftover elements (if cols isn't a multiple of 16)
            // Process 4 at a time
            for (; j + 3 < cols; j += 4) {
                total = vfmaq_f32(total, vld1q_f32(row + j), vld1q_f32(x + j));
            }

            // Horizontal sum: add the 4 floats inside the register into one number
            // [a, b, c, d] → a + b + c + d
            // vaddvq_f32(reg) : Horizontal sum, collapse 4 floats into 1
            float sum = vaddvq_f32(total);

            // Scalar tail: handle any remaining elements (0-3 floats)
            for (; j < cols; j++) {
                sum += row[j] * x[j];
            }

            out[i] = sum;
        }
        });
        return;
    }
#endif
    // Naive path
    parallel_for(rows, [&](int start, int end) {
    for (int i = start; i < end; i++) {
        float sum = 0.0f;
        // Pointer to the start of row i in the weight matrix
        const float* row = W + i * cols;
        for (int j = 0; j < cols; j++) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
    });
}


// -------------------- Matrix-Vector Multiply (F16 weights) --------------------
//
// Same as above, but weights are stored as 16-bit floats.
// We convert each weight to F32 on the fly before multiplying.
//
// This is the common case for TinyLlama F16 GGUF files.
// The conversion overhead is negligible because the CPU is already
// waiting for the next cache line of weights to arrive from RAM.
//
// NEON optimization: uses vcvt_f32_f16() — a single hardware instruction that
// converts 4 half-floats to 4 full floats in one cycle. This replaces
// 4 calls to the 30-line fp16_to_f32() function. Since TinyLlama's weights
// are all F16, this is where the biggest speedup comes from.

void matmul_f16(float* out, const std::uint16_t* W, const float* x,
                int rows, int cols) {
#ifdef __ARM_NEON
    if (g_backend == Backend::NEON) {
        parallel_for(rows, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            const std::uint16_t* row = W + i * cols;

            float32x4_t sum0 = vdupq_n_f32(0.0f);
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);

            int j = 0;

            // Main loop: process 16 half-floats per iteration
            for (; j + 15 < cols; j += 16) {
                // Load 4 uint16 values, reinterpret as half-float, convert to float32
                // This replaces 4 calls to fp16_to_f32() with ONE hardware instruction
                //
                // vld1_u16:             load 4 × uint16 from memory
                // vreinterpret_f16_u16: tell the CPU "these bits are half-floats"
                // vcvt_f32_f16:         convert 4 half-floats → 4 full floats (1 cycle!)
                float32x4_t w0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(row + j)));
                float32x4_t w1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(row + j + 4)));
                float32x4_t w2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(row + j + 8)));
                float32x4_t w3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(row + j + 12)));

                float32x4_t x0 = vld1q_f32(x + j);
                float32x4_t x1 = vld1q_f32(x + j + 4);
                float32x4_t x2 = vld1q_f32(x + j + 8);
                float32x4_t x3 = vld1q_f32(x + j + 12);

                sum0 = vfmaq_f32(sum0, w0, x0);
                sum1 = vfmaq_f32(sum1, w1, x1);
                sum2 = vfmaq_f32(sum2, w2, x2);
                sum3 = vfmaq_f32(sum3, w3, x3);
            }

            float32x4_t total = vaddq_f32(vaddq_f32(sum0, sum1),
                                          vaddq_f32(sum2, sum3));

            // Leftover: 4 at a time
            for (; j + 3 < cols; j += 4) {
                float32x4_t w = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(row + j)));
                float32x4_t v = vld1q_f32(x + j);
                total = vfmaq_f32(total, w, v);
            }

            float sum = vaddvq_f32(total);

            // Scalar tail
            for (; j < cols; j++) {
                sum += fp16_to_f32(row[j]) * x[j];
            }

            out[i] = sum;
        }
        });
        return;
    }
#endif
    // Naive path
    parallel_for(rows, [&](int start, int end) {
    for (int i = start; i < end; i++) {
        float sum = 0.0f;
        // Pointer to the start of row i (each element is 2 bytes)
        const std::uint16_t* row = W + i * cols;
        for (int j = 0; j < cols; j++) {
            float w = fp16_to_f32(row[j]);
            sum += w * x[j];
        }
        out[i] = sum;
    }
    });
}


// -------------------- Matrix-Vector Multiply (Q8_0 weights) --------------------
//
// Q8_0 quantization: weights are stored in blocks of 32.
// Each block has one F16 scale factor and 32 int8 values.
//
// To get the real float weight: float_value = int8_value * scale
//
// Memory layout of one row with 2048 columns:
//   [block 0 (34 bytes)][block 1 (34 bytes)]...[block 63 (34 bytes)]
//   2048 / 32 = 64 blocks per row, 64 × 34 = 2176 bytes per row
//   Compare to F16: 2048 × 2 = 4096 bytes per row (~1.88× smaller)
//
// The key insight for performance: we can compute the dot product of
// the 32 int8 values with the 32 input floats FIRST, then multiply
// by the scale once at the end. This means only 1 fp16_to_f32 call
// per block instead of 32.
//
// NEON optimization: load 8 int8s at a time, widen to float32, FMA.
// Per block of 32 elements: 4 iterations of 8 elements each.

void matmul_q8_0(float* out, const block_q8_0* W, const float* x,
                 int rows, int cols) {
    int blocks_per_row = cols / QK8_0;

#ifdef __ARM_NEON
    if (g_backend == Backend::NEON) {
        parallel_for(rows, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            const block_q8_0* row_blocks = W + i * blocks_per_row;
            float sum = 0.0f;

            for (int b = 0; b < blocks_per_row; b++) {
                // row_blocks[b] gets you the current block b at a certain row
                float scale = fp16_to_f32(row_blocks[b].d);
                // Pointer to the weights 
                const std::int8_t* qs = row_blocks[b].qs;
                const float* xb = x + b * QK8_0;

                // Two accumulators for the dot product within this block
                float32x4_t acc0 = vdupq_n_f32(0.0f);
                float32x4_t acc1 = vdupq_n_f32(0.0f);

                // Process 8 elements per iteration, 4 iterations for 32 elements
                for (int j = 0; j < QK8_0; j += 8) {
                    // Load 8 int8 values
                    int8x8_t raw = vld1_s8(qs + j);
                    /*
                    We have 8 numbers but they're tiny integers. We can't multiply them with floats yet. 
                    NEON has no instruction that multiplies int8 directly with float32. 
                    We need to gradually widen them until they're floats.
                    What we are doing is :
                    int8 (8-bit) → int16 (16-bit) → int32 (32-bit) → float32 (32-bit)
                    */

                    // Widen int8 → int16 (8 values)
                    int16x8_t wide = vmovl_s8(raw);

                    // Split into two groups of 4 and widen to int32 → float32
                    //   vget_low_s16:  first 4 int16 values
                    //   vmovl_s16:     int16 → int32 (4 values)
                    //   vcvtq_f32_s32: int32 → float32 (4 values)
                    float32x4_t flo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wide)));
                    float32x4_t fhi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wide)));

                    // Load 8 input values (already float32)
                    float32x4_t x0 = vld1q_f32(xb + j);
                    float32x4_t x1 = vld1q_f32(xb + j + 4);

                    // FMA: accumulate int8_as_float * input
                    acc0 = vfmaq_f32(acc0, flo, x0);
                    acc1 = vfmaq_f32(acc1, fhi, x1);
                }

                // Horizontal sum of the block's dot product, then multiply by scale
                float block_dot = vaddvq_f32(vaddq_f32(acc0, acc1));
                sum += scale * block_dot;
            }

            out[i] = sum;
        }
        });
        return;
    }
#endif
    // Naive path

    parallel_for(rows, [&](int start, int end) {
    // For each row loop
    for (int i = start; i < end; i++) {
        const block_q8_0* row_blocks = W + i * blocks_per_row;
        float sum = 0.0f;

        // For each block loop
        for (int b = 0; b < blocks_per_row; b++) {
            // Convert scale from F16 to F32 (once per block of 32 weights)
            // row_blocks[b] gets you the current block b at a certain row
            float scale = fp16_to_f32(row_blocks[b].d);
            const std::int8_t* qs = row_blocks[b].qs;
            const float* xb = x + b * QK8_0;

            // Dot product of 32 int8 values with 32 float inputs
            float block_sum = 0.0f;
            // For each weight loop
            for (int j = 0; j < QK8_0; j++) {
                block_sum += static_cast<float>(qs[j]) * xb[j];
            }

            // Multiply the dot product by the block's scale
            sum += scale * block_sum;
        }

        out[i] = sum;
    }
    });
}


// -------------------- Matrix-Vector Multiply (Q4_0 weights) --------------------
//
// Q4_0 quantization: weights are stored in blocks of 32.
// Each block has one F16 scale factor and 16 bytes of packed 4-bit values.
//
// Two values are packed into each byte:
//   byte[j] low nibble  (byte & 0x0F) → element j        (first 16 elements)
//   byte[j] high nibble (byte >> 4)    → element j + 16   (second 16 elements)
//
// The raw 4-bit value is unsigned (0-15). Subtract 8 to get signed range (-8 to +7).
// Final weight: float_value = (nibble - 8) * scale
//
// Memory: 18 bytes per block of 32 weights
//   2048 / 32 = 64 blocks per row, 64 × 18 = 1152 bytes per row
//   Compare to F16: 4096 bytes → ~3.56× smaller
//
// NEON optimization: load all 16 bytes, extract low/high nibbles with
// masking and shifting, convert to float, FMA with input values.

void matmul_q4_0(float* out, const block_q4_0* W, const float* x,
                 int rows, int cols) {
    int blocks_per_row = cols / QK4_0;

#ifdef __ARM_NEON
    if (g_backend == Backend::NEON) {
        parallel_for(rows, [&](int start, int end) {
        for (int i = start; i < end; i++) {
            const block_q4_0* row_blocks = W + i * blocks_per_row;
            float sum = 0.0f;

            for (int b = 0; b < blocks_per_row; b++) {
                float scale = fp16_to_f32(row_blocks[b].d);
                const std::uint8_t* qs = row_blocks[b].qs;
                const float* xb = x + b * QK4_0;

                float32x4_t acc0 = vdupq_n_f32(0.0f);
                float32x4_t acc1 = vdupq_n_f32(0.0f);

                // Load all 16 packed bytes at once
                // vld1q_u8(ptr): Load 16 uint8 values from memory into a 128-bit register
                uint8x16_t raw = vld1q_u8(qs);

                // There are 32 nibbles total. Each byte has 2 nibbles, and we have 16 bytes:
                // 16 bytes × 2 nibbles per byte = 32 nibbles = 32 weights.
                // 1 nibble - > 0000

                /*
                All 16 bytes laid out:
                byte[0]:  low nibble → weight 0,    high nibble → weight 16
                byte[1]:  low nibble → weight 1,    high nibble → weight 17
                byte[2]:  low nibble → weight 2,    high nibble → weight 18
                ...
                byte[15]: low nibble → weight 15,   high nibble → weight 31
                */

                // Extract low nibbles → elements 0-15
                /*
                vdupq_n_u8(val): Fill a 128-bit register with the same uint8 value in all 16 slots.
                vandq_u8(a, b): Bitwise AND between two registers, 16 bytes at once
                */
                uint8x16_t lo_nibbles = vandq_u8(raw, vdupq_n_u8(0x0F));

                // Extract high nibbles → elements 16-31
                // vshrq_n_u8(a, 4): Shift every byte right by 4 bits
                uint8x16_t hi_nibbles = vshrq_n_u8(raw, 4);

                // We can only widen 8 values at a time (64-bit → 128-bit), so we split the 16 low nibbles:
                // ---- Process first 16 elements (low nibbles) ----
                // Split into two halves of 8
                // vget_low_u8(a): Take the first 8 bytes from a 16-byte register. 128-bit → 64-bit (lower half)
                // vget_high_u8(a): Take the last 8 bytes from a 16-byte register. 128-bit → 64-bit (upper half)
                uint8x8_t lo_a = vget_low_u8(lo_nibbles);    // elements 0-7
                uint8x8_t lo_b = vget_high_u8(lo_nibbles);   // elements 8-15

                // Objective here: Widen uint8 → int16, subtract 8 to center (unsigned 0-15 → signed -8 to +7)
                // vmovl_u8(a): Widen 8 uint8 → 8 uint16 (same numbers, bigger containers) 64-bit → 128-bit
                // `vreinterpretq_s16_u16(...)` — tell the compiler "treat these as signed int16 now". Zero cost, just changes the type label. The bits don't change.
                // `vsubq_s16(..., 8)` — subtract 8 from each
                // vsubq_s16(a, b): Subtract two int16 registers element-wise
                int16x8_t lo16_a = vreinterpretq_s16_u16(vmovl_u8(lo_a));
                lo16_a = vsubq_s16(lo16_a, vdupq_n_s16(8));
                int16x8_t lo16_b = vreinterpretq_s16_u16(vmovl_u8(lo_b));
                lo16_b = vsubq_s16(lo16_b, vdupq_n_s16(8));

                // Elements 0-3: int16 → int32 → float32, FMA with x[0..3]
                // vget_low_s16(a): Take first 4 int16 values from an 8-value register. 128-bit → 64-bit (lower half)
                // vmovl_s16(a): Widen 4 int16 → 4 int32. 64-bit → 128-bit
                // vcvtq_f32_s32(a): Convert 4 int32 → 4 float32. Integer to floating point
                float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_a)));
                // Multiply with input floats and accumulate
                acc0 = vfmaq_f32(acc0, f0, vld1q_f32(xb));

                // Elements 4-7
                // vget_high_s16(a): Take last 4 int16 values from an 8-value register. 128-bit → 64-bit (upper half)
                float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_a)));
                acc1 = vfmaq_f32(acc1, f1, vld1q_f32(xb + 4));

                // Elements 8-11
                float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16_b)));
                acc0 = vfmaq_f32(acc0, f2, vld1q_f32(xb + 8));

                // Elements 12-15
                float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16_b)));
                acc1 = vfmaq_f32(acc1, f3, vld1q_f32(xb + 12));

                // ---- Process second 16 elements (high nibbles) ----
                uint8x8_t hi_a = vget_low_u8(hi_nibbles);    // elements 16-23
                uint8x8_t hi_b = vget_high_u8(hi_nibbles);   // elements 24-31

                int16x8_t hi16_a = vreinterpretq_s16_u16(vmovl_u8(hi_a));
                hi16_a = vsubq_s16(hi16_a, vdupq_n_s16(8));
                int16x8_t hi16_b = vreinterpretq_s16_u16(vmovl_u8(hi_b));
                hi16_b = vsubq_s16(hi16_b, vdupq_n_s16(8));

                // Elements 16-19
                float32x4_t f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_a)));
                acc0 = vfmaq_f32(acc0, f4, vld1q_f32(xb + 16));

                // Elements 20-23
                float32x4_t f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16_a)));
                acc1 = vfmaq_f32(acc1, f5, vld1q_f32(xb + 20));

                // Elements 24-27
                float32x4_t f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16_b)));
                acc0 = vfmaq_f32(acc0, f6, vld1q_f32(xb + 24));

                // Elements 28-31
                float32x4_t f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16_b)));
                acc1 = vfmaq_f32(acc1, f7, vld1q_f32(xb + 28));

                // Sum up and multiply by scale
                float block_dot = vaddvq_f32(vaddq_f32(acc0, acc1));
                sum += scale * block_dot;
            }

            out[i] = sum;
        }
        });
        return;
    }
#endif
    // Naive path
    parallel_for(rows, [&](int start, int end) {
    for (int i = start; i < end; i++) {
        const block_q4_0* row_blocks = W + i * blocks_per_row;
        float sum = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            float scale = fp16_to_f32(row_blocks[b].d);
            const std::uint8_t* qs = row_blocks[b].qs;
            const float* xb = x + b * QK4_0;

            float block_sum = 0.0f;
            for (int j = 0; j < QK4_0 / 2; j++) {
                // Low nibble → element j (first 16 elements)
                int v0 = static_cast<int>(qs[j] & 0x0F) - 8;
                // High nibble → element j + 16 (second 16 elements)
                int v1 = static_cast<int>(qs[j] >> 4) - 8;

                block_sum += static_cast<float>(v0) * xb[j];
                /*
                Why j and j + QK4_0 / 2?
                The 32 weights in a block are split into two halves. All low nibbles are the first 16 elements, all high nibbles are the second 16:
                qs[0]:  low nibble → element 0,    high nibble → element 16
                qs[1]:  low nibble → element 1,    high nibble → element 17
                qs[2]:  low nibble → element 2,    high nibble → element 18
                ...
                qs[15]: low nibble → element 15,   high nibble → element 31
                That's why the loop runs j = 0 to 15 (QK4_0 / 2 = 16 iterations), and the low nibble weight multiplies with xb[j] (elements 0–15) while the high nibble weight multiplies with xb[j + 16] (elements 16–31).
                */
                block_sum += static_cast<float>(v1) * xb[j + QK4_0 / 2];
            }

            sum += scale * block_sum;
        }

        out[i] = sum;
    }
    });
}


// -------------------- Matrix-Vector Multiply (type dispatch) --------------------
//
// This is the function called in the forward pass.
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
        case GGML_TYPE_Q8_0:
            matmul_q8_0(out, static_cast<const block_q8_0*>(W), x, rows, cols);
            break;
        case GGML_TYPE_Q4_0:
            matmul_q4_0(out, static_cast<const block_q4_0*>(W), x, rows, cols);
            break;
        default:
            throw std::runtime_error(
                "matmul: unsupported weight type.");
    }
}


// -------------------- RMSNorm --------------------
//
// RMSNorm is simpler than LayerNorm:
//   - LayerNorm: subtract mean, divide by stddev, scale + shift
//   - RMSNorm:   divide by RMS, scale only (no mean, no shift)
// 
// The formula: xb = (x / √(mean(x²) + ε)) × weight
//
// Steps:
//   1. Compute mean of squares: ms = (1/n) * sum(x_i^2)
//   2. Compute normalization factor: rsqrt = 1 / sqrt(ms + eps)
//   3. Scale: out_i = x_i * rsqrt * weight_i
//
// The epsilon (eps) prevents division by zero when the vector is all zeros.
// TinyLlama uses eps = 1e-5 (from config.rms_norm_eps).
//
// NEON: both the sum-of-squares loop and the normalize loop benefit
// from processing 4 floats at a time.

void rmsnorm(float* out, const float* x, const float* weight,
             int size, float eps) {
#ifdef __ARM_NEON
    if (g_backend == Backend::NEON) {
        // Step 1: sum of squares using NEON
        float32x4_t ss_vec = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 3 < size; i += 4) {
            float32x4_t v = vld1q_f32(x + i);
            ss_vec = vfmaq_f32(ss_vec, v, v);   // ss += x[i]^2 (4 at a time)
        }
        float ss = vaddvq_f32(ss_vec);
        // Scalar tail
        for (; i < size; i++) {
            ss += x[i] * x[i];
        }

        // Step 2: 1 / sqrt(mean_of_squares + eps)
        float rms = 1.0f / std::sqrt(ss / static_cast<float>(size) + eps);

        // Step 3: normalize and scale using NEON
        float32x4_t rms_vec = vdupq_n_f32(rms);
        i = 0;
        for (; i + 3 < size; i += 4) {
            float32x4_t v = vld1q_f32(x + i);
            float32x4_t w = vld1q_f32(weight + i);
            // out = x * rms * weight
            float32x4_t result = vmulq_f32(vmulq_f32(v, rms_vec), w);
            vst1q_f32(out + i, result);
        }
        // Scalar tail
        for (; i < size; i++) {
            out[i] = x[i] * rms * weight[i];
        }
        return;
    }
#endif
    // Naive path
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
//
// Not NEON-optimized: the exp() call dominates and has no NEON equivalent.

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
//
// Not NEON-optimized: exp() has no NEON equivalent.

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
#ifdef __ARM_NEON
    if (g_backend == Backend::NEON) {
        int i = 0;
        for (; i + 3 < size; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            vst1q_f32(a + i, vmulq_f32(va, vb));
        }
        for (; i < size; i++) {
            a[i] *= b[i];
        }
        return;
    }
#endif
    // Naive path
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
#ifdef __ARM_NEON
    if (g_backend == Backend::NEON) {
        int i = 0;
        for (; i + 3 < size; i += 4) {
            float32x4_t vo = vld1q_f32(out + i);
            float32x4_t vx = vld1q_f32(x + i);
            vst1q_f32(out + i, vaddq_f32(vo, vx));
        }
        for (; i < size; i++) {
            out[i] += x[i];
        }
        return;
    }
#endif
    // Naive path
    for (int i = 0; i < size; i++) {
        out[i] += x[i];
    }
}


// -------------------- RoPE (Rotary Positional Embeddings) --------------------
//
// Not NEON-optimized: uses sin()/cos() which have no NEON equivalent,
// and the loop is over pairs (32 iterations per head) — too small
// for NEON to make a meaningful difference.
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
//
// RoPE modifies the q and k vectors in place — the original values are overwritten with rotated values.
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
//
//
// attention is always from the perspective of the token being processed right now. When we're at position 3 processing "sat", the query belongs to "sat". We compare "sat"'s query against every cached key to find out which past tokens are relevant **to "sat"**.
//
// NEON optimization: The dot products in attention (q·k and att·v) use
// the same NEON FMA pattern as matmul. head_dim is 64, so we process
// 16 floats per iteration = 4 iterations per dot product.

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

    // Find where this layer's portion of the cache starts. 
    // The entire key cache is one giant flat array. Layer 0 gets the first chunk, layer 1 gets the next, etc. 
    // For layer 5: layer_offset = 5 × 2048 × 256 = 2,621,440. 
    // That's how many floats to skip to reach layer 5's section.
    int layer_offset = layer * n_ctx * kv_dim;

    float* k_cache_pos = key_cache + layer_offset + pos * kv_dim;
    float* v_cache_pos = value_cache + layer_offset + pos * kv_dim;


    // Copy the current token's 256-float key and 256-float value into the cache. 
    // After this, the cache now has keys and values for positions 0, 1, 2, AND 3. 
    // These will stay in the cache for all future tokens.
    // When position 50 is processed later, it can look back and see what we stored here.


    // memcpy is a library function that's been hand-optimized by the platform developers (Apple, in your case).
    // Under the hood it uses the widest possible memory operations available —
    // on your M2, it'll use 128-bit or even larger transfers to copy the whole block in far fewer operations. 
    // It also handles alignment and cache-line considerations that a naive loop doesn't.
    // For 256 floats (1KB), the difference is small in absolute terms. 
    // But this copy happens every token, every layer (22 times), twice (keys and values) — 
    // so 44 memcpy calls per token. It's a minor win but a free one.

    std::memcpy(k_cache_pos, k, static_cast<size_t>(kv_dim) * sizeof(float));
    std::memcpy(v_cache_pos, v, static_cast<size_t>(kv_dim) * sizeof(float));

    // ---- Step 2: For each query head, compute attention ----

    // Loop through all 32 query heads. Each head independently decides what to pay attention to. 
    // One head might focus on grammar, another on meaning, another on nearby words, etc.
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
        // This tells us "how relevant is position p to the current token?"

        for (int p = 0; p <= pos; p++) {
            // Pointer to the cached key at position p, KV head kvh
            const float* k_cached = key_cache + layer_offset + p * kv_dim + kvh * head_dim;

#ifdef __ARM_NEON
            if (g_backend == Backend::NEON) {
                // NEON dot product for attention score
                // head_dim = 64: processes 16 per iteration = 4 iterations total
                float32x4_t s0 = vdupq_n_f32(0.0f);
                float32x4_t s1 = vdupq_n_f32(0.0f);
                float32x4_t s2 = vdupq_n_f32(0.0f);
                float32x4_t s3 = vdupq_n_f32(0.0f);

                int d = 0;
                for (; d + 15 < head_dim; d += 16) {
                    s0 = vfmaq_f32(s0, vld1q_f32(q_head + d),      vld1q_f32(k_cached + d));
                    s1 = vfmaq_f32(s1, vld1q_f32(q_head + d + 4),  vld1q_f32(k_cached + d + 4));
                    s2 = vfmaq_f32(s2, vld1q_f32(q_head + d + 8),  vld1q_f32(k_cached + d + 8));
                    s3 = vfmaq_f32(s3, vld1q_f32(q_head + d + 12), vld1q_f32(k_cached + d + 12));
                }

                float score = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1),
                                                   vaddq_f32(s2, s3)));
                // Scalar tail
                for (; d < head_dim; d++) {
                    score += q_head[d] * k_cached[d];
                }

                // High score = "these two tokens are very relevant to each other." 
                // Low score = "not relevant."
                att[p] = score * scale;
            } else {
#endif
                // Dot product of q_head and k_cached (both are 64 floats)
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q_head[d] * k_cached[d];
                }

                // High score = "these two tokens are very relevant to each other." 
                // Low score = "not relevant."
                att[p] = score * scale;
#ifdef __ARM_NEON
            }
#endif
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

#ifdef __ARM_NEON
        if (g_backend == Backend::NEON) {
            // Zero the output for this head
            {
                int d = 0;
                float32x4_t zero = vdupq_n_f32(0.0f);
                for (; d + 3 < head_dim; d += 4) {
                    vst1q_f32(out_head + d, zero);
                }
                for (; d < head_dim; d++) {
                    out_head[d] = 0.0f;
                }
            }

            // Accumulate weighted values
            for (int t = 0; t <= pos; t++) {
                // Pointer to cached value at position t, KV head kvh
                const float* v_cached = value_cache + layer_offset + t * kv_dim + kvh * head_dim;
                // Broadcast the attention weight to all 4 NEON lanes
                float32x4_t w = vdupq_n_f32(att[t]);

                int d = 0;
                for (; d + 3 < head_dim; d += 4) {
                    float32x4_t o = vld1q_f32(out_head + d);
                    float32x4_t val = vld1q_f32(v_cached + d);
                    o = vfmaq_f32(o, w, val);    // out += weight * value
                    vst1q_f32(out_head + d, o);
                }
                for (; d < head_dim; d++) {
                    out_head[d] += att[t] * v_cached[d];
                }
            }
        } else {
#endif
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
#ifdef __ARM_NEON
        }
#endif
    }
    // At this point, out[0..2047] contains the concatenated output of all 32 heads.
    // Next step (outside this function): multiply by W_o to project back.
}


// -------------------- SwiGLU Feed-Forward Network --------------------
//
// This is the simpler half of each transformer block.
// It's just five steps using ops we already built.
//
// The expand-then-shrink pattern (2048 → 5632 → 2048) is common in
// neural networks. The bigger intermediate space gives the network
// more "room to think." It's like writing out your work on a big
// whiteboard and then condensing the answer onto a notecard.
//
// Why SwiGLU specifically?
//   - Old models used: out = W_down * relu(W_up * x)
//   - SwiGLU uses:     out = W_down * (silu(W_gate * x) ⊙ (W_up * x))
//   - The gating mechanism (⊙) lets the network learn which parts of
//     the expanded representation to keep and which to suppress.
//   - Empirically, models trained with SwiGLU perform better.

void ffn_swiglu(float* out, const float* input,
                float* hb, float* hb2,
                const void* W_gate, const void* W_up, const void* W_down,
                int n_ff, int n_embd, ggml_type type) {

    // Step 1: gate = W_gate * input
    // Expand from 2048 to 5632 through the gate projection
    matmul(hb, W_gate, input, n_ff, n_embd, type);

    // Step 2: up = W_up * input
    // Expand from 2048 to 5632 through the up projection (different weights)
    matmul(hb2, W_up, input, n_ff, n_embd, type);

    // Step 3: Apply SiLU activation to the gate
    // silu(x) = x * sigmoid(x)
    // This introduces non-linearity — without it, stacking layers
    // of matmuls would collapse into a single matmul (linear algebra).
    silu(hb, n_ff);

    // Step 4: Element-wise multiply gate and up
    // hb = silu(gate) ⊙ up
    // The gate controls how much of each dimension passes through.
    // If gate[i] is close to 0 after silu, that dimension gets suppressed
    // regardless of what up[i] contains.
    elementwise_mul(hb, hb2, n_ff);

    // Step 5: Shrink back from 5632 to 2048
    // out = W_down * hb
    matmul(out, W_down, hb, n_embd, n_ff, type);
}


// -------------------- Global random engine --------------------
// We use one shared random engine so results vary between calls.
// seeded with random_device for non-deterministic output.

static std::mt19937& get_rng() {
    static std::mt19937 rng(std::random_device{}());
    return rng;
}


// -------------------- Argmax (Greedy) Sampling --------------------
//
// Walk through all 32,000 logits, find the biggest one, return its index.
// That's it. No randomness, no creativity.
//
// Example:
//   logits = [..., -2.1, 8.7, 1.3, ...]
//                         ^-- index 3042 is highest
//   returns 3042

int sample_argmax(const float* logits, int n_vocab) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}


// -------------------- Temperature Sampling --------------------
//
// Temperature controls how "creative" the model is:
//
//   temp = 0.1: Almost greedy. Model is very confident, picks
//               the obvious next word almost every time.
//   temp = 1.0: Normal. The probabilities are as the model intended.
//   temp = 2.0: Very random. Even unlikely words get picked often.
//
// How it works:
//   1. Divide every logit by temperature
//      - Low temp → differences between logits get exaggerated
//        (big logits get even bigger relative to small ones)
//      - High temp → differences shrink (everything becomes similar)
//   2. Softmax → probabilities
//   3. Roll a random number and pick a token based on probabilities
//
// Example with logits [3.0, 1.0, 0.5]:
//   temp=0.5: logits become [6.0, 2.0, 1.0] → softmax ≈ [0.98, 0.02, 0.00]
//             (almost always picks the first token)
//   temp=1.0: logits stay   [3.0, 1.0, 0.5] → softmax ≈ [0.78, 0.11, 0.06]
//   temp=2.0: logits become [1.5, 0.5, 0.25]→ softmax ≈ [0.47, 0.17, 0.13]
//             (much more likely to pick less common tokens)

int sample_temperature(float* logits, int n_vocab, float temperature) {
    // Scale logits by temperature
    if (temperature != 1.0f && temperature > 0.0f) {
        float inv_temp = 1.0f / temperature;
        for (int i = 0; i < n_vocab; i++) {
            logits[i] *= inv_temp;
        }
    }

    // Convert to probabilities
    softmax(logits, n_vocab);

    // Sample: generate random number in [0, 1), walk through
    // probabilities until cumulative sum exceeds it
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(get_rng());

    float cumsum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        cumsum += logits[i];
        if (cumsum > r) {
            return i;
        }
    }

    // Fallback (shouldn't happen if softmax sums to 1.0)
    return n_vocab - 1;
}


// -------------------- Top-P (Nucleus) Sampling --------------------
//
// The problem with pure temperature sampling: even with reasonable
// temperature, the model might occasionally pick a wildly unlikely
// token (probability 0.001%) that produces garbage.
//
// Top-P fixes this by cutting off the tail. If top_p = 0.9:
//   1. Sort tokens by probability (highest first)
//   2. Walk down the sorted list, accumulating probability
//   3. Once you've accumulated 90%, stop — ignore everything below
//   4. Sample only from that top group
//
// Example: probabilities after softmax = [0.40, 0.30, 0.15, 0.10, 0.05]
//   with top_p = 0.9:
//     token 0: cumsum = 0.40 (keep)
//     token 1: cumsum = 0.70 (keep)
//     token 2: cumsum = 0.85 (keep)
//     token 3: cumsum = 0.95 > 0.9 (keep this one, then stop)
//     token 4: ignored
//   Sample from tokens 0-3 only, re-normalized.
//
// This is the standard sampling method for chat models.

int sample_top_p(float* logits, int n_vocab, float temperature, float top_p) {
    // Apply temperature
    if (temperature != 1.0f && temperature > 0.0f) {
        float inv_temp = 1.0f / temperature;
        for (int i = 0; i < n_vocab; i++) {
            logits[i] *= inv_temp;
        }
    }

    // Convert to probabilities
    softmax(logits, n_vocab);

    // Create index array sorted by probability (descending)
    // We need to track which token ID each probability belongs to
    std::vector<int> indices(n_vocab);
    std::iota(indices.begin(), indices.end(), 0);  // fill with 0, 1, 2, ...

    std::partial_sort(indices.begin(), indices.begin() + std::min(n_vocab, 100), indices.end(),
        [&logits](int a, int b) {
            return logits[a] > logits[b];
        });

    // Walk through sorted tokens, accumulate probability until we hit top_p
    float cumsum = 0.0f;
    int cutoff = 0;
    for (int i = 0; i < n_vocab; i++) {
        cumsum += logits[indices[i]];
        cutoff = i + 1;
        if (cumsum >= top_p) break;
    }

    // Sample from the top-p set
    // Re-normalize the kept probabilities
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(get_rng()) * cumsum;

    float running = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        running += logits[indices[i]];
        if (running > r) {
            return indices[i];
        }
    }

    // Fallback
    return indices[0];
}