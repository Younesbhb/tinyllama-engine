#!/bin/bash
# ============================================================================
#  run_benchmark.sh — Full Benchmark Suite for the LLM Inference Engine
# ============================================================================
#
#  Usage:
#    ./run_benchmark.sh <model_f16.gguf> [model_q8.gguf] [model_q4.gguf] ...
#
#  What it does:
#    1. Backend comparison   — naive vs NEON (single-threaded)
#    2. Thread scaling       — 1, 2, 4, 8 threads (NEON backend)
#    3. Quantization comparison — all model files you provide
#
#  Requirements:
#    - Compiled benchmark binary (make benchmark)
#    - At least one GGUF model file
#    - Runs on Apple Silicon (M1/M2/M3) for NEON benchmarks
#
#  Output:
#    - Human-readable results to stdout
#    - CSV results to benchmark_results.csv
#
# ============================================================================

set -e

BENCHMARK="./benchmark"
CSV_FILE="benchmark_results.csv"

# Benchmark parameters
WARMUP=2
TRIALS=5
DECODE_TOKENS=20

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ---- Argument parsing ----

if [ $# -lt 1 ]; then
    echo -e "${RED}Error: No model file specified.${NC}"
    echo ""
    echo "Usage: $0 <model1.gguf> [model2.gguf] [model3.gguf] ..."
    echo ""
    echo "Examples:"
    echo "  $0 tinyllama-1.1b-f16.gguf"
    echo "  $0 tinyllama-f16.gguf tinyllama-q8.gguf tinyllama-q4.gguf"
    echo ""
    echo "Options (set via environment variables):"
    echo "  WARMUP=N          Warmup iterations (default: 2)"
    echo "  TRIALS=N          Timed trials (default: 5)"
    echo "  DECODE_TOKENS=N   Tokens to decode (default: 20)"
    echo "  THREADS=\"1 2 4 8\" Thread counts to test (default: \"1 2 4 8\")"
    exit 1
fi

# Override defaults from environment
WARMUP=${WARMUP:-2}
TRIALS=${TRIALS:-5}
DECODE_TOKENS=${DECODE_TOKENS:-20}
THREAD_COUNTS="${THREADS:-1 2 4 8}"

# Collect model files
MODEL_FILES=("$@")
PRIMARY_MODEL="${MODEL_FILES[0]}"

# Verify benchmark binary exists
if [ ! -f "$BENCHMARK" ]; then
    echo -e "${YELLOW}Benchmark binary not found. Building...${NC}"
    make benchmark
    if [ ! -f "$BENCHMARK" ]; then
        echo -e "${RED}Build failed. Run 'make benchmark' manually.${NC}"
        exit 1
    fi
fi

# Verify model files exist
for model in "${MODEL_FILES[@]}"; do
    if [ ! -f "$model" ]; then
        echo -e "${RED}Error: Model file not found: $model${NC}"
        exit 1
    fi
done

COMMON_ARGS="--warmup $WARMUP --trials $TRIALS --decode-tokens $DECODE_TOKENS"

# ---- Print header ----

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║          LLM INFERENCE ENGINE — BENCHMARK SUITE            ║${NC}"
echo -e "${BOLD}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BOLD}║${NC}  Date:          $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "${BOLD}║${NC}  System:        $(uname -m) / $(uname -s)"
echo -e "${BOLD}║${NC}  CPU:           $(sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu 2>/dev/null | grep 'Model name' | sed 's/.*: *//' || echo 'unknown')"
echo -e "${BOLD}║${NC}  Models:        ${#MODEL_FILES[@]} file(s)"
echo -e "${BOLD}║${NC}  Warmup:        $WARMUP iterations"
echo -e "${BOLD}║${NC}  Trials:        $TRIALS per config"
echo -e "${BOLD}║${NC}  Decode tokens: $DECODE_TOKENS"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Initialize CSV
echo "label,backend,threads,quant,prompt_toks,decode_toks,prefill_median_ms,prefill_tok_s,decode_median_ms_tok,decode_tok_s" > "$CSV_FILE"


# ============================================================================
#  TEST 1: Backend Comparison (naive vs NEON, single-threaded)
# ============================================================================

echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}${BOLD}  TEST 1: Backend Comparison (naive vs NEON)${NC}"
echo -e "${BLUE}${BOLD}  Model: $(basename $PRIMARY_MODEL)   Threads: 1${NC}"
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

for backend in naive neon; do
    echo -e "\n${GREEN}▸ Running: backend=$backend threads=1${NC}"
    $BENCHMARK "$PRIMARY_MODEL" --backend $backend --threads 1 \
        $COMMON_ARGS

    # Also write CSV row
    $BENCHMARK "$PRIMARY_MODEL" --backend $backend --threads 1 \
        $COMMON_ARGS --csv --label "backend_cmp" >> "$CSV_FILE"
done

# Compute speedup from CSV
echo ""
echo -e "${YELLOW}${BOLD}  Backend Speedup Summary:${NC}"
# Extract decode tok/s for naive and neon from the last two CSV rows
NAIVE_TOKS=$(tail -2 "$CSV_FILE" | head -1 | cut -d',' -f10)
NEON_TOKS=$(tail -1 "$CSV_FILE" | cut -d',' -f10)
if [ -n "$NAIVE_TOKS" ] && [ -n "$NEON_TOKS" ]; then
    SPEEDUP=$(echo "scale=2; $NEON_TOKS / $NAIVE_TOKS" | bc 2>/dev/null || echo "N/A")
    echo -e "  Naive:  ${NAIVE_TOKS} tok/s"
    echo -e "  NEON:   ${NEON_TOKS} tok/s"
    echo -e "  ${BOLD}Speedup: ${SPEEDUP}x${NC}"
fi


# ============================================================================
#  TEST 2: Thread Scaling (NEON backend)
# ============================================================================

echo ""
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}${BOLD}  TEST 2: Thread Scaling (NEON backend)${NC}"
echo -e "${BLUE}${BOLD}  Model: $(basename $PRIMARY_MODEL)   Threads: $THREAD_COUNTS${NC}"
echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

THREAD_RESULTS=""
for threads in $THREAD_COUNTS; do
    echo -e "\n${GREEN}▸ Running: backend=neon threads=$threads${NC}"
    $BENCHMARK "$PRIMARY_MODEL" --backend neon --threads $threads \
        $COMMON_ARGS

    $BENCHMARK "$PRIMARY_MODEL" --backend neon --threads $threads \
        $COMMON_ARGS --csv --label "thread_scaling" >> "$CSV_FILE"

    TOK_S=$(tail -1 "$CSV_FILE" | cut -d',' -f10)
    THREAD_RESULTS="$THREAD_RESULTS  $threads threads: $TOK_S tok/s\n"
done

echo ""
echo -e "${YELLOW}${BOLD}  Thread Scaling Summary:${NC}"
echo -e "$THREAD_RESULTS"

# Compute scaling efficiency
SINGLE_THREAD_TOKS=$(grep "thread_scaling" "$CSV_FILE" | head -1 | cut -d',' -f10)
echo -e "${YELLOW}  Scaling efficiency (vs 1 thread):${NC}"
for line in $(grep "thread_scaling" "$CSV_FILE" | cut -d',' -f3,10); do
    T=$(echo $line | cut -d',' -f1)
    TOKS=$(echo $line | cut -d',' -f2)
    if [ -n "$SINGLE_THREAD_TOKS" ] && [ -n "$TOKS" ]; then
        SPEEDUP=$(echo "scale=2; $TOKS / $SINGLE_THREAD_TOKS" | bc 2>/dev/null || echo "N/A")
        EFFICIENCY=$(echo "scale=0; $SPEEDUP * 100 / $T" | bc 2>/dev/null || echo "N/A")
        echo -e "    $T threads: ${SPEEDUP}x speedup  (${EFFICIENCY}% efficient)"
    fi
done


# ============================================================================
#  TEST 3: Quantization Comparison (if multiple models provided)
# ============================================================================

if [ ${#MODEL_FILES[@]} -gt 1 ]; then
    # Pick the best thread count from test 2
    BEST_THREADS=$(grep "thread_scaling" "$CSV_FILE" | sort -t',' -k10 -rn | head -1 | cut -d',' -f3)
    if [ -z "$BEST_THREADS" ]; then
        BEST_THREADS=1
    fi

    echo ""
    echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}${BOLD}  TEST 3: Quantization Format Comparison${NC}"
    echo -e "${BLUE}${BOLD}  Backend: NEON   Threads: $BEST_THREADS (best from test 2)${NC}"
    echo -e "${BLUE}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    QUANT_RESULTS=""
    for model in "${MODEL_FILES[@]}"; do
        echo -e "\n${GREEN}▸ Running: $(basename $model)${NC}"
        $BENCHMARK "$model" --backend neon --threads $BEST_THREADS \
            $COMMON_ARGS

        $BENCHMARK "$model" --backend neon --threads $BEST_THREADS \
            $COMMON_ARGS --csv --label "quant_cmp" >> "$CSV_FILE"

        QUANT=$(tail -1 "$CSV_FILE" | cut -d',' -f4)
        TOK_S=$(tail -1 "$CSV_FILE" | cut -d',' -f10)
        MS_TOK=$(tail -1 "$CSV_FILE" | cut -d',' -f9)
        SIZE=$(ls -lh "$model" | awk '{print $5}')
        QUANT_RESULTS="$QUANT_RESULTS  $QUANT ($SIZE):  $TOK_S tok/s  ($MS_TOK ms/tok)\n"
    done

    echo ""
    echo -e "${YELLOW}${BOLD}  Quantization Comparison Summary:${NC}"
    echo -e "$QUANT_RESULTS"
else
    echo ""
    echo -e "${YELLOW}  Skipping quantization comparison (provide multiple model files to enable).${NC}"
    echo -e "${YELLOW}  Example: $0 model-f16.gguf model-q8.gguf model-q4.gguf${NC}"
fi


# ============================================================================
#  Final Summary
# ============================================================================

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║                    BENCHMARK COMPLETE                       ║${NC}"
echo -e "${BOLD}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BOLD}║${NC}  CSV results saved to: ${GREEN}$CSV_FILE${NC}"
echo -e "${BOLD}║${NC}"
echo -e "${BOLD}║${NC}  To view CSV:  ${BLUE}column -t -s',' $CSV_FILE${NC}"
echo -e "${BOLD}║${NC}  To plot:      Import $CSV_FILE into a spreadsheet"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""