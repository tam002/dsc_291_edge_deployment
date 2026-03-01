#!/usr/bin/env bash
# benchmarks/run_power.sh
# Measures power consumption during llama-bench on macOS (Apple Silicon / Intel).
# Requires: sudo (for powermetrics), unplugged battery for accurate results.
#
# Usage:
#   bash benchmarks/run_power.sh --model phi3
#   bash benchmarks/run_power.sh --model llama32
#   bash benchmarks/run_power.sh --model phi3 --threads 8

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_KEY="phi3"
THREADS=12
SAMPLE_MS=500   # powermetrics sample interval in ms

# ── Model paths ───────────────────────────────────────────────────────────────
PHI3_PATH="$HOME/models/phi3_q4/Phi-3-mini-4k-instruct-q4.gguf"
LLAMA_PATH="$HOME/models/llama3.2_3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
LLAMA_BENCH="./llama-bench"
RESULTS_DIR="results/raw"
mkdir -p "$RESULTS_DIR"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)   MODEL_KEY="$2"; shift 2 ;;
        --threads) THREADS="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

case "$MODEL_KEY" in
    phi3)    MODEL_PATH="$PHI3_PATH";  MODEL_NAME="Phi-3 Mini 3.82B Q4_K_M" ;;
    llama32) MODEL_PATH="$LLAMA_PATH"; MODEL_NAME="Llama 3.2 3B Q4_K_M" ;;
    *) echo "Unknown model: $MODEL_KEY (choose: phi3, llama32)"; exit 1 ;;
esac

# ── Checks ────────────────────────────────────────────────────────────────────
if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: run_power.sh is macOS-only (uses powermetrics + system_profiler)."
    exit 1
fi
if [[ "$EUID" -ne 0 ]]; then
    echo "ERROR: powermetrics requires sudo. Run as: sudo bash benchmarks/run_power.sh --model $MODEL_KEY"
    exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Run: bash models/download_models.sh"
    exit 1
fi
if [[ ! -f "$LLAMA_BENCH" ]]; then
    echo "ERROR: $LLAMA_BENCH not found. Run setup.sh first."
    exit 1
fi

# Warn if plugged in
POWER_SOURCE=$(system_profiler SPPowerDataType 2>/dev/null | grep "Power Source" | head -1 | awk -F: '{print $2}' | xargs)
if [[ "$POWER_SOURCE" == *"AC Power"* ]]; then
    echo "WARNING: Machine appears to be plugged in. Unplug for accurate battery drain measurements."
    read -r -p "Continue anyway? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || exit 0
fi

# ── Timestamp + output files ──────────────────────────────────────────────────
TS=$(date +"%Y%m%d_%H%M%S")
POWER_LOG="$RESULTS_DIR/power_raw_${MODEL_KEY}_${TS}.txt"
SUMMARY_CSV="$RESULTS_DIR/power_summary_${MODEL_KEY}_${TS}.csv"
BENCH_LOG="$RESULTS_DIR/bench_${MODEL_KEY}_${TS}.txt"

# ── Battery state before ──────────────────────────────────────────────────────
get_battery_info() {
    system_profiler SPPowerDataType 2>/dev/null
}

battery_before=$(get_battery_info)
batt_pct_before=$(echo "$battery_before" | grep "State of Charge" | awk -F: '{print $2}' | xargs | tr -d '%')
batt_mah_before=$(echo "$battery_before" | grep "Current Capacity" | head -1 | awk -F: '{print $2}' | xargs | tr -d ' mAh')

echo ""
echo "════════════════════════════════════════════════════"
echo "  llm-edge-benchmark — Power Test"
echo "════════════════════════════════════════════════════"
echo "  Model       : $MODEL_NAME"
echo "  Threads     : $THREADS"
echo "  Sample rate : ${SAMPLE_MS}ms"
echo "  Power source: $POWER_SOURCE"
echo "  Battery before: ${batt_pct_before}% (${batt_mah_before} mAh)"
echo "════════════════════════════════════════════════════"
echo ""

# ── Start powermetrics in background ─────────────────────────────────────────
echo "==> Starting powermetrics (logging to $POWER_LOG)..."
powermetrics \
    --samplers cpu_power,gpu_power,ane_power \
    -i "$SAMPLE_MS" \
    --format text \
    -o "$POWER_LOG" &
POWERMETRICS_PID=$!

# Give powermetrics a moment to initialise
sleep 2

BENCH_START=$(date +%s)

# ── Run llama-bench ───────────────────────────────────────────────────────────
echo "==> Running llama-bench..."
"$LLAMA_BENCH" \
    -m "$MODEL_PATH" \
    -p 512 \
    -n 128 \
    -r 3 \
    -t "$THREADS" \
    2>&1 | tee "$BENCH_LOG"

BENCH_END=$(date +%s)
ELAPSED=$((BENCH_END - BENCH_START))

# ── Stop powermetrics ─────────────────────────────────────────────────────────
sleep 1  # capture final sample
kill "$POWERMETRICS_PID" 2>/dev/null || true
wait "$POWERMETRICS_PID" 2>/dev/null || true
echo ""
echo "==> powermetrics stopped."

# ── Battery state after ───────────────────────────────────────────────────────
battery_after=$(get_battery_info)
batt_pct_after=$(echo "$battery_after" | grep "State of Charge" | awk -F: '{print $2}' | xargs | tr -d '%')
batt_mah_after=$(echo "$battery_after" | grep "Current Capacity" | head -1 | awk -F: '{print $2}' | xargs | tr -d ' mAh')

batt_pct_delta=$(echo "$batt_pct_before $batt_pct_after" | awk '{printf "%.1f", $1 - $2}')
batt_mah_delta=$(echo "$batt_mah_before $batt_mah_after" | awk '{printf "%.0f", $1 - $2}')

# ── Parse powermetrics log ────────────────────────────────────────────────────
echo "==> Parsing power log..."

# Extract CPU, GPU, ANE power readings (in mW) and compute stats
python3 - "$POWER_LOG" << 'PYEOF'
import sys, re, statistics

log = open(sys.argv[1]).read()

def extract_series(pattern, text):
    return [float(x) for x in re.findall(pattern, text)]

# powermetrics text format lines like: "CPU Power: 1234 mW"
cpu_vals = extract_series(r"CPU Power:\s*([\d.]+)\s*mW", log)
gpu_vals = extract_series(r"GPU Power:\s*([\d.]+)\s*mW", log)
ane_vals = extract_series(r"ANE Power:\s*([\d.]+)\s*mW", log)
pkg_vals = extract_series(r"Package Power:\s*([\d.]+)\s*mW", log)

def stats(vals, label):
    if not vals:
        print(f"  {label}: no data")
        return None, None, None
    mean = statistics.mean(vals)
    peak = max(vals)
    low  = min(vals)
    print(f"  {label}: mean={mean:.0f} mW  peak={peak:.0f} mW  min={low:.0f} mW  (n={len(vals)})")
    return mean, peak, low

print("\n── Power readings (mW) ──────────────────────────")
cpu_mean, cpu_peak, _ = stats(cpu_vals, "CPU    ")
gpu_mean, gpu_peak, _ = stats(gpu_vals, "GPU    ")
ane_mean, ane_peak, _ = stats(ane_vals, "ANE    ")
pkg_mean, pkg_peak, _ = stats(pkg_vals, "Package")

PYEOF

# ── Throughput from bench log ─────────────────────────────────────────────────
pp_tps=$(grep "pp512" "$BENCH_LOG" | grep -oE "[0-9]+\.[0-9]+" | head -1)
tg_tps=$(grep "tg128" "$BENCH_LOG" | grep -oE "[0-9]+\.[0-9]+" | head -1)

# ── Compute energy (mWh) from mAh delta × nominal voltage ────────────────────
# Apple Silicon nominal battery voltage ~11.1V (varies by model)
VOLTAGE=11.1
energy_mwh=$(echo "$batt_mah_delta $VOLTAGE" | awk '{printf "%.1f", $1 * $2}')

# ── Write summary CSV ─────────────────────────────────────────────────────────
cat > "$SUMMARY_CSV" << CSVEOF
timestamp,model_name,threads,elapsed_s,batt_pct_before,batt_pct_after,batt_pct_delta,batt_mah_before,batt_mah_after,batt_mah_delta,energy_mwh,pp512_tps,tg128_tps,power_log,bench_log
$TS,"$MODEL_NAME",$THREADS,$ELAPSED,$batt_pct_before,$batt_pct_after,$batt_pct_delta,$batt_mah_before,$batt_mah_after,$batt_mah_delta,$energy_mwh,$pp_tps,$tg_tps,$POWER_LOG,$BENCH_LOG
CSVEOF

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  POWER SUMMARY"
echo "════════════════════════════════════════════════════"
echo "  Model          : $MODEL_NAME"
echo "  Elapsed        : ${ELAPSED}s"
echo "  Battery before : ${batt_pct_before}% (${batt_mah_before} mAh)"
echo "  Battery after  : ${batt_pct_after}% (${batt_mah_after} mAh)"
echo "  Battery drained: ${batt_pct_delta}% (${batt_mah_delta} mAh)"
echo "  Est. energy    : ${energy_mwh} mWh (@ ${VOLTAGE}V nominal)"
echo "  pp512 speed    : ${pp_tps} t/s"
echo "  tg128 speed    : ${tg_tps} t/s"
echo "════════════════════════════════════════════════════"
echo ""
echo "  Raw power log  : $POWER_LOG"
echo "  Summary CSV    : $SUMMARY_CSV"
echo "  Bench log      : $BENCH_LOG"
echo ""