"""
mlx/run_mlx_benchmark.py
Benchmarks Llama 3.2 3B Instruct using Apple MLX framework.
Compares against llama.cpp results on same metrics:
  pp_tps | tg_tps | peak_ram_mib | tokens_per_joule

MLX uses Apple Silicon GPU + ANE intelligently, giving better
power efficiency than llama.cpp CPU backend.

Usage:
    python mlx/run_mlx_benchmark.py
    sudo python mlx/run_mlx_benchmark.py   # includes power measurements
"""

import csv
import os
import re
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import mlx.core as mx
    from mlx_lm import load, generate
except ImportError:
    print("ERROR: MLX not installed. Run: pip install mlx mlx-lm")
    sys.exit(1)

MODEL_NAME  = "Llama 3.2 3B Instruct"
MLX_REPO    = "mlx-community/Llama-3.2-3B-Instruct-4bit"
MLX_DIR     = os.path.expanduser("~/models/llama3.2_3b_mlx")
RESULTS_DIR = Path("results/raw")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BATTERY_V   = 11.4

WIKI_RAW    = os.path.expanduser("~/wiki.test.raw")
N_PREFILL   = 512   # tokens to use as prefill prompt
N_GENERATE  = 128
N_RUNS      = 3


def load_wiki_prompt(tokenizer) -> str:
    """
    Build a ~512-token prefill prompt from WikiText-2 — the same corpus used
    by run_sweep.py for perplexity scoring. Using a shared real-text corpus
    makes prefill results directly comparable across llama.cpp and MLX, and
    avoids the artificial compressibility of a repeated synthetic sentence.
    """
    if not Path(WIKI_RAW).exists():
        print(f"  [warn] WikiText-2 not found at {WIKI_RAW}.")
        print(f"         Download it per the README, or prefill will use a fallback prompt.")
        return "The history of artificial intelligence spans many decades. " * 30

    # Read lines until we have at least N_PREFILL tokens
    text   = ""
    tokens = []
    with open(WIKI_RAW) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("="):   # skip headers
                continue
            text   += " " + line
            tokens  = tokenizer.encode(text)
            if len(tokens) >= N_PREFILL:
                break

    # Decode exactly N_PREFILL tokens so the count is deterministic
    prompt = tokenizer.decode(tokens[:N_PREFILL])
    return prompt


def get_peak_ram_mib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 * 1024), 1)  # macOS returns bytes


def get_battery_mah():
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPPowerDataType"], text=True, stderr=subprocess.DEVNULL)
        m = re.search(r"Current Capacity[^:]*:\s*([\d]+)", out)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def start_powermetrics(log_path):
    if os.geteuid() != 0:
        return None
    return subprocess.Popen([
        "powermetrics", "--samplers", "cpu_power,gpu_power,ane_power",
        "-i", "500", "--format", "text", "-o", str(log_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_power_log(log_path):
    try:
        txt = Path(log_path).read_text()
        vals = [float(x) for x in re.findall(r"Package Power:\s*([\d.]+)\s*mW", txt)]
        if vals:
            return {"pkg_mean_mw": round(sum(vals)/len(vals), 1),
                    "pkg_peak_mw": round(max(vals), 1)}
    except Exception:
        pass
    return {}


def benchmark_prefill(model, tokenizer):
    """
    Measure prompt-processing speed (tokens/sec) — prefill phase only.
    Prompt is sourced from WikiText-2 (same corpus as PPL in run_sweep.py)
    for a fair, reproducible comparison with llama.cpp prefill numbers.
    Uses generate(..., max_tokens=1) so nearly all elapsed time is prefill.
    """
    prompt   = load_wiki_prompt(tokenizer)
    n_tokens = len(tokenizer.encode(prompt))
    times    = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        generate(model, tokenizer, prompt=prompt, max_tokens=1, verbose=False)
        times.append(time.perf_counter() - t0)
    mean_t = sum(times) / len(times)
    print(f"\n    Prefill prompt: {n_tokens} tokens from WikiText-2")
    return round(n_tokens / mean_t, 2), n_tokens


def benchmark_generation(model, tokenizer):
    """Measure token generation speed (tokens/sec)."""
    prompt = "Explain quantum computing in simple terms."
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        _ = generate(model, tokenizer, prompt=prompt,
                     max_tokens=N_GENERATE, verbose=False)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    mean_t = sum(times) / len(times)
    return round(N_GENERATE / mean_t, 2)


def main():
    print(f"\n{'='*60}")
    print(f"  MLX Benchmark — {MODEL_NAME}")
    print(f"  Repo: {MLX_REPO}")
    print(f"{'='*60}\n")

    # Download MLX model if needed
    if not Path(MLX_DIR).exists():
        print(f"==> Downloading MLX model to {MLX_DIR}...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MLX_REPO, local_dir=MLX_DIR)
    else:
        print(f"==> Using cached MLX model: {MLX_DIR}")

    print("==> Loading model into MLX...")
    t_load = time.perf_counter()
    model, tokenizer = load(MLX_DIR)
    load_time_ms = round((time.perf_counter() - t_load) * 1000, 1)
    print(f"    Load time: {load_time_ms} ms")

    ram_after_load = get_peak_ram_mib()

    # Power monitoring
    pm_log = RESULTS_DIR / f"_mlx_pm_{int(time.time())}.txt"
    pm = start_powermetrics(pm_log)
    if pm:
        print("==> powermetrics started")
        time.sleep(2)
    else:
        print("==> Power monitoring skipped (run with sudo)")

    batt_before = get_battery_mah()
    t_bench_start = time.perf_counter()

    print("==> Prefill benchmark (pp512)...", end=" ", flush=True)
    pp_tps, n_prompt_tokens = benchmark_prefill(model, tokenizer)
    print(f"{pp_tps} t/s")

    print("==> Generation benchmark (tg128)...", end=" ", flush=True)
    tg_tps = benchmark_generation(model, tokenizer)
    print(f"{tg_tps} t/s")

    bench_elapsed = time.perf_counter() - t_bench_start
    batt_after = get_battery_mah()
    peak_ram = get_peak_ram_mib()

    power_data = {}
    if pm:
        time.sleep(1)
        pm.terminate(); pm.wait()
        power_data = parse_power_log(pm_log)
        Path(pm_log).unlink(missing_ok=True)

        if "pkg_mean_mw" in power_data:
            total_tg_tokens = N_GENERATE * N_RUNS
            energy_j = power_data["pkg_mean_mw"] / 1000 * bench_elapsed
            power_data["tokens_per_joule"] = round(total_tg_tokens / energy_j, 2)

    batt_data = {}
    if batt_before and batt_after:
        mah_delta = batt_before - batt_after
        batt_data = {
            "batt_mah_delta": round(mah_delta, 1),
            "energy_mwh":     round(mah_delta * BATTERY_V, 1),
        }

    result = {
        "timestamp":          datetime.now().isoformat(),
        "framework":          "MLX",
        "model":              MODEL_NAME,
        "quant":              "4bit",        # MLX model is 4-bit quantised
        "mlx_repo":           MLX_REPO,
        "load_time_ms":       load_time_ms,
        "peak_ram_mib":       peak_ram,
        "pp_tps":             pp_tps,
        "tg_tps":             tg_tps,
        "elapsed_s":          round(bench_elapsed, 1),
        # Power fields use same names as run_sweep.py for notebook compatibility
        "pkg_power_mean_mw":  power_data.get("pkg_mean_mw"),
        "pkg_power_peak_mw":  power_data.get("pkg_peak_mw"),
        "tokens_per_joule":   power_data.get("tokens_per_joule"),
        "batt_mah_delta":     batt_data.get("batt_mah_delta"),
        "energy_mwh":         batt_data.get("energy_mwh"),
    }

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"mlx_benchmark_{ts}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader(); writer.writerow(result)

    print(f"\n{'='*60}")
    print(f"  MLX RESULTS — {MODEL_NAME}")
    print(f"{'='*60}")
    print(f"  pp_tps         : {pp_tps} t/s")
    print(f"  tg_tps         : {tg_tps} t/s")
    print(f"  Peak RAM       : {peak_ram} MiB")
    if "pkg_mean_mw" in power_data:
        print(f"  Pkg power (mean): {power_data['pkg_mean_mw']} mW")
        print(f"  tokens/joule    : {power_data.get('tokens_per_joule','N/A')}")
    print(f"  Results: {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
