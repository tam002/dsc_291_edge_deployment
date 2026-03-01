"""
benchmarks/run_sweep.py
Full quantisation sweep — runs PPL, throughput, RAM, and power across
Q2_K through Q8_0. Produces unified CSV for the analysis notebook.

Usage:
    sudo python benchmarks/run_sweep.py            # full sweep with power
    python benchmarks/run_sweep.py --skip-ppl      # throughput + power only
    python benchmarks/run_sweep.py --quants Q4_K_M Q8_0
    python benchmarks/run_sweep.py --threads 8
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

MODEL_NAME  = "Phi-3 Mini 3.82B Instruct"
MODEL_DIR   = os.path.expanduser("~/models/phi3")
PREFIX      = "Phi-3-mini-4k-instruct"
LLAMA_BENCH = "./llama-bench"
LLAMA_PPL   = "./llama-perplexity"
WIKI_RAW    = os.path.expanduser("~/wiki.test.raw")
WIKI_SMALL  = os.path.expanduser("~/wiki.test.small.raw")
RESULTS_DIR = Path("results/raw")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_QUANTS  = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
BENCH_ARGS  = ["-p", "512", "-n", "128", "-r", "3"]
CTX_SIZE    = 512
WIKI_LINES  = 200
BATTERY_V   = 11.4  # M1 MacBook nominal voltage


def model_path(quant):
    return os.path.join(MODEL_DIR, f"{PREFIX}-{quant}.gguf")


def prepare_corpus():
    if not Path(WIKI_RAW).exists():
        print(f"ERROR: {WIKI_RAW} not found. Run setup.sh.", file=sys.stderr)
        sys.exit(1)
    open(WIKI_SMALL, "w").writelines(open(WIKI_RAW).readlines()[:WIKI_LINES])
    return WIKI_SMALL


def get_battery_mah():
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPPowerDataType"], text=True, stderr=subprocess.DEVNULL)
        m = re.search(r"Current Capacity[^:]*:\s*([\d]+)", out)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def parse_bench(stdout):
    r = {}
    for line in stdout.splitlines():
        if "pp512" in line:
            m = re.search(r"([\d.]+)\s*[±]\s*([\d.]+)", line)
            if m:
                r["pp_tps"] = float(m.group(1))
                r["pp_std"] = float(m.group(2))
        if "tg128" in line:
            m = re.search(r"([\d.]+)\s*[±]\s*([\d.]+)", line)
            if m:
                r["tg_tps"] = float(m.group(1))
                r["tg_std"] = float(m.group(2))
    return r


def parse_time_v(stderr):
    r = {}
    for line in stderr.splitlines():
        if "Maximum resident set size" in line:
            m = re.search(r"(\d+)", line)
            if m:
                r["peak_ram_kb"]  = int(m.group(1))
                r["peak_ram_mib"] = round(int(m.group(1)) / 1024, 1)
        if "Elapsed (wall clock)" in line:
            r["wall_clock"] = line.split(":")[-1].strip()
    return r


def parse_ppl(output):
    r = {}
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", output)
    if m:
        r["ppl"]     = float(m.group(1))
        r["ppl_unc"] = float(m.group(2))
    chunks = re.findall(r"\[\d+\]([\d.]+)", output)
    r["chunks"] = len(chunks)
    return r


def run_throughput_and_ram(mpath, threads):
    cmd = ["/usr/bin/time", "-v", LLAMA_BENCH,
           "-m", mpath, *BENCH_ARGS, "-t", str(threads)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {**parse_bench(proc.stdout), **parse_time_v(proc.stderr)}


def run_power(mpath, threads):
    if os.geteuid() != 0:
        print("    [power] Skipping — needs sudo for powermetrics.")
        return {}

    power_log = RESULTS_DIR / f"_pm_{int(time.time())}.txt"
    pm = subprocess.Popen([
        "powermetrics", "--samplers", "cpu_power,gpu_power,ane_power",
        "-i", "500", "--format", "text", "-o", str(power_log)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

    batt_before = get_battery_mah()
    t0 = time.perf_counter()
    subprocess.run([LLAMA_BENCH, "-m", mpath, *BENCH_ARGS, "-t", str(threads)],
                   capture_output=True)
    elapsed = time.perf_counter() - t0
    batt_after = get_battery_mah()

    time.sleep(1)
    pm.terminate(); pm.wait()

    result = {"elapsed_s": round(elapsed, 1)}
    try:
        log_txt = power_log.read_text()
        pkg_vals = [float(x) for x in
                    re.findall(r"Package Power:\s*([\d.]+)\s*mW", log_txt)]
        if pkg_vals:
            mean_mw = sum(pkg_vals) / len(pkg_vals)
            result["pkg_power_mean_mw"] = round(mean_mw, 1)
            result["pkg_power_peak_mw"] = round(max(pkg_vals), 1)
            # tg128 × 3 runs = 384 tokens generated total
            energy_j = mean_mw / 1000 * elapsed
            result["tokens_per_joule"] = round(384 / energy_j, 2) if energy_j > 0 else None
        power_log.unlink(missing_ok=True)
    except Exception as e:
        print(f"    [power] Parse warning: {e}")

    if batt_before and batt_after:
        mah_delta = batt_before - batt_after
        result["batt_mah_delta"] = round(mah_delta, 1)
        result["energy_mwh"]     = round(mah_delta * BATTERY_V, 1)

    return result


def run_perplexity(mpath):
    corpus = prepare_corpus()
    cmd = [LLAMA_PPL, "-m", mpath, "-f", corpus, "--ctx-size", str(CTX_SIZE)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return parse_ppl(proc.stdout + proc.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quants",   nargs="+", default=ALL_QUANTS)
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--threads",  type=int, default=8)
    args = parser.parse_args()

    is_mac = sys.platform == "darwin"
    print(f"\n{'='*60}")
    print(f"  Quantisation Sweep — {MODEL_NAME}")
    print(f"  Levels : {', '.join(args.quants)}")
    print(f"  Threads: {args.threads}  |  PPL: {'off' if args.skip_ppl else 'on'}")
    print(f"  Power  : {'on (sudo detected)' if os.geteuid()==0 else 'off (run with sudo)'}")
    print(f"{'='*60}\n")

    all_results = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for quant in args.quants:
        mpath = model_path(quant)
        if not Path(mpath).exists():
            print(f"  [SKIP] {quant} — {mpath} not found")
            continue

        size_gib = round(Path(mpath).stat().st_size / (1024**3), 2)
        print(f"\n── {quant}  ({size_gib} GiB) ──────────────────────────")

        row = {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME, "quant": quant, "size_gib": size_gib,
        }

        print("  Throughput + RAM...", end=" ", flush=True)
        row.update(run_throughput_and_ram(mpath, args.threads))
        print("done")

        if is_mac:
            print("  Power...", end=" ", flush=True)
            row.update(run_power(mpath, args.threads))
            print("done")

        if not args.skip_ppl:
            print("  Perplexity...", end=" ", flush=True)
            row.update(run_perplexity(mpath))
            print("done")

        all_results.append(row)
        print(f"  pp={row.get('pp_tps','?')} tg={row.get('tg_tps','?')} "
              f"RAM={row.get('peak_ram_mib','?')}MiB "
              f"PPL={row.get('ppl','?')} "
              f"tok/J={row.get('tokens_per_joule','?')}")

    if not all_results:
        print("No results. Download models first: bash models/download_models.sh")
        sys.exit(1)

    out_path = RESULTS_DIR / f"sweep_{ts}.csv"
    all_keys = list(dict.fromkeys(k for r in all_results for k in r))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*60}")
    print(f"  Done. {len(all_results)} levels tested.")
    print(f"  Results: {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
