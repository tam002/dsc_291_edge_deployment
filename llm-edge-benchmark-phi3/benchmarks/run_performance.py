"""
benchmarks/run_performance.py
Measures prompt-processing throughput, token-generation throughput, and
peak RSS memory for Llama 3.2 3B Instruct Q4_K_M using llama-bench.

Usage:
    python benchmarks/run_performance.py
    python benchmarks/run_performance.py --threads 8
"""

import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

MODEL_NAME = "Phi-3 Mini 3.82B Q4_K_M"
MODEL_PATH = os.path.expanduser("~/models/phi3_q4/Phi-3-mini-4k-instruct-q4.gguf")
LLAMA_BENCH = "./llama-bench"
RESULTS_DIR = Path("results/raw")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BENCH_ARGS  = ["-p", "512", "-n", "128", "-r", "3"]


def parse_bench_table(stdout):
    results = {}
    for line in stdout.splitlines():
        if "pp512" in line:
            m = re.search(r"([\d.]+)\s*±\s*([\d.]+)", line)
            if m:
                results["pp512_mean"] = float(m.group(1))
                results["pp512_std"]  = float(m.group(2))
        if "tg128" in line:
            m = re.search(r"([\d.]+)\s*±\s*([\d.]+)", line)
            if m:
                results["tg128_mean"] = float(m.group(1))
                results["tg128_std"]  = float(m.group(2))
    return results


def parse_time_output(stderr):
    results = {}
    for line in stderr.splitlines():
        if "Maximum resident set size" in line:
            m = re.search(r"(\d+)", line)
            if m:
                results["peak_rss_kb"]  = int(m.group(1))
                results["peak_rss_mib"] = round(int(m.group(1)) / 1024, 1)
        if "Elapsed (wall clock)" in line:
            results["wall_clock"] = line.split(":")[-1].strip()
        if "Percent of CPU" in line:
            results["cpu_percent"] = line.split(":")[-1].strip()
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()

    if not Path(LLAMA_BENCH).exists():
        print(f"ERROR: {LLAMA_BENCH} not found. Run setup.sh first.", file=sys.stderr)
        sys.exit(1)
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found at {MODEL_PATH}\nRun: bash models/download_models.sh", file=sys.stderr)
        sys.exit(1)

    cmd = ["/usr/bin/time", "-v", LLAMA_BENCH, "-m", MODEL_PATH, *BENCH_ARGS, "-t", str(args.threads)]
    print(f"Running: {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    result = {
        "timestamp":  datetime.now().isoformat(),
        "model_name": MODEL_NAME,
        "model_path": MODEL_PATH,
        **parse_bench_table(proc.stdout),
        **parse_time_output(proc.stderr),
    }

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"performance_{ts}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    print(f"\n{'='*55}")
    print(f"  Model     : {MODEL_NAME}")
    print(f"  pp512     : {result.get('pp512_mean','N/A')} ± {result.get('pp512_std','')} t/s")
    print(f"  tg128     : {result.get('tg128_mean','N/A')} ± {result.get('tg128_std','')} t/s")
    print(f"  Peak RAM  : {result.get('peak_rss_mib','N/A')} MiB")
    print(f"{'='*55}")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
