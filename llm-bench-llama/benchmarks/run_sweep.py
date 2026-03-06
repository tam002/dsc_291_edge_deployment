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
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

MODEL_NAME  = "Llama 3.2 3B Instruct"
MODEL_DIR   = os.path.expanduser("~/models/llama3.2_3b")
PREFIX      = "Llama-3.2-3B-Instruct"
LLAMA_BENCH = "./llama-bench"
LLAMA_PPL   = "./llama-perplexity"
LLAMA_CLI   = "./llama-cli"
WIKI_RAW    = os.path.expanduser("~/wiki.test.raw")
WIKI_SMALL  = os.path.expanduser("~/wiki.test.small.raw")
RESULTS_DIR = Path("results/raw")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_QUANTS  = ["Q3_K_L", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
BENCH_ARGS  = ["-p", "512", "-n", "128", "-r", "3"]
CTX_SIZE    = 512
WIKI_LINES   = 200
BATTERY_V    = 11.4  # M1 MacBook nominal voltage
GSM8K_CACHE  = os.path.expanduser("~/gsm8k_test_100.json")
GSM8K_N      = 100


def model_path(quant):
    return os.path.join(MODEL_DIR, f"{PREFIX}-{quant}.gguf")


def prepare_corpus():
    if not Path(WIKI_RAW).exists():
        print(f"ERROR: {WIKI_RAW} not found. Run setup.sh.", file=sys.stderr)
        sys.exit(1)
    open(WIKI_SMALL, "w").writelines(open(WIKI_RAW).readlines()[:WIKI_LINES])
    return WIKI_SMALL



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


def parse_time_output(stderr):
    """
    Parse /usr/bin/time output for peak RAM.
    macOS (BSD time -l): 'maximum resident set size' in BYTES (lowercase label).
    Linux (GNU time -v): 'Maximum resident set size' in KB.
    Both are matched case-insensitively here.
    """
    r = {}
    for line in stderr.splitlines():
        if "maximum resident set size" in line.lower():
            m = re.search(r"(\d+)", line)
            if m:
                raw = int(m.group(1))
                # macOS BSD time reports bytes; Linux GNU time reports kilobytes
                r["peak_ram_mib"] = round(raw / (1024 * 1024), 1) if sys.platform == "darwin" \
                                    else round(raw / 1024, 1)
        if "elapsed" in line.lower() and "wall clock" in line.lower():
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


def run_throughput_ram_and_power(mpath, threads):
    """
    Single llama-bench pass that captures throughput, RAM, AND power simultaneously.
    Previously these were two separate bench runs (doubled wall-clock time per quant).
    powermetrics is started first, then the ONE bench run is measured for both
    throughput/RAM and power.
    """
    time_flag  = "-l" if sys.platform == "darwin" else "-v"
    result     = {}
    power_log  = RESULTS_DIR / f"_pm_{int(time.time())}.txt"
    has_power  = sys.platform == "darwin" and os.geteuid() == 0

    # Start powermetrics before the bench run (Mac + sudo only)
    pm = None
    print(f"    [power] has_power={has_power}, euid={os.geteuid()}, platform={sys.platform}")
    if has_power:
        pm = subprocess.Popen([
            "powermetrics", "--samplers", "cpu_power,gpu_power,ane_power",
            "-i", "500", "--format", "text", "-o", str(power_log)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        time.sleep(2)  # allow powermetrics to warm up
        print(f"    [power] powermetrics pid={pm.pid}, returncode={pm.returncode} (None=still running)")


    # Single bench run — timed for power, parsed for throughput + RAM
    t0   = time.perf_counter()
    cmd  = ["/usr/bin/time", time_flag, LLAMA_BENCH,
            "-m", mpath, *BENCH_ARGS, "-t", str(threads)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if pm:
        time.sleep(1)
        pm.terminate()
        _, pm_stderr = pm.communicate()
        print(f"    [power] powermetrics exit code={pm.returncode}")
        if pm_stderr:
            print(f"    [power] powermetrics stderr: {pm_stderr.decode(errors='replace')[:500]}")

    # Throughput + RAM from this single run
    result.update(parse_bench(proc.stdout))
    result.update(parse_time_output(proc.stderr))
    result["elapsed_s"] = round(elapsed, 1)

    # Power from powermetrics log
    if has_power and pm:
        try:
            log_txt  = power_log.read_text()
            print(f"    [power] log size={len(log_txt)} bytes")
            power_lines = [l for l in log_txt.splitlines()
                           if re.search(r"power|watt|mW", l, re.IGNORECASE)]
            print(f"    [power] power-related lines ({len(power_lines)} total), first 20:")
            for l in power_lines[:20]:
                print(f"      {l}")

            # Try "Package Power" (Intel), then "Combined Power" (Apple Silicon)
            pkg_vals = [float(x) for x in
                        re.findall(r"Package Power:\s*([\d.]+)\s*mW", log_txt)]
            if not pkg_vals:
                pkg_vals = [float(x) for x in
                            re.findall(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW", log_txt)]
            # Final fallback: sum CPU + GPU + ANE per sample
            if not pkg_vals:
                cpu_vals = [float(x) for x in re.findall(r"CPU Power:\s*([\d.]+)\s*mW", log_txt)]
                gpu_vals = [float(x) for x in re.findall(r"GPU Power:\s*([\d.]+)\s*mW", log_txt)]
                ane_vals = [float(x) for x in re.findall(r"ANE Power:\s*([\d.]+)\s*mW", log_txt)]
                if cpu_vals:
                    n = min(len(cpu_vals), len(gpu_vals) or len(cpu_vals), len(ane_vals) or len(cpu_vals))
                    pkg_vals = [
                        (cpu_vals[i] if i < len(cpu_vals) else 0) +
                        (gpu_vals[i] if i < len(gpu_vals) else 0) +
                        (ane_vals[i] if i < len(ane_vals) else 0)
                        for i in range(n)
                    ]
            print(f"    [power] pkg_vals found: {pkg_vals[:5]}")
            if pkg_vals:
                mean_mw = sum(pkg_vals) / len(pkg_vals)
                result["pkg_power_mean_mw"] = round(mean_mw, 1)
                result["pkg_power_peak_mw"] = round(max(pkg_vals), 1)
                avg_watts = mean_mw / 1000
                result["avg_watts"] = round(avg_watts, 3)
                tg_tps = result.get("tg_tps")
                result["tokens_per_joule"] = round(tg_tps / avg_watts, 2) if (tg_tps and avg_watts > 0) else None
            power_log.unlink(missing_ok=True)
        except Exception as e:
            print(f"    [power] Parse warning: {e}")

    elif sys.platform == "darwin" and os.geteuid() != 0:
        print("    [power] Skipping — run with sudo to enable powermetrics.")

    return result


def run_perplexity(mpath):
    corpus = prepare_corpus()
    cmd = [LLAMA_PPL, "-m", mpath, "-f", corpus, "--ctx-size", str(CTX_SIZE)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return parse_ppl(proc.stdout + proc.stderr)


def load_gsm8k():
    if Path(GSM8K_CACHE).exists():
        with open(GSM8K_CACHE) as f:
            return json.load(f)
    print(f"  [gsm8k] Downloading {GSM8K_N} questions from HuggingFace...", end=" ", flush=True)
    url = (
        f"https://datasets-server.huggingface.co/rows"
        f"?dataset=openai%2Fgsm8k&config=main&split=test&offset=0&limit={GSM8K_N}"
    )
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    rows = [{"question": r["row"]["question"], "answer": r["row"]["answer"]}
            for r in data["rows"]]
    with open(GSM8K_CACHE, "w") as f:
        json.dump(rows, f)
    print("done")
    return rows


def extract_gsm8k_answer(text):
    """Return the number after the last '####' in text, or None."""
    matches = re.findall(r"####\s*([\d,.\-]+)", text)
    if matches:
        return matches[-1].replace(",", "").strip()
    return None


def run_gsm8k(mpath, threads):
    questions = load_gsm8k()
    correct = 0
    for q in questions:
        prompt = (
            "Solve this math problem step by step. "
            "Put your final answer as a number after ####.\n\n"
            f"Question: {q['question']}\n\nAnswer:"
        )
        cmd = [
            LLAMA_CLI, "-m", mpath, "-t", str(threads),
            "-n", "200", "--temp", "0", "--log-disable", "-p", prompt,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            pred = extract_gsm8k_answer(proc.stdout)
            gt   = extract_gsm8k_answer(q["answer"])
            if pred is not None and gt is not None and pred == gt:
                correct += 1
        except subprocess.TimeoutExpired:
            pass
    return {"gsm8k_acc": round(100 * correct / len(questions), 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quants",    nargs="+", default=ALL_QUANTS)
    parser.add_argument("--skip-ppl",  action="store_true")
    parser.add_argument("--skip-gsm8k", action="store_true")
    parser.add_argument("--threads",   type=int, default=8)
    args = parser.parse_args()

    is_mac      = sys.platform == "darwin"
    has_power   = is_mac and os.geteuid() == 0
    print(f"\n{'='*60}")
    print(f"  Quantisation Sweep — {MODEL_NAME}")
    print(f"  Levels : {', '.join(args.quants)}")
    print(f"  Threads: {args.threads}  |  PPL: {'off' if args.skip_ppl else 'on'}  |  GSM8K: {'off' if args.skip_gsm8k else 'on'}")
    print(f"  Power  : {'on (sudo + Mac)' if has_power else 'off (needs sudo on Mac)'}")
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
            "avg_watts": None, "tokens_per_joule": None, "gsm8k_acc": None,
        }

        # Single bench pass — captures throughput, RAM, and power together
        print("  Throughput + RAM + Power...", end=" ", flush=True)
        row.update(run_throughput_ram_and_power(mpath, args.threads))
        print("done")

        if not args.skip_ppl:
            print("  Perplexity...", end=" ", flush=True)
            row.update(run_perplexity(mpath))
            print("done")

        if not args.skip_gsm8k:
            print(f"  GSM8K ({GSM8K_N} questions)...", end=" ", flush=True)
            row.update(run_gsm8k(mpath, args.threads))
            print("done")

        all_results.append(row)

        # Clear per-quant summary with pp and tg on separate lines
        print(f"\n  ┌─ {quant} results ───────────────────────────")
        print(f"  │  Prefill  (pp): {str(row.get('pp_tps', '—')):>8} ± {str(row.get('pp_std', '—'))} t/s")
        print(f"  │  Generate (tg): {str(row.get('tg_tps', '—')):>8} ± {str(row.get('tg_std', '—'))} t/s")
        print(f"  │  Peak RAM     : {str(row.get('peak_ram_mib', '—')):>8} MiB")
        if row.get("pkg_power_mean_mw"):
            print(f"  │  Pkg power    : {row['pkg_power_mean_mw']:>8.1f} mW (mean)")
        if row.get("tokens_per_joule"):
            print(f"  │  tok / joule  : {row['tokens_per_joule']:>8.2f}")
        if row.get("ppl"):
            print(f"  │  Perplexity   : {row['ppl']:>8.4f} ± {row['ppl_unc']:.5f}")
        if row.get("gsm8k_acc") is not None:
            print(f"  │  GSM8K acc    : {row['gsm8k_acc']:>7.1f}%")
        print(f"  └────────────────────────────────────────────")

    if not all_results:
        print("No results. Download models first: bash models/download_models.sh")
        sys.exit(1)

    out_path = RESULTS_DIR / f"sweep_{ts}.csv"
    fieldnames = [
        "timestamp", "model", "quant", "size_gib",
        "pp_tps", "pp_std", "tg_tps", "tg_std",
        "peak_ram_mib", "elapsed_s",
        "avg_watts", "tokens_per_joule",
        "pkg_power_mean_mw", "pkg_power_peak_mw",
        "ppl", "ppl_unc", "chunks",
        "gsm8k_acc",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*60}")
    print(f"  Done. {len(all_results)} levels tested.")
    print(f"  Results: {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
