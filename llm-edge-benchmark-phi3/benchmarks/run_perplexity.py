"""
benchmarks/run_perplexity.py
Measures perplexity for Llama 3.2 3B Instruct Q4_K_M on Wikitext-2.

Usage:
    python benchmarks/run_perplexity.py
    python benchmarks/run_perplexity.py --lines 200 --ctx-size 512
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

MODEL_NAME        = "Phi-3 Mini 3.82B Q4_K_M"
MODEL_PATH        = os.path.expanduser("~/models/phi3_q4/Phi-3-mini-4k-instruct-q4.gguf")
LLAMA_PERPLEXITY  = "./llama-perplexity"
WIKI_RAW          = os.path.expanduser("~/wiki.test.raw")
WIKI_SMALL        = os.path.expanduser("~/wiki.test.small.raw")
RESULTS_DIR       = Path("results/raw")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def prepare_corpus(lines):
    if not Path(WIKI_RAW).exists():
        print(f"ERROR: {WIKI_RAW} not found. Run setup.sh.", file=sys.stderr)
        sys.exit(1)
    with open(WIKI_RAW) as f:
        content = f.readlines()[:lines]
    with open(WIKI_SMALL, "w") as f:
        f.writelines(content)
    print(f"Corpus: {WIKI_SMALL} ({lines} lines)")
    return WIKI_SMALL


def parse_output(text):
    results = {}
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", text)
    if m:
        results["ppl"]             = float(m.group(1))
        results["ppl_uncertainty"] = float(m.group(2))
    chunks = re.findall(r"\[(\d+)\]([\d.]+)", text)
    if chunks:
        results["chunks"]      = len(chunks)
        results["chunk_ppls"]  = ",".join(v for _, v in chunks)
    m = re.search(r"load time\s*=\s*([\d.]+)\s*ms", text)
    if m:
        results["load_time_ms"] = float(m.group(1))
    m = re.search(r"prompt eval time\s*=.*?([\d.]+)\s*tokens per second", text)
    if m:
        results["prompt_eval_tps"] = float(m.group(1))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lines",    type=int, default=200)
    parser.add_argument("--ctx-size", type=int, default=512, dest="ctx_size")
    args = parser.parse_args()

    if not Path(LLAMA_PERPLEXITY).exists():
        print(f"ERROR: {LLAMA_PERPLEXITY} not found. Run setup.sh.", file=sys.stderr)
        sys.exit(1)
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    corpus = prepare_corpus(args.lines)
    cmd = [LLAMA_PERPLEXITY, "-m", MODEL_PATH, "-f", corpus, "--ctx-size", str(args.ctx_size)]
    print(f"Running: {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    parsed = parse_output(proc.stdout + proc.stderr)
    result = {
        "timestamp":  datetime.now().isoformat(),
        "model_name": MODEL_NAME,
        "ctx_size":   args.ctx_size,
        **{k: v for k, v in parsed.items() if k != "chunk_ppls"},
    }

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"perplexity_{ts}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    print(f"\n{'='*50}")
    print(f"  Model : {MODEL_NAME}")
    print(f"  PPL   : {parsed.get('ppl','N/A')} ± {parsed.get('ppl_uncertainty','N/A')}")
    print(f"  Chunks: {parsed.get('chunks','N/A')}")
    print(f"{'='*50}")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
