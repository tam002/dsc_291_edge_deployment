"""
benchmarks/run_humaneval.py
Evaluates Phi-3 Mini 3.82B Q4_K_M on HumanEval (pass@1).

Usage:
    python benchmarks/run_humaneval.py
    python benchmarks/run_humaneval.py --samples 50
"""

import argparse
import ast
import csv
import io
import os
import re
import subprocess
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: pip install datasets", file=sys.stderr); sys.exit(1)

MODEL_NAME    = "Phi-3 Mini 3.82B Q4_K_M"
MODEL_PATH    = os.path.expanduser("~/models/phi3_q4/Phi-3-mini-4k-instruct-q4.gguf")
CHAT_TEMPLATE = "<|user|>\n{prompt}<|end|>\n<|assistant|>"
LLAMA_CLI     = "./llama-cli"
RESULTS_DIR   = Path("results/raw")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM = ("You are an expert Python programmer. Complete the function below. "
          "Return ONLY the function body, no explanations or markdown.")


def extract_code(raw, prompt):
    raw = re.sub(r"```python\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    return raw.strip()


def query(prompt):
    user_msg  = f"{SYSTEM}\n\n{prompt}"
    formatted = CHAT_TEMPLATE.format(prompt=user_msg)
    cmd = [LLAMA_CLI, "-m", MODEL_PATH, "--prompt", formatted,
           "--n-predict", "512", "--temp", "0.2",
           "--no-display-prompt", "-t", "12", "--log-disable"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return proc.stdout.strip()


def safe_exec(code, test, entry_point):
    ns = {}
    full = code + "\n\n" + test + f"\n\ncheck({entry_point})\n"
    try:
        ast.parse(full)
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            exec(compile(full, "<string>", "exec"), ns)  # noqa: S102
        return True, ""
    except AssertionError as e:
        return False, str(e)
    except Exception:
        return False, traceback.format_exc(limit=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    if not Path(LLAMA_CLI).exists():
        print(f"ERROR: {LLAMA_CLI} not found.", file=sys.stderr); sys.exit(1)
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found at {MODEL_PATH}", file=sys.stderr); sys.exit(1)

    print(f"Loading HumanEval ({args.samples} tasks)...")
    dataset = load_dataset("openai_humaneval", split="test")
    samples = list(dataset.select(range(min(args.samples, len(dataset)))))

    passed, rows = 0, []
    for i, s in enumerate(samples):
        raw     = query(s["prompt"])
        sol     = s["prompt"] + "\n" + extract_code(raw, s["prompt"])
        ok, err = safe_exec(sol, s["test"], s["entry_point"])
        if ok: passed += 1
        rows.append({"idx": i, "task_id": s["task_id"], "passed": ok,
                     "error": err[:150] if err else ""})
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(samples)} | pass@1: {passed/(i+1):.1%}")

    pass_at_1 = passed / len(samples)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = RESULTS_DIR / f"humaneval_{ts}.csv"
    summary   = {"timestamp": datetime.now().isoformat(), "model_name": MODEL_NAME,
                 "n_samples": len(samples), "passed": passed,
                 "pass_at_1": round(pass_at_1, 4)}
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader(); writer.writerow(summary)

    print(f"\n{'='*50}")
    print(f"  Model     : {MODEL_NAME}")
    print(f"  HumanEval : pass@1 = {pass_at_1:.1%} ({passed}/{len(samples)})")
    print(f"{'='*50}\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
