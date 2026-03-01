"""
benchmarks/run_gsm8k.py
Evaluates Phi-3 Mini 3.82B Q4_K_M on GSM8K (few-shot math reasoning).

Usage:
    python benchmarks/run_gsm8k.py
    python benchmarks/run_gsm8k.py --shots 8 --samples 100
"""

import argparse
import csv
import os
import re
import subprocess
import sys
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

FEW_SHOT = [
    {"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "answer": "72"},
    {"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "answer": "10"},
    {"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?", "answer": "5"},
]


def build_prompt(question, n_shots):
    p = "Solve the following math problems step by step. At the end write the final answer as: #### <number>\n\n"
    for ex in FEW_SHOT[:n_shots]:
        p += f"Question: {ex['question']}\nAnswer: #### {ex['answer']}\n\n"
    p += f"Question: {question}\nAnswer:"
    return p


def extract_answer(text):
    matches = re.findall(r"####\s*([\d,.-]+)", text)
    if matches:
        return matches[-1].replace(",", "").strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        m = re.search(r"([\d,.-]+)$", lines[-1])
        if m:
            return m.group(1).replace(",", "").strip()
    return None


def normalise(v):
    try: return str(float(v))
    except: return str(v).strip().lower()


def query(prompt):
    formatted = CHAT_TEMPLATE.format(prompt=prompt)
    cmd = [LLAMA_CLI, "-m", MODEL_PATH, "--prompt", formatted,
           "--n-predict", "256", "--temp", "0", "--no-display-prompt",
           "-t", "12", "--log-disable"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return proc.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots",   type=int, default=8)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    if not Path(LLAMA_CLI).exists():
        print(f"ERROR: {LLAMA_CLI} not found.", file=sys.stderr); sys.exit(1)
    if not Path(MODEL_PATH).exists():
        print(f"ERROR: Model not found at {MODEL_PATH}", file=sys.stderr); sys.exit(1)

    print(f"Loading GSM8K ({args.samples} samples, {args.shots}-shot)...")
    dataset = load_dataset("gsm8k", "main", split="test")
    samples = list(dataset.select(range(min(args.samples, len(dataset)))))

    correct, rows = 0, []
    for i, sample in enumerate(samples):
        gold_m  = re.search(r"####\s*([\d,.-]+)", sample["answer"])
        gold    = normalise(gold_m.group(1).replace(",", "")) if gold_m else ""
        output  = query(build_prompt(sample["question"], args.shots))
        pred    = normalise(extract_answer(output)) if extract_answer(output) else ""
        ok      = pred == gold
        if ok: correct += 1
        rows.append({"idx": i, "gold": gold, "pred": pred, "correct": ok})
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(samples)} | acc: {correct/(i+1):.1%}")

    accuracy = correct / len(samples)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"gsm8k_{ts}.csv"
    summary  = {"timestamp": datetime.now().isoformat(), "model_name": MODEL_NAME,
                "n_shots": args.shots, "n_samples": len(samples),
                "correct": correct, "accuracy": round(accuracy, 4)}
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader(); writer.writerow(summary)

    print(f"\n{'='*50}")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  GSM8K   : {accuracy:.1%} ({correct}/{len(samples)})")
    print(f"{'='*50}\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
