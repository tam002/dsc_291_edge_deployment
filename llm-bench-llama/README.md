# llm-edge-benchmark — Llama 3.2 3B Instruct

> Edge deployment benchmark for Llama 3.2 3B Instruct on Apple Silicon (M1).  
> Frameworks: **llama.cpp (Metal)** and **MLX**.  
> Primary metric: **tokens/joule** — throughput per unit of battery drain.

**Why this model?** Fastest throughput and lowest memory footprint in Phase 1 head-to-head testing — best for latency-sensitive and memory-constrained edge deployments.

---

## Hardware Target

| Component | Spec |
|-----------|------|
| Device    | M1 MacBook (Air or Pro) |
| Backend 1 | llama.cpp with Metal GPU offload |
| Backend 2 | Apple MLX (GPU + ANE) |
| OS        | macOS 13+ |

---

## Quick Start

```bash
# 1. Install dependencies + build llama.cpp with Metal
bash setup.sh

# 2. Download all quantisation levels (Q2_K → Q8_0)
bash models/download_models.sh

# 3. Run the full quantisation sweep (unplug for power measurements)
sudo python benchmarks/run_sweep.py

# 4. Run MLX benchmark
python mlx/run_mlx_benchmark.py

# 5. Launch the demo UI
python ui/chat_app.py
```

---

## Experiment Plan

### Step 1 — Quantisation Sweep (llama.cpp Metal)
Run all six quantisation levels and record:

| Quant | Size (GiB) | PPL ↓ | pp t/s ↑ | tg t/s ↑ | RAM (MiB) ↓ | tok/J ↑ |
|-------|-----------|-------|---------|---------|------------|--------|
| Q4_K_M | 1.87 | 12.33 ± 0.51 | 110.43 ± 0.36 | 19.22 ± 0.14 | 2,114 | TBD |
| Q2_K  | TBD | TBD | TBD | TBD | TBD | TBD |
| Q3_K_M| TBD | TBD | TBD | TBD | TBD | TBD |
| Q5_K_M| TBD | TBD | TBD | TBD | TBD | TBD |
| Q6_K  | TBD | TBD | TBD | TBD | TBD | TBD |
| Q8_0  | TBD | TBD | TBD | TBD | TBD | TBD |

Q4_K_M populated from Phase 1 results. Run `run_sweep.py` to fill remaining rows.

### Step 2 — MLX Backend Comparison
Take the best quantisation level from Step 1 (highest tok/J) and compare:

| Backend | pp t/s | tg t/s | RAM (MiB) | tok/J |
|---------|--------|--------|-----------|-------|
| llama.cpp Metal | — | — | — | — |
| MLX (GPU/ANE)   | — | — | — | — |

---

## Reproducing Results

```bash
# Full sweep (needs sudo for powermetrics / battery drain)
sudo python benchmarks/run_sweep.py

# Single quant level only
sudo python benchmarks/run_sweep.py --quants Q4_K_M

# Skip perplexity (much faster, throughput + power only)
sudo python benchmarks/run_sweep.py --skip-ppl

# MLX benchmark
python mlx/run_mlx_benchmark.py
```

---

## Project Structure

```
llm-edge-benchmark/
├── README.md
├── setup.sh                     # llama.cpp Metal build + MLX install
├── requirements.txt
├── models/
│   └── download_models.sh       # Q2_K → Q8_0 from bartowski/HuggingFace
├── benchmarks/
│   └── run_sweep.py             # Full quant sweep (PPL + throughput + power)
├── mlx/
│   └── run_mlx_benchmark.py     # MLX backend benchmark
├── results/
│   ├── raw/                     # CSV outputs from each run
│   └── plots/                   # Generated figures
├── ui/
│   └── chat_app.py              # Live chat UI with quant/backend selector
└── notebooks/
    └── analysis.ipynb           # Sweep analysis + MLX comparison plots
```

---

## License

MIT
