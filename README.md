# llm-edge-benchmark — Llama 3.2 3B Instruct

> Edge deployment benchmark for Llama 3.2 3B Instruct on Apple Silicon.  
> Frameworks: **llama.cpp (Metal)** and **MLX**.  
> Primary metric: **tokens/joule** — throughput per unit of battery drain.

**Why this model?** Fastest throughput and lowest memory footprint in Phase 1 head-to-head testing — best for latency-sensitive and memory-constrained edge deployments.

---

## Hardware Target

| Component | Spec |
|-----------|------|
| Device    | MacBook Pro 14" 2021 — M1 Pro, 16 GB |
| Backend 1 | llama.cpp with Metal GPU offload |
| Backend 2 | Apple MLX (GPU + ANE) |
| OS        | macOS 13+ |

---

## Setup

### 1. Prerequisites

```bash
# Install Homebrew if not already present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install cmake git python3
```

### 2. Build llama.cpp with Metal

```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cmake -B ~/llama.cpp/build ~/llama.cpp \
    -DGGML_METAL=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build ~/llama.cpp/build --config Release -j $(sysctl -n hw.logicalcpu)

# Symlink binaries into the project root
for bin in llama-bench llama-perplexity llama-cli; do
    ln -sf ~/llama.cpp/build/bin/$bin ./$bin
done
```

### 3. Install Python dependencies

```bash
pip3 install -r requirements.txt
pip3 install mlx mlx-lm
```

### 4. Download WikiText-2 (for perplexity scoring)

```bash
curl -L "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" \
     -o /tmp/wikitext.zip
unzip -o /tmp/wikitext.zip -d /tmp/wikitext/
cp /tmp/wikitext/wikitext-2-raw/wiki.test.raw ~/wiki.test.raw
```

### 5. Download models

All six quantisation levels from bartowski's HuggingFace repo (imatrix-calibrated GGUFs):

```bash
BASE="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main"
mkdir -p ~/models/llama3.2_3b

for Q in Q2_K Q3_K_M Q4_K_M Q5_K_M Q6_K Q8_0; do
    curl -L --progress-bar "$BASE/Llama-3.2-3B-Instruct-${Q}.gguf" \
         -o ~/models/llama3.2_3b/Llama-3.2-3B-Instruct-${Q}.gguf
done
```

To download a single level only:

```bash
curl -L --progress-bar "$BASE/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
     -o ~/models/llama3.2_3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

---

## Running Benchmarks

```bash
# Full sweep — all 6 quant levels, PPL + GSM8K + power (unplug laptop first)
sudo python benchmarks/run_sweep.py

# Skip perplexity for a faster run
sudo python benchmarks/run_sweep.py --skip-ppl

# Skip GSM8K accuracy scoring
sudo python benchmarks/run_sweep.py --skip-gsm8k

# Single quant level only
sudo python benchmarks/run_sweep.py --quants Q4_K_M

# MLX backend benchmark
sudo python mlx/run_mlx_benchmark.py

# Launch the demo UI
python ui/chat_app.py
```

> `sudo` is required for `powermetrics` — the tool that measures CPU/GPU/ANE power draw.  
> Run without `sudo` to get throughput and RAM only (power fields will be blank).

---

## Experiment Plan

### Step 1 — Quantisation Sweep (llama.cpp Metal)

| Quant  | Size (GiB) | PPL ↓        | pp t/s ↑         | tg t/s ↑        | RAM (MiB) ↓ | GSM8K % ↑ | tok/J ↑ |
|--------|-----------|--------------|------------------|-----------------|------------|-----------|---------|
| Q2_K   | TBD       | TBD          | TBD              | TBD             | TBD        | TBD       | TBD     |
| Q3_K_M | TBD       | TBD          | TBD              | TBD             | TBD        | TBD       | TBD     |
| Q4_K_M | 1.87      | 12.33 ± 0.51 | 110.43 ± 0.36    | 19.22 ± 0.14    | 2,114      | TBD       | TBD     |
| Q5_K_M | TBD       | TBD          | TBD              | TBD             | TBD        | TBD       | TBD     |
| Q6_K   | TBD       | TBD          | TBD              | TBD             | TBD        | TBD       | TBD     |
| Q8_0   | TBD       | TBD          | TBD              | TBD             | TBD        | TBD       | TBD     |

### Step 2 — MLX Backend Comparison

Take the best quant level from Step 1 (highest tok/J) and compare frameworks:

| Backend         | pp t/s | tg t/s | RAM (MiB) | tok/J |
|-----------------|--------|--------|-----------|-------|
| llama.cpp Metal | —      | —      | —         | —     |
| MLX (GPU/ANE)   | —      | —      | —         | —     |

---

## Project Structure

```
llm-edge-benchmark/
├── README.md
├── requirements.txt
├── benchmarks/
│   └── run_sweep.py             # Full quant sweep (PPL + GSM8K + throughput + power)
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
