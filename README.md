# llm-edge-benchmark — Llama 3.2 3B Instruct

> Edge deployment benchmark for Llama 3.2 3B Instruct on Apple Silicon.  
> Framework: **llama.cpp (Metal)**  
> Primary metric: **tokens/joule** — throughput per unit of battery drain.

**Why this model?** Fastest throughput and lowest memory footprint in Phase 1 head-to-head testing — best for latency-sensitive and memory-constrained edge deployments.

---

## Hardware Target

| Component | Spec |
|-----------|------|
| Device    | MacBook with Apple M1, 8 GB unified memory |
| Backend   | llama.cpp with Metal GPU offload |
| OS        | macOS 13+ |

---

## Setup

Create a virtual environment (Python 3.10+ required):

```bash
python3.10 -m venv venv
source venv/bin/activate
```

Clone the repository:

```bash
git clone https://github.com/tam002/dsc_291_edge_deployment.git
cd dsc_291_edge_deployment/llm-bench-llama
```

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
for bin in llama-bench llama-perplexity llama-cli llama-server; do
    ln -sf ~/llama.cpp/build/bin/$bin ./$bin
done
```

### 3. Install Python dependencies

```bash
pip3 install -r requirements.txt
```

### 4. Download WikiText-2 (for perplexity scoring)

```bash
curl -L "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" \
     -o /tmp/wikitext.zip
unzip -o /tmp/wikitext.zip -d /tmp/wikitext/
cp /tmp/wikitext/wikitext-2-raw/wiki.test.raw ~/wiki.test.raw
```

### 5. Download models

Five quantisation levels from bartowski's HuggingFace repo (imatrix-calibrated GGUFs):

```bash
bash models/download_models.sh
```

To download a single level only:

```bash
bash models/download_models.sh --quant Q4_K_M
```

---

## Running Benchmarks

> `sudo` is required for `powermetrics` — the tool that measures CPU/GPU/ANE power draw.  
> Run without `sudo` to get throughput and RAM only (power fields will be blank).

```bash
# Full sweep — all 5 quant levels, PPL + GSM8K + power (unplug laptop first)
sudo python benchmarks/run_sweep.py

# Skip perplexity for a faster run
sudo python benchmarks/run_sweep.py --skip-ppl

# Skip GSM8K accuracy scoring
sudo python benchmarks/run_sweep.py --skip-gsm8k

# Single quant level only
sudo python benchmarks/run_sweep.py --quants Q4_K_M
```

Results are saved as CSV files to `results/raw/`.

#### The current GSM8K benchmark in run_sweep.py is very inefficient! Do these steps below instead:

1. Install llm_eval:
```bash
pip install llm_eval
```

2. Open two terminals:

- Start the model on the first terminal:

    Example Usage (Replace with model of choice if necessary):
    ```bash
    /llama-server -m ~/models/llama3.2_3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf --port 8080 -ngl 0 --c 8192
    ```

- Run the evaluation on the second terminal:
    ```bash
    lm_eval --model gguf \
        --model_args base_url=http://localhost:8080/ \
        --tasks gsm8k --limit 50
    ```

> Expect runtimes to be very slow without a GPU, to run the same test with GPU, replace `-ngl 0` with `-ngl 99`

---

## Demo UI

An interactive Streamlit dashboard lets you load any quantisation level, enter a prompt, and view live metrics (generation speed, prefill speed, memory usage, tokens/joule).

### Setup

```bash
pip install streamlit requests sseclient-py psutil pandas
```

### Run

```bash
streamlit run demo_ui/demo_ui.py
```

### Usage

1. Select a quantisation level and click **Load Model** — this starts `llama-server` in the background.
2. Enter a prompt in the **Enter Prompt** box and click **Run Prompt**.
3. Model output streams in real time alongside live performance charts.

> **Note:** The power and tokens/joule values shown in the UI use a fixed 5W estimate, not live `powermetrics` data. Use `run_sweep.py` for accurate power measurements.

---

## Quantisation Levels

| Quant    | Size (GB) | Prefill (t/s) | Generation (t/s) | Peak RAM (MiB) | Avg Watts | tok/J | PPL   |
|----------|-----------|---------------|------------------|----------------|-----------|-------|-------|
| Q3_K_L   | 1.69      | 235.0         | 21.2             | 1,848          | 9.0       | 2.36  | 11.60 |
| Q4_K_M   | 1.88      | 257.1         | 25.8             | 2,260          | 10.3      | 2.51  | 11.26 |
| Q5_K_M   | 2.16      | 230.7         | 19.7             | 2,330          | 9.0       | 2.18  | 11.14 |
| Q6_K     | 2.46      | 228.3         | 16.7             | 2,637          | 7.9       | 2.10  | 11.14 |
| Q8_0     | 3.19      | 283.6         | 15.6             | 3,377          | 6.9       | 2.25  | 11.06 |

**Q4_K_M** is the recommended default — highest generation speed (25.8 t/s) and best energy efficiency (2.51 tok/J) with competitive accuracy. Use **Q8_0** if memory is not a constraint and raw quality is the priority.

---

## Project Structure

```
dsc_291_edge_deployment/
├── README.md
├── demo_ui/
│   ├── README.md                # Demo UI setup and usage notes
│   └── demo_ui.py               # Streamlit dashboard
└── llm-bench-llama/
    ├── requirements.txt
    ├── benchmarks/
    │   └── run_sweep.py         # Full quant sweep (PPL + GSM8K + throughput + power)
    └── models/
        └── download_models.sh   # Model downloader (all quants or single)
```

## License

MIT
