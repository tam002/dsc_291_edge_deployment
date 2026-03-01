# llm-edge-benchmark — Llama 3.2 3B

> Benchmarking Llama 3.2 3B Instruct (Q4_K_M GGUF) for edge/CPU deployment using llama.cpp.

This repo reproduces throughput, memory, perplexity, and reasoning benchmarks for **Llama 3.2 3B Instruct Q4_K_M**, and provides a live chat UI with real-time inference metrics.

**Why Llama 3.2 3B?** In Phase 1 testing it was 52% faster at prompt processing, 20% faster at token generation, and used 18% less RAM compared to Phi-3 Mini — making it the best fit for latency-sensitive and resource-constrained deployments.

---

## Hardware Tested On

| Component | Spec |
|-----------|------|
| CPU       | 24-thread (12 used for inference) |
| Backend   | CPU only (no GPU offload) |
| OS        | Linux x86_64 |
| SIMD      | AVX2, F16C, FMA, BMI2 |
| llama.cpp | build 8184 (319146247) |

---

## Quick Start

```bash
git clone https://github.com/tam002/dsc_291_edge_deployment
cd dsc_291_edge_deployment
bash setup.sh
bash models/download_models.sh
```

---

## Reproducing the Benchmarks

### 1. Throughput (tokens/sec + RAM)

```bash
python benchmarks/run_performance.py
```

### 2. Perplexity

```bash
python benchmarks/run_perplexity.py --lines 200
```

### 3. GSM8K (Math Reasoning)

```bash
python benchmarks/run_gsm8k.py --shots 8
```

### 4. HumanEval (Code Generation)

```bash
python benchmarks/run_humaneval.py
```

---

## Phase 1 Results

| Metric | Llama 3.2 3B Q4_K_M |
|--------|----------------------|
| Prompt processing (pp512) | **110.43 ± 0.36 t/s** |
| Token generation (tg128) | **19.22 ± 0.14 t/s** |
| Peak RAM | **2,114 MiB** |
| Perplexity (wiki, ctx=512) | 12.33 ± 0.51 |

See `notebooks/analysis.ipynb` for full analysis and plots.

---

## Project Structure

```
llm-edge-benchmark/
├── README.md
├── setup.sh
├── requirements.txt
├── models/
│   └── download_models.sh
├── benchmarks/
│   ├── run_performance.py
│   ├── run_perplexity.py
│   ├── run_gsm8k.py
│   └── run_humaneval.py
├── quantization/
│   └── quantize_model.sh
├── results/
│   ├── raw/
│   └── plots/
├── ui/
│   └── chat_app.py
└── notebooks/
    └── analysis.ipynb
```

---

## License

MIT
