# Demo UI — Edge LLM Benchmark Dashboard

An interactive Streamlit dashboard for running live inference with any quantisation level of Llama 3.2 3B Instruct and viewing real-time performance metrics.

---

## Prerequisites

Complete the setup in the root [`README.md`](../README.md) first — llama.cpp must be built and at least one model downloaded before the UI will work.

---

## Setup

Install the required Python packages:

```bash
pip install streamlit requests sseclient-py psutil pandas
```

---

## Running the UI

```bash
streamlit run demo_ui/demo_ui.py
```

---

## Usage

1. Select a quantisation level from the dropdown and click **Load Model** — this starts `llama-server` in the background. Wait for the success message before continuing.
2. Enter a prompt in the **Enter Prompt** box and click **Run Prompt**.
3. Output streams in real time alongside live charts for generation speed, prefill speed, memory usage, and efficiency.

---

## Metrics

| Metric | Source |
|--------|--------|
| Prefill Speed (t/s) | Measured live via token timing |
| Generation Speed (t/s) | Measured live via token timing |
| Tokens Generated | Counted from stream |
| Peak Memory (MB) | Process RSS via `psutil` |
| Power (W) | **Fixed estimate of 5.0 W** — not a live measurement |
| Tokens/Joule | Derived from the 5.0 W estimate above |

> **Note:** Power and tokens/joule values in the UI are approximations based on a hardcoded 5 W estimate. For accurate power measurements use `run_sweep.py` with `sudo`, which reads live data from `powermetrics`.

---

## Known Issues

- Switching quantisation levels or clearing output may cause some metric displays to reset to their default values. Clicking **Load Model** again after switching resolves this.
