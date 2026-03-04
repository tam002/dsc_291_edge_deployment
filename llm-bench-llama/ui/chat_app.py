"""
ui/chat_app.py — llm-edge-benchmark demo UI
Llama 3.2 3B Instruct — Quantisation Explorer

Features:
- Select quantisation level (Q2_K → Q8_0) live
- Select backend: llama.cpp (Metal) or MLX
- Real-time metrics: pp t/s, tg t/s, TTFT, peak RAM, tokens/joule, pkg power
- Sweep results table loaded from results/raw/sweep_*.csv if present
"""

import csv
import glob
import os
import re
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import gradio as gr

# ── Model config ──────────────────────────────────────────────────────────────

MODEL_DIR   = os.path.expanduser("~/models/llama3.2_3b")
MLX_DIR     = os.path.expanduser("~/models/llama3.2_3b_mlx")
PREFIX      = "Llama-3.2-3B-Instruct"
ALL_QUANTS  = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
CHAT_FORMAT = "llama-3"
MODEL_LABEL = "Llama 3.2 3B Instruct"
BATTERY_V   = 11.4

# ── Runtime state ─────────────────────────────────────────────────────────────

_llm        = None
_loaded_key = None   # "Q4_K_M|llamacpp" or "mlx"
_lock       = threading.Lock()


def gguf_path(quant):
    return os.path.join(MODEL_DIR, f"{PREFIX}-{quant}.gguf")


def available_quants():
    return [q for q in ALL_QUANTS if Path(gguf_path(q)).exists()]


def get_peak_ram_mib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 * 1024), 1)  # macOS: bytes


def load_sweep_results():
    """Load latest sweep CSV if present. Uses path relative to this file's location."""
    # Use __file__ so this works regardless of which directory the user runs from
    base     = Path(__file__).parent.parent  # ui/ -> project root
    pattern  = str(base / "results" / "raw" / "sweep_*.csv")
    files    = sorted(glob.glob(pattern), reverse=True)
    if not files:
        return None
    return list(csv.DictReader(open(files[0])))


# ── Backend: llama.cpp ────────────────────────────────────────────────────────

def load_llamacpp(quant):
    global _llm, _loaded_key
    try:
        from llama_cpp import Llama
    except ImportError:
        raise RuntimeError("llama-cpp-python not installed. Run: pip install llama-cpp-python")

    mpath = gguf_path(quant)
    if not Path(mpath).exists():
        raise FileNotFoundError(f"Model not found: {mpath}\nRun: bash models/download_models.sh --quant {quant}")

    key = f"{quant}|llamacpp"
    with _lock:
        if _loaded_key != key:
            _llm = Llama(
                model_path=mpath,
                n_ctx=4096,
                n_threads=8,
                n_gpu_layers=-1,   # full Metal offload on M1
                chat_format=CHAT_FORMAT,
                verbose=False,
            )
            _loaded_key = key
    return _llm


def load_mlx_model():
    """
    Returns (model, tokenizer) tuple.
    Caches both in _llm as a tuple — callers unpack with: model, tokenizer = load_mlx_model()
    Bug fix: previously stored load() result in _llm then returned _llm, but load_mlx_model()
    callers tried to unpack it as model, tokenizer = load_mlx_model() which would have worked
    only if the function returned the tuple directly — which it now does explicitly.
    """
    global _llm, _loaded_key
    try:
        from mlx_lm import load
    except ImportError:
        raise RuntimeError("MLX not installed. Run: pip install mlx mlx-lm")

    if not Path(MLX_DIR).exists():
        raise FileNotFoundError(f"MLX model not found at {MLX_DIR}\nRun: python mlx/run_mlx_benchmark.py first")

    with _lock:
        if _loaded_key != "mlx":
            _llm = load(MLX_DIR)   # returns (model, tokenizer) tuple
            _loaded_key = "mlx"
    return _llm   # caller unpacks as: model, tokenizer = load_mlx_model()


def chat_llamacpp(message, history, quant, max_tokens, temperature):
    llm = load_llamacpp(quant)
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": message})

    full_resp = ""
    t_start   = time.perf_counter()
    t_first   = None
    n_tokens  = 0

    for chunk in llm.create_chat_completion(messages=messages,
                                             max_tokens=max_tokens,
                                             temperature=temperature,
                                             stream=True):
        delta = chunk["choices"][0]["delta"].get("content", "")
        if delta:
            if t_first is None:
                t_first = time.perf_counter()
            full_resp += delta
            n_tokens  += 1
            elapsed    = time.perf_counter() - t_start
            tg_tps     = n_tokens / elapsed if elapsed > 0 else 0
            ttft_ms    = (t_first - t_start) * 1000 if t_first else 0
            ram        = get_peak_ram_mib()

            yield history + [[message, full_resp]], metrics_html({
                "backend": f"llama.cpp Metal · {quant}",
                "tg_tps":  tg_tps,
                "tokens":  n_tokens,
                "ttft_ms": ttft_ms,
                "ram":     ram,
                "elapsed": elapsed,
            })


def chat_mlx(message, history, max_tokens, temperature):
    """
    MLX chat with streaming via mlx_lm's built-in stream_generate.
    Previously used generate() and yielded only once at the end — inconsistent
    with llama.cpp's token-by-token streaming. Now streams each token.
    """
    from mlx_lm import stream_generate
    model, tokenizer = load_mlx_model()

    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    full_resp = ""
    t_start   = time.perf_counter()
    t_first   = None
    n_tokens  = 0

    for token_text in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        if t_first is None:
            t_first = time.perf_counter()
        full_resp += token_text
        n_tokens  += 1
        elapsed    = time.perf_counter() - t_start
        tg_tps     = n_tokens / elapsed if elapsed > 0 else 0
        ttft_ms    = (t_first - t_start) * 1000 if t_first else 0

        yield history + [[message, full_resp]], metrics_html({
            "backend": "MLX (GPU/ANE)",
            "tg_tps":  tg_tps,
            "tokens":  n_tokens,
            "ttft_ms": ttft_ms,
            "ram":     get_peak_ram_mib(),
            "elapsed": elapsed,
        })


# ── Metrics HTML ──────────────────────────────────────────────────────────────

def metrics_html(m):
    ttft = f"{m['ttft_ms']:.0f} ms" if m.get("ttft_ms") else "—"
    return f"""
<div class="metrics-grid">
  <div class="mc full"><div class="ml">BACKEND</div>
    <div class="mv accent2">{m.get('backend','—')}</div></div>
  <div class="mc"><div class="ml">TG SPEED</div>
    <div class="mv hi">{m.get('tg_tps',0):.1f} <span class="unit">t/s</span></div></div>
  <div class="mc"><div class="ml">TOKENS OUT</div>
    <div class="mv">{m.get('tokens',0)}</div></div>
  <div class="mc"><div class="ml">TTFT</div>
    <div class="mv">{ttft}</div></div>
  <div class="mc"><div class="ml">PEAK RAM</div>
    <div class="mv">{m.get('ram',0):.0f} <span class="unit">MiB</span></div></div>
  <div class="mc"><div class="ml">ELAPSED</div>
    <div class="mv">{m.get('elapsed',0):.1f} <span class="unit">s</span></div></div>
</div>"""


def sweep_table_html():
    rows = load_sweep_results()
    if not rows:
        return "<div class='no-data'>Run <code>python benchmarks/run_sweep.py</code> to populate this table.</div>"

    cols = ["quant", "size_gib", "pp_tps", "tg_tps", "peak_ram_mib", "ppl", "tokens_per_joule"]
    labels = ["Quant", "Size (GiB)", "pp t/s", "tg t/s", "RAM (MiB)", "PPL ↓", "tok/J ↑"]

    def fmt(row, col):
        v = row.get(col, "")
        if v == "" or v is None:
            return "—"
        try:
            f = float(v)
            return f"{f:.2f}" if col not in ["peak_ram_mib"] else f"{f:.0f}"
        except:
            return v

    thead = "".join(f"<th>{l}</th>" for l in labels)
    tbody = ""
    for row in rows:
        cells = "".join(f"<td>{fmt(row, c)}</td>" for c in cols)
        tbody += f"<tr>{cells}</tr>"

    return f"""
<div class='sweep-wrap'>
  <div class='sweep-title'>Quantisation Sweep Results</div>
  <table class='sweep-table'><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>
</div>"""


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

:root {
  --bg:      #080b10;
  --s1:      #0e1219;
  --s2:      #151b26;
  --border:  #1e2738;
  --accent:  #00e5ff;
  --accent2: #7c9ef8;
  --green:   #39d98a;
  --text:    #dde4f0;
  --muted:   #4a5568;
  --mono:    'IBM Plex Mono', monospace;
  --sans:    'IBM Plex Sans', sans-serif;
}

body, .gradio-container { background: var(--bg) !important; font-family: var(--sans) !important; color: var(--text) !important; }

.header { padding: 18px 0 14px; border-bottom: 1px solid var(--border); margin-bottom: 18px; }
.header-top { display: flex; align-items: baseline; gap: 10px; }
.header h1 { font-family: var(--mono); font-size: 1.15rem; font-weight: 600; color: var(--accent); margin: 0; letter-spacing: -0.3px; }
.header .sub { font-family: var(--mono); font-size: 0.7rem; color: var(--muted); }
.model-tag { display: inline-block; margin-top: 6px; font-family: var(--mono); font-size: 0.65rem; color: var(--accent2); background: rgba(124,158,248,0.08); border: 1px solid rgba(124,158,248,0.2); border-radius: 4px; padding: 2px 8px; }

/* Chatbot */
.chatbot-wrap .wrap { background: var(--s1) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
.chatbot-wrap .message.user { background: var(--s2) !important; border: 1px solid var(--border) !important; color: var(--text) !important; font-family: var(--sans) !important; border-radius: 8px !important; }
.chatbot-wrap .message.bot { background: transparent !important; color: var(--text) !important; font-family: var(--sans) !important; }

/* Metrics */
.metrics-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; padding: 2px; }
.mc { background: var(--s2); border: 1px solid var(--border); border-radius: 8px; padding: 9px 12px; }
.mc.full { grid-column: span 3; }
.ml { font-family: var(--mono); font-size: 0.58rem; font-weight: 600; color: var(--muted); letter-spacing: 1.2px; margin-bottom: 4px; }
.mv { font-family: var(--mono); font-size: 1.0rem; font-weight: 600; color: var(--text); }
.mv.hi { color: var(--accent); font-size: 1.25rem; }
.mv.accent2 { color: var(--accent2); font-size: 0.78rem; }
.mv .unit { font-size: 0.65rem; color: var(--muted); font-weight: 400; }

/* Controls */
label { font-family: var(--mono) !important; font-size: 0.68rem !important; color: var(--muted) !important; letter-spacing: 0.8px !important; text-transform: uppercase !important; }
.gradio-dropdown > div, textarea, input { background: var(--s1) !important; border: 1px solid var(--border) !important; color: var(--text) !important; font-family: var(--sans) !important; border-radius: 8px !important; }
textarea:focus, input:focus { border-color: var(--accent) !important; outline: none !important; }

/* Buttons */
button.primary { background: var(--accent) !important; color: #080b10 !important; font-family: var(--mono) !important; font-weight: 600 !important; font-size: 0.72rem !important; border: none !important; border-radius: 7px !important; }
button.secondary { background: transparent !important; border: 1px solid var(--border) !important; color: var(--muted) !important; font-family: var(--mono) !important; font-size: 0.7rem !important; border-radius: 7px !important; }

/* Backend toggle */
.backend-row { display: flex; gap: 8px; margin-top: 4px; }
.backend-btn { flex: 1; padding: 8px; border-radius: 6px; border: 1px solid var(--border); background: var(--s2); color: var(--muted); font-family: var(--mono); font-size: 0.68rem; cursor: pointer; transition: all 0.15s; }
.backend-btn.active { border-color: var(--accent); color: var(--accent); background: rgba(0,229,255,0.06); }

/* Sweep table */
.sweep-wrap { margin-top: 6px; }
.sweep-title { font-family: var(--mono); font-size: 0.65rem; color: var(--muted); letter-spacing: 1px; margin-bottom: 8px; text-transform: uppercase; }
.sweep-table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 0.75rem; }
.sweep-table th { color: var(--muted); font-weight: 600; font-size: 0.6rem; letter-spacing: 0.8px; text-transform: uppercase; padding: 6px 10px; border-bottom: 1px solid var(--border); text-align: right; }
.sweep-table th:first-child { text-align: left; }
.sweep-table td { padding: 7px 10px; border-bottom: 1px solid rgba(30,39,56,0.6); color: var(--text); text-align: right; }
.sweep-table td:first-child { text-align: left; color: var(--accent2); font-weight: 600; }
.sweep-table tr:hover td { background: var(--s2); }
.no-data { font-family: var(--mono); font-size: 0.72rem; color: var(--muted); text-align: center; padding: 20px; background: var(--s1); border: 1px solid var(--border); border-radius: 8px; }
.no-data code { color: var(--accent); }
"""

# ── UI Build ──────────────────────────────────────────────────────────────────

def build_ui():
    quants = available_quants() or ALL_QUANTS

    with gr.Blocks(css=CSS, title=f"llm-edge · {MODEL_LABEL}") as app:

        gr.HTML(f"""
        <div class="header">
          <div class="header-top">
            <h1>llm-edge-benchmark</h1>
            <span class="sub">// {MODEL_LABEL}</span>
          </div>
          <span class="model-tag">M1 Mac · llama.cpp Metal + MLX · Quant Explorer</span>
        </div>""")

        with gr.Row():
            # ── Left: chat ────────────────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="", height=480,
                                     elem_classes=["chatbot-wrap"], show_label=False)

                with gr.Row():
                    msg_box   = gr.Textbox(placeholder="Send a message…",
                                           show_label=False, scale=5, lines=1)
                    send_btn  = gr.Button("SEND",  variant="primary",    scale=1)
                    clear_btn = gr.Button("CLEAR", variant="secondary",  scale=1)

                # Sweep results table
                sweep_html = gr.HTML(value=sweep_table_html())
                refresh_btn = gr.Button("↻ Refresh sweep results", variant="secondary", size="sm")

            # ── Right: controls + metrics ─────────────────────────────────────
            with gr.Column(scale=1):
                backend_dd = gr.Dropdown(
                    choices=["llama.cpp (Metal)", "MLX (GPU/ANE)"],
                    value="llama.cpp (Metal)",
                    label="BACKEND",
                )
                quant_dd = gr.Dropdown(
                    choices=quants,
                    value=quants[-2] if len(quants) >= 2 else quants[0],
                    label="QUANTISATION",
                    info="Disabled when using MLX",
                )
                temp_sl = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="TEMPERATURE")
                max_tok = gr.Slider(64, 1024, value=512, step=64,    label="MAX TOKENS")

                gr.HTML("<div style='height:10px'></div>")
                gr.HTML("<div style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;color:#4a5568;letter-spacing:1px;margin-bottom:6px;text-transform:uppercase'>Live Metrics</div>")
                metrics_box = gr.HTML(
                    value="<div class='no-data' style='padding:12px'>Waiting for first message…</div>"
                )

        # ── Wiring ────────────────────────────────────────────────────────────

        def on_send(message, history, backend, quant, temperature, max_tokens):
            if not message.strip():
                yield history, "", metrics_box.value
                return
            if "MLX" in backend:
                gen = chat_mlx(message, history, int(max_tokens), temperature)
            else:
                gen = chat_llamacpp(message, history, quant, int(max_tokens), temperature)
            for h, m in gen:
                yield h, "", m

        def on_backend_change(backend):
            return gr.update(interactive="llama.cpp" in backend)

        def on_refresh():
            return sweep_table_html()

        send_btn.click(
            fn=on_send,
            inputs=[msg_box, chatbot, backend_dd, quant_dd, temp_sl, max_tok],
            outputs=[chatbot, msg_box, metrics_box],
        )
        msg_box.submit(
            fn=on_send,
            inputs=[msg_box, chatbot, backend_dd, quant_dd, temp_sl, max_tok],
            outputs=[chatbot, msg_box, metrics_box],
        )
        clear_btn.click(fn=lambda: ([], []), outputs=[chatbot])
        backend_dd.change(fn=on_backend_change, inputs=backend_dd, outputs=quant_dd)
        refresh_btn.click(fn=on_refresh, outputs=sweep_html)

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_ui().launch(server_port=args.port, share=args.share)
