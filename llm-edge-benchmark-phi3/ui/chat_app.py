"""
ui/chat_app.py
Live chat UI for llm-edge-benchmark.
Runs the selected model via llama-cpp-python and shows real-time metrics:
  - tokens/sec, total tokens, TTFT, peak RAM, context usage.

Usage:
    python ui/chat_app.py
    python ui/chat_app.py --model llama32 --port 7860
"""

import argparse
import os
import resource
import threading
import time
from pathlib import Path

import gradio as gr

# ── Model registry ────────────────────────────────────────────────────────────

MODELS = {
    "phi3": {
        "name": "Phi-3 Mini 3.82B Q4_K_M",
        "path": os.path.expanduser("~/models/phi3_q4/Phi-3-mini-4k-instruct-q4.gguf"),
        "ctx": 4096,
        "chat_format": "chatml",
        "system": "You are a helpful AI assistant.",
    },
}

# ── Globals ───────────────────────────────────────────────────────────────────

_llm = None
_loaded_key = None
_lock = threading.Lock()


def get_peak_ram_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux: kB, macOS: bytes
    if os.uname().sysname == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def load_model(model_key: str):
    global _llm, _loaded_key
    from llama_cpp import Llama

    cfg = MODELS[model_key]
    model_path = cfg["path"]

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run: bash models/download_models.sh"
        )

    with _lock:
        if _loaded_key != model_key:
            _llm = Llama(
                model_path=model_path,
                n_ctx=cfg["ctx"],
                n_threads=12,
                chat_format=cfg["chat_format"],
                verbose=False,
            )
            _loaded_key = model_key

    return _llm


# ── Inference ─────────────────────────────────────────────────────────────────

def chat_stream(message: str, history: list, model_key: str, max_tokens: int, temperature: float):
    """Generator: yields (updated_history, metrics_html) on each token."""
    cfg = MODELS[model_key]

    try:
        llm = load_model(model_key)
    except FileNotFoundError as e:
        yield history + [[message, str(e)]], _metrics_html({})
        return

    messages = [{"role": "system", "content": cfg["system"]}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    # Metrics state
    tokens_generated = 0
    full_response = ""
    t_start = time.perf_counter()
    t_first_token = None
    ram_before = get_peak_ram_mb()

    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    for chunk in stream:
        delta = chunk["choices"][0]["delta"].get("content", "")
        if delta:
            if t_first_token is None:
                t_first_token = time.perf_counter()
            full_response += delta
            tokens_generated += 1
            elapsed = time.perf_counter() - t_start
            tps = tokens_generated / elapsed if elapsed > 0 else 0
            ttft_ms = (t_first_token - t_start) * 1000 if t_first_token else 0
            ram_now = get_peak_ram_mb()

            metrics = {
                "model": cfg["name"],
                "tokens": tokens_generated,
                "tps": tps,
                "ttft_ms": ttft_ms,
                "elapsed_s": elapsed,
                "ram_mb": ram_now,
                "ctx_used": len(messages) * 20,  # rough estimate
            }

            updated_history = history + [[message, full_response]]
            yield updated_history, _metrics_html(metrics)

    if not full_response:
        yield history + [[message, "(no response)"]], _metrics_html({})


def _metrics_html(m: dict) -> str:
    if not m:
        return "<div class='metrics-empty'>No active session</div>"

    return f"""
<div class="metrics-grid">
  <div class="metric-card">
    <div class="metric-label">MODEL</div>
    <div class="metric-value model-name">{m.get('model', '—')}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">TOKENS/SEC</div>
    <div class="metric-value highlight">{m.get('tps', 0):.1f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">TOKENS OUT</div>
    <div class="metric-value">{m.get('tokens', 0)}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">TTFT</div>
    <div class="metric-value">{m.get('ttft_ms', 0):.0f} ms</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">ELAPSED</div>
    <div class="metric-value">{m.get('elapsed_s', 0):.1f} s</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">PEAK RAM</div>
    <div class="metric-value">{m.get('ram_mb', 0):.0f} MB</div>
  </div>
</div>
"""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --bg: #0d0f14;
  --surface: #13161e;
  --surface2: #1a1e2a;
  --border: #252a38;
  --accent: #5af0c4;
  --accent2: #7b8cde;
  --text: #e2e8f0;
  --text-muted: #64748b;
  --danger: #f87171;
  --mono: 'JetBrains Mono', monospace;
  --sans: 'DM Sans', sans-serif;
}

body, .gradio-container {
  background: var(--bg) !important;
  font-family: var(--sans) !important;
  color: var(--text) !important;
}

.main-header {
  display: flex;
  align-items: baseline;
  gap: 12px;
  padding: 20px 0 8px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 20px;
}
.main-header h1 {
  font-family: var(--mono);
  font-size: 1.3rem;
  font-weight: 700;
  color: var(--accent);
  margin: 0;
  letter-spacing: -0.5px;
}
.main-header span {
  font-size: 0.8rem;
  color: var(--text-muted);
  font-family: var(--mono);
}

/* Chatbot */
.chatbot-wrap .wrap {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
.chatbot-wrap .message.user {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: var(--sans) !important;
  border-radius: 8px !important;
}
.chatbot-wrap .message.bot {
  background: transparent !important;
  color: var(--text) !important;
  font-family: var(--sans) !important;
}

/* Metrics panel */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  padding: 4px;
}
.metric-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 14px;
}
.metric-label {
  font-family: var(--mono);
  font-size: 0.6rem;
  font-weight: 600;
  color: var(--text-muted);
  letter-spacing: 1px;
  margin-bottom: 4px;
}
.metric-value {
  font-family: var(--mono);
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--text);
}
.metric-value.highlight {
  color: var(--accent);
  font-size: 1.3rem;
}
.metric-value.model-name {
  font-size: 0.72rem;
  color: var(--accent2);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.metrics-empty {
  font-family: var(--mono);
  font-size: 0.75rem;
  color: var(--text-muted);
  text-align: center;
  padding: 16px;
}

/* Controls */
.controls-panel label {
  font-size: 0.78rem !important;
  color: var(--text-muted) !important;
  font-family: var(--mono) !important;
  letter-spacing: 0.5px !important;
}

/* Input box */
.input-row textarea {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: var(--sans) !important;
  border-radius: 8px !important;
}
.input-row textarea:focus {
  border-color: var(--accent) !important;
  outline: none !important;
}

/* Buttons */
button.primary {
  background: var(--accent) !important;
  color: #0d0f14 !important;
  font-family: var(--mono) !important;
  font-weight: 700 !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.5px !important;
  border: none !important;
  border-radius: 7px !important;
}
button.secondary {
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--text-muted) !important;
  font-family: var(--mono) !important;
  font-size: 0.75rem !important;
  border-radius: 7px !important;
}

/* Sliders / dropdowns */
.gradio-dropdown > div,
input[type="range"] {
  background: var(--surface2) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}
"""

def build_ui(default_model: str):
    with gr.Blocks(css=CSS, title="llm-edge-benchmark — Chat") as app:

        gr.HTML("""
        <div class="main-header">
          <h1>llm-edge-benchmark</h1>
          <span>// live inference metrics</span>
        </div>
        """)

        with gr.Row():
            # ── Left: chat ────────────────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="",
                    height=520,
                    elem_classes=["chatbot-wrap"],
                    show_label=False,
                )

                with gr.Row(elem_classes=["input-row"]):
                    msg_box = gr.Textbox(
                        placeholder="Type a message…",
                        show_label=False,
                        scale=5,
                        lines=1,
                    )
                    send_btn = gr.Button("SEND", variant="primary", scale=1)
                    clear_btn = gr.Button("CLEAR", variant="secondary", scale=1)

            # ── Right: controls + metrics ──────────────────────────────────
            with gr.Column(scale=1, elem_classes=["controls-panel"]):
                model_dd = gr.Dropdown(
                    choices=[(v["name"], k) for k, v in MODELS.items()],
                    value=default_model,
                    label="MODEL",
                    interactive=True,
                )
                temp_slider = gr.Slider(
                    minimum=0.0, maximum=1.5, value=0.7, step=0.05,
                    label="TEMPERATURE",
                )
                max_tokens_slider = gr.Slider(
                    minimum=64, maximum=1024, value=512, step=64,
                    label="MAX TOKENS",
                )

                gr.HTML("<div style='height:12px'></div>")
                gr.HTML("<div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#64748b;letter-spacing:1px;margin-bottom:6px'>INFERENCE METRICS</div>")

                metrics_box = gr.HTML(
                    value="<div class='metrics-empty'>Waiting for first message…</div>"
                )

        # ── Event wiring ──────────────────────────────────────────────────────
        history_state = gr.State([])

        def on_send(message, history, model_key, temperature, max_tokens):
            if not message.strip():
                yield history, gr.update(), _metrics_html({})
                return
            gen = chat_stream(message, history, model_key, int(max_tokens), temperature)
            for updated_history, metrics_html in gen:
                yield updated_history, "", metrics_html

        def on_clear():
            return [], []

        send_btn.click(
            fn=on_send,
            inputs=[msg_box, chatbot, model_dd, temp_slider, max_tokens_slider],
            outputs=[chatbot, msg_box, metrics_box],
        )
        msg_box.submit(
            fn=on_send,
            inputs=[msg_box, chatbot, model_dd, temp_slider, max_tokens_slider],
            outputs=[chatbot, msg_box, metrics_box],
        )
        clear_btn.click(fn=on_clear, outputs=[chatbot, history_state])

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default="phi3")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_ui(args.model)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
