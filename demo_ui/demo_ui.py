import streamlit as st
import time
import subprocess
import os
import requests
import sseclient
import json
import psutil
import pandas as pd

MODEL_DIR = os.path.expanduser("~/models/llama3.2_3b")
PREFIX = "Llama-3.2-3B-Instruct"

MODEL_OPTIONS = {
    "Q2_K": "Llama-3.2-3B-Instruct-Q2_K.gguf",
    "Q4_K_M": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "Q6_K": "Llama-3.2-3B-Instruct-Q6_K.gguf",
    "Q8_0": "Llama-3.2-3B-Instruct-Q8_0.gguf",
}

LLAMA_SERVER = "../llm-bench-llama/llama-server"

SERVER_PROCESS = None


def start_server(model_name):
    global SERVER_PROCESS

    model_file = os.path.join(MODEL_DIR, MODEL_OPTIONS[model_name])

    if SERVER_PROCESS is not None:
        SERVER_PROCESS.kill()

    cmd = [
        LLAMA_SERVER,
        "-m", model_file,
        "--port", "8080",
        "-c", "4096",
        "--threads", "8"
    ]

    SERVER_PROCESS = subprocess.Popen(cmd)

    time.sleep(5)


# -----------------------
# Streamlit state
# -----------------------

if "generated_text" not in st.session_state:
    st.session_state.generated_text = ""

if "prefill_speed" not in st.session_state:
    st.session_state.prefill_speed = 0.0

if "gen_speed" not in st.session_state:
    st.session_state.gen_speed = 0.0

if "token_count" not in st.session_state:
    st.session_state.token_count = 0

if "peak_memory" not in st.session_state:
    st.session_state.peak_memory = 0

if "avg_power" not in st.session_state:
    st.session_state.avg_power = 0

if "tokens_per_joule" not in st.session_state:
    st.session_state.tokens_per_joule = 0

if "gen_speed_history" not in st.session_state:
    st.session_state.gen_speed_history = []

if "prefill_speed_history" not in st.session_state:
    st.session_state.prefill_speed_history = []

if "token_history" not in st.session_state:
    st.session_state.token_history = []
    
if "time_history" not in st.session_state:
    st.session_state.time_history = []

if "memory_history" not in st.session_state:
    st.session_state.memory_history = []

if "efficiency_history" not in st.session_state:
    st.session_state.efficiency_history = []
    
# -----------------------
# UI
# -----------------------

st.title("Edge LLM Benchmark Dashboard")

selected_model = st.selectbox(
    "Select Quantization",
    list(MODEL_OPTIONS.keys()),
    index = 1
)

if st.button("Load Model"):
    st.write("Loading model...")
    start_server(selected_model)
    st.success(f"{selected_model} loaded!")

prompt = st.text_area("Enter prompt", "Write a short poem about AI.")
run_button = st.button("Run Prompt")

process = psutil.Process(os.getpid())

left, metric_left, metric_right = st.columns([4, 2, 2])

# -----------------------
# Output area
# -----------------------

with left:
    st.subheader("Model Output")
    output_box = st.empty()
    output_box.markdown(st.session_state.generated_text)

with metric_left:
    st.markdown("### Metrics")

    prefill_metric = st.empty()
    gen_metric = st.empty()
    tok_metric = st.empty()

    prefill_metric.metric("Prefill Speed", f"{st.session_state.prefill_speed:.1f} t/s")
    gen_metric.metric("Generation Speed", f"{st.session_state.gen_speed:.1f} t/s")
    tok_metric.metric("Tokens Generated", st.session_state.token_count)

with metric_right:
    st.markdown("### ")

    memory_metric = st.empty()
    power_metric = st.empty()
    efficiency_metric = st.empty()

    memory_metric.metric("Peak Memory", f"{st.session_state.peak_memory:.1f} MB")
    power_metric.metric("Power", f"{st.session_state.avg_power:.1f} W")
    efficiency_metric.metric("Tokens/Joule", f"{st.session_state.tokens_per_joule:.1f}")

st.markdown("### Performance Charts")

st.markdown("Prefill Speed")
prefill_chart = st.empty()

st.markdown("Generation Speed")
gen_chart = st.empty()

st.markdown("Memory Usage")
memory_chart = st.empty()

st.markdown("Efficiency")
efficiency_chart = st.empty()

# -----------------------
# Run prompt
# -----------------------

if run_button:

    start_time = time.time()
    first_token_time = None
    token_count = 0

    st.session_state.generated_text = ""

    url = "http://localhost:8080/v1/chat/completions"

    data = {
        "model": MODEL_OPTIONS[selected_model],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.7,
        "stream": True
    }

    with requests.post(url, json=data, stream=True) as response:

        response.raise_for_status()

        client = sseclient.SSEClient(response)

        for event in client.events():

            if event.data == "[DONE]":
                break

            try:

                payload = json.loads(event.data)
                delta = payload["choices"][0]["delta"]
                token = delta.get("content", "")

                if token:

                    if first_token_time is None:
                        first_token_time = time.time()

                    token_count += 1

                    st.session_state.generated_text += token
                    output_box.markdown(st.session_state.generated_text)

                    now = time.time()

                    if first_token_time:

                        prefill_time = first_token_time - start_time
                        gen_time = now - first_token_time

                        gen_speed = token_count / gen_time if gen_time > 0 else 0
                        prefill_speed = token_count / prefill_time if prefill_time > 0 else 0

                        # memory usage
                        memory_mb = process.memory_info().rss / (1024 * 1024)

                        # simple power estimate
                        avg_watts = 5.0

                        elapsed = now - start_time
                        tokens_per_joule = token_count / (avg_watts * elapsed) if elapsed > 0 else 0

                        st.session_state.prefill_speed = prefill_speed
                        st.session_state.gen_speed = gen_speed
                        st.session_state.token_count = token_count
                        st.session_state.peak_memory = memory_mb
                        st.session_state.avg_power = avg_watts
                        st.session_state.tokens_per_joule = tokens_per_joule

                        st.session_state.gen_speed_history.append(gen_speed)
                        st.session_state.prefill_speed_history.append(prefill_speed)
                        st.session_state.token_history.append(token_count)

                        st.session_state.time_history.append(elapsed)
                        st.session_state.memory_history.append(memory_mb)
                        st.session_state.efficiency_history.append(tokens_per_joule)

                        prefill_metric.metric("Prefill Speed", f"{prefill_speed:.1f} t/s")
                        gen_metric.metric("Generation Speed", f"{gen_speed:.1f} t/s")
                        tok_metric.metric("Tokens Generated", token_count)
                        memory_metric.metric("Peak Memory", f"{memory_mb:.1f} MB")
                        power_metric.metric("Power", f"{avg_watts:.1f} W")
                        efficiency_metric.metric("Tokens/Joule", f"{tokens_per_joule:.2f}")

                        prefill_df = pd.DataFrame({
                            "Time (sec)": st.session_state.time_history,
                            "Prefill Speed (tokens / sec)": st.session_state.prefill_speed_history
                        })

                        prefill_chart.line_chart(prefill_df, x = "Time (sec)", y = "Prefill Speed (tokens / sec)")

                        gen_df = pd.DataFrame({
                            "Time (sec)": st.session_state.time_history,
                            "Generation Speed (tokens / sec)": st.session_state.gen_speed_history
                        })

                        gen_chart.line_chart(gen_df, x = "Time (sec)", y = "Generation Speed (tokens / sec)")
                        
                        memory_df = pd.DataFrame({
                            "Time (sec)": st.session_state.time_history,
                            "Memory Usage (MB)": st.session_state.memory_history
                        })

                        memory_chart.line_chart(memory_df, x = "Time (sec)", y = "Memory Usage (MB)")

                        efficiency_df = pd.DataFrame({
                            "Time (sec)": st.session_state.time_history,
                            "Efficiency (tokens / joule)": st.session_state.efficiency_history
                        })

                        efficiency_chart.line_chart(efficiency_df, x = "Time (sec)", y = "Efficiency (tokens / joule)")
            except Exception:
                pass
