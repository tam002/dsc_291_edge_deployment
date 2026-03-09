import streamlit as st
import time
import subprocess
import os
import requests
import sseclient
import json
import psutil

MODEL_DIR = os.path.expanduser("~/models/llama3.2_3b")
PREFIX = "Llama-3.2-3B-Instruct"

MODEL_OPTIONS = {
    "Q2_K": "Llama-3.2-3B-Instruct-Q2_K.gguf",
    "Q4_K_M": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "Q6_K": "Llama-3.2-3B-Instruct-Q6_K.gguf",
    "Q8_0": "Llama-3.2-3B-Instruct-Q8_0.gguf",
}

LLAMA_SERVER = "/Users/jennifer/llama.cpp/build/bin/llama-server"

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

    memory_metric.metric("Peak Memory", "0 MB")
    power_metric.metric("Power", "0 W")
    efficiency_metric.metric("Tokens/Joule", "0")

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

                        prefill_metric.metric("Prefill Speed", f"{prefill_speed:.1f} t/s")
                        gen_metric.metric("Generation Speed", f"{gen_speed:.1f} t/s")
                        tok_metric.metric("Tokens Generated", token_count)
                        memory_metric.metric("Peak Memory", f"{memory_mb:.1f} MB")
                        power_metric.metric("Power", f"{avg_watts:.1f} W")
                        efficiency_metric.metric("Tokens/Joule", f"{tokens_per_joule:.2f}")

            except Exception:
                pass


# st.set_page_config(
#     page_title="Edge LLM Performance Dashboard",
#     layout="wide"
# )

# st.title("Edge LLM Performance Dashboard")

# st.caption("Quantization sweep analysis for on-device LLM inference")

# DATA_DIR = Path("results/raw")


# # ----------------------------
# # Load latest benchmark CSV
# # ----------------------------

# def load_latest_results():
#     files = sorted(DATA_DIR.glob("sweep_*.csv"))
#     if not files:
#         return None

#     latest = files[-1]
#     return pd.read_csv(latest), latest


# data = load_latest_results()

# if data is None:
#     st.warning("No benchmark results found. Run run_sweep.py first.")
#     st.stop()

# df, file_path = data

# st.success(f"Loaded results from: {file_path.name}")


# # ----------------------------
# # Sidebar filters
# # ----------------------------

# st.sidebar.header("Filters")

# quants = st.sidebar.multiselect(
#     "Quantization levels",
#     df["quant"].unique(),
#     default=df["quant"].unique()
# )

# filtered = df[df["quant"].isin(quants)]


# # ----------------------------
# # Summary metrics
# # ----------------------------

# st.header("Key Metrics")

# best_tps = filtered["tg_tps"].max()
# best_eff = filtered["tokens_per_joule"].max()
# lowest_ram = filtered["peak_ram_mib"].min()
# best_acc = filtered["gsm8k_acc"].max()

# col1, col2, col3, col4 = st.columns(4)

# col1.metric("Best Generation Speed", f"{best_tps:.1f} tok/s")
# col2.metric("Best Efficiency", f"{best_eff:.2f} tok/J")
# col3.metric("Lowest RAM", f"{lowest_ram/1024:.2f} GB")
# col4.metric("Best GSM8K Accuracy", f"{best_acc:.1f} %")


# # ----------------------------
# # Throughput chart
# # ----------------------------

# st.header("Throughput")

# throughput_df = filtered.set_index("quant")[["pp_tps", "tg_tps"]]

# st.line_chart(throughput_df)


# # ----------------------------
# # Efficiency vs Accuracy
# # ----------------------------

# st.header("Efficiency vs Accuracy")

# scatter_df = filtered[[
#     "quant",
#     "tokens_per_joule",
#     "gsm8k_acc"
# ]]

# st.scatter_chart(
#     scatter_df,
#     x="tokens_per_joule",
#     y="gsm8k_acc"
# )


# # ----------------------------
# # RAM usage chart
# # ----------------------------

# st.header("Memory Usage")

# ram_df = filtered.set_index("quant")[["peak_ram_mib"]]
# ram_df["peak_ram_gb"] = ram_df["peak_ram_mib"] / 1024

# st.bar_chart(ram_df["peak_ram_gb"])


# # ----------------------------
# # Power usage
# # ----------------------------

# st.header("Power Consumption")

# power_df = filtered.set_index("quant")[["avg_watts"]]

# st.bar_chart(power_df)


# # ----------------------------
# # Raw data table
# # ----------------------------

# st.header("Benchmark Data")

# st.dataframe(filtered, width='stretch')


# # ----------------------------
# # Optional live metrics demo
# # ----------------------------

# st.header("Live Generation Metrics (Demo)")

# run_live = st.button("Simulate Live Generation")

# if run_live:

#     placeholder = st.empty()

#     tokens = 0
#     start = time.time()

#     chart_data = []

#     for i in range(50):

#         time.sleep(0.1)

#         tokens += 1
#         elapsed = time.time() - start

#         tps = tokens / elapsed

#         chart_data.append(tps)

#         placeholder.line_chart(chart_data)

#         st.write(f"Tokens/sec: {tps:.2f}")