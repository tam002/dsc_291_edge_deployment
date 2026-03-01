#!/usr/bin/env bash
# setup.sh — Install llama.cpp + Python dependencies for llm-edge-benchmark
set -e

echo "==> Checking dependencies..."
command -v cmake >/dev/null 2>&1 || { echo "cmake required. Install with: sudo apt install cmake"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 required."; exit 1; }
command -v pip3 >/dev/null 2>&1 || { echo "pip3 required."; exit 1; }

# ── llama.cpp ─────────────────────────────────────────────────────────────────
LLAMA_DIR="$HOME/llama.cpp"

if [ ! -d "$LLAMA_DIR" ]; then
    echo "==> Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
else
    echo "==> llama.cpp already cloned, pulling latest..."
    git -C "$LLAMA_DIR" pull
fi

echo "==> Building llama.cpp (CPU, AVX2)..."
cmake -B "$LLAMA_DIR/build" "$LLAMA_DIR" \
    -DLLAMA_NATIVE=ON \
    -DGGML_OPENMP=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "$LLAMA_DIR/build" --config Release -j "$(nproc)"

echo "==> Symlinking binaries..."
BINS=(llama-bench llama-perplexity llama-cli llama-quantize)
for bin in "${BINS[@]}"; do
    src="$LLAMA_DIR/build/bin/$bin"
    if [ -f "$src" ]; then
        ln -sf "$src" "./$bin"
        echo "   linked: ./$bin"
    fi
done

# ── Python deps ───────────────────────────────────────────────────────────────
echo "==> Installing Python dependencies..."
pip3 install -r requirements.txt

# ── Wiki test corpus ──────────────────────────────────────────────────────────
WIKI_URL="https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
if [ ! -f "$HOME/wiki.test.raw" ]; then
    echo "==> Downloading wikitext-2 test corpus..."
    curl -L "$WIKI_URL" -o /tmp/wikitext.zip
    unzip -o /tmp/wikitext.zip -d /tmp/wikitext/
    cp /tmp/wikitext/wikitext-2-raw/wiki.test.raw "$HOME/wiki.test.raw"
    echo "   saved to $HOME/wiki.test.raw"
else
    echo "==> wiki.test.raw already present, skipping."
fi

echo ""
echo "✓ Setup complete. Run: bash models/download_models.sh"
