#!/usr/bin/env bash
# setup.sh — Install llama.cpp (Metal) + MLX + Python deps on M1 Mac
set -e

echo "==> Checking prerequisites..."
command -v cmake  >/dev/null || { echo "Install cmake: brew install cmake"; exit 1; }
command -v python3>/dev/null || { echo "Install python3"; exit 1; }
command -v brew   >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }

# ── llama.cpp with Metal backend ──────────────────────────────────────────────
LLAMA_DIR="$HOME/llama.cpp"
if [ ! -d "$LLAMA_DIR" ]; then
    echo "==> Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
else
    echo "==> Updating llama.cpp..."
    git -C "$LLAMA_DIR" pull
fi

echo "==> Building llama.cpp with Metal (Apple Silicon GPU)..."
cmake -B "$LLAMA_DIR/build" "$LLAMA_DIR" \
    -DGGML_METAL=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_DIR/build" --config Release -j "$(sysctl -n hw.logicalcpu)"

echo "==> Symlinking binaries..."
for bin in llama-bench llama-perplexity llama-cli; do
    src="$LLAMA_DIR/build/bin/$bin"
    [ -f "$src" ] && ln -sf "$src" "./$bin" && echo "   linked $bin"
done

# ── Python deps ───────────────────────────────────────────────────────────────
echo "==> Installing Python dependencies..."
pip3 install -r requirements.txt

# ── MLX ───────────────────────────────────────────────────────────────────────
echo "==> Installing MLX..."
pip3 install mlx mlx-lm

# ── Wiki corpus ───────────────────────────────────────────────────────────────
if [ ! -f "$HOME/wiki.test.raw" ]; then
    echo "==> Downloading wikitext-2..."
    curl -L "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" \
         -o /tmp/wikitext.zip
    unzip -o /tmp/wikitext.zip -d /tmp/wikitext/
    cp /tmp/wikitext/wikitext-2-raw/wiki.test.raw "$HOME/wiki.test.raw"
fi

echo ""
echo "✓ Setup complete."
echo "  Next: bash models/download_models.sh"
