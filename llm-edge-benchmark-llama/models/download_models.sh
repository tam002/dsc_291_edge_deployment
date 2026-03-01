#!/usr/bin/env bash
# models/download_models.sh — Download Llama 3.2 3B Instruct Q4_K_M GGUF
set -e

MODELS_DIR="$HOME/models"
mkdir -p "$MODELS_DIR/llama3.2_3b"

DEST="$MODELS_DIR/llama3.2_3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
URL="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

if [ -f "$DEST" ]; then
    echo "Model already exists: $DEST"
else
    echo "==> Downloading Llama 3.2 3B Instruct Q4_K_M (~1.87 GiB)..."
    curl -L --progress-bar "$URL" -o "$DEST"
    echo "   Saved to $DEST"
fi

echo ""
echo "✓ Done. Model path: $DEST"
