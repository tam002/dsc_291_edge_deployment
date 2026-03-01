#!/usr/bin/env bash
# models/download_models.sh — Download Phi-3 Mini 4K Instruct Q4_K_M GGUF
set -e

MODELS_DIR="$HOME/models"
mkdir -p "$MODELS_DIR/phi3_q4"

DEST="$MODELS_DIR/phi3_q4/Phi-3-mini-4k-instruct-q4.gguf"
URL="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"

if [ -f "$DEST" ]; then
    echo "Model already exists: $DEST"
else
    echo "==> Downloading Phi-3 Mini 4K Instruct Q4_K_M (~2.23 GiB)..."
    curl -L --progress-bar "$URL" -o "$DEST"
    echo "   Saved to $DEST"
fi

echo ""
echo "✓ Done. Model path: $DEST"
