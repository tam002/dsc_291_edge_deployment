#!/usr/bin/env bash
# models/download_models.sh
# Downloads all quantisation levels of Llama 3.2 3B Instruct from bartowski's
# HuggingFace repo. Pre-built imatrix-calibrated GGUFs.
#
# Usage:
#   bash models/download_models.sh
#   bash models/download_models.sh --quant Q4_K_M

set -e

BASE_URL="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main"
MODELS_DIR="$HOME/models/llama3.2_3b"
mkdir -p "$MODELS_DIR"

QUANT_LEVELS=(Q3_K_L Q4_K_M Q5_K_M Q6_K Q8_0)

SINGLE=""
if [[ "$1" == "--quant" && -n "$2" ]]; then SINGLE="$2"; fi

download_quant() {
    local q="$1"
    local filename="Llama-3.2-3B-Instruct-${q}.gguf"
    local dest="$MODELS_DIR/$filename"
    if [ -f "$dest" ]; then
        echo "  [skip] $filename already exists"
    else
        echo "  [dl]   $filename"
        curl -L --progress-bar "$BASE_URL/$filename" -o "$dest"
        echo "  [ok]   $(du -sh "$dest" | cut -f1)"
    fi
}

echo "==> Llama 3.2 3B Instruct — downloading GGUFs"
echo "    Destination: $MODELS_DIR"
echo ""

if [ -n "$SINGLE" ]; then
    download_quant "$SINGLE"
else
    for q in "${QUANT_LEVELS[@]}"; do download_quant "$q"; done
fi

echo ""
echo "Model paths for benchmarks:"
for q in "${QUANT_LEVELS[@]}"; do
    f="$MODELS_DIR/Llama-3.2-3B-Instruct-${q}.gguf"
    [ -f "$f" ] && echo "  $q -> $f"
done
