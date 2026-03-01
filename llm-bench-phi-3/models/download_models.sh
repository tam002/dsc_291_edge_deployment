#!/usr/bin/env bash
# models/download_models.sh
# Downloads all quantisation levels of Phi-3 Mini 4K Instruct from bartowski's
# HuggingFace repo. Pre-built imatrix-calibrated GGUFs.
#
# Usage:
#   bash models/download_models.sh
#   bash models/download_models.sh --quant Q4_K_M

set -e

BASE_URL="https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF/resolve/main"
MODELS_DIR="$HOME/models/phi3"
mkdir -p "$MODELS_DIR"

QUANT_LEVELS=(Q2_K Q3_K_M Q4_K_M Q5_K_M Q6_K Q8_0)

SINGLE=""
if [[ "$1" == "--quant" && -n "$2" ]]; then SINGLE="$2"; fi

download_quant() {
    local q="$1"
    local filename="Phi-3-mini-4k-instruct-${q}.gguf"
    local dest="$MODELS_DIR/$filename"
    if [ -f "$dest" ]; then
        echo "  [skip] $filename already exists"
    else
        echo "  [dl]   $filename"
        curl -L --progress-bar "$BASE_URL/$filename" -o "$dest"
        echo "  [ok]   $(du -sh "$dest" | cut -f1)"
    fi
}

echo "==> Phi-3 Mini 4K Instruct — downloading GGUFs"
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
    f="$MODELS_DIR/Phi-3-mini-4k-instruct-${q}.gguf"
    [ -f "$f" ] && echo "  $q -> $f"
done
