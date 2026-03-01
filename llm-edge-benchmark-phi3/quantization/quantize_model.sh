#!/usr/bin/env bash
# quantization/quantize_model.sh — Quantise Llama 3.2 3B Instruct from HuggingFace
set -e

LLAMA_DIR="$HOME/llama.cpp"
OUT_DIR="$HOME/models/phi3_custom"
HF_ID="microsoft/Phi-3-mini-4k-instruct"
BASE_NAME="Phi-3-mini-4k-instruct"
QUANT="${1:-Q4_K_M}"

mkdir -p "$OUT_DIR"

echo "==> Downloading $HF_ID..."
HF_LOCAL="$OUT_DIR/hf_model"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$HF_ID', local_dir='$HF_LOCAL', ignore_patterns=['*.bin'])
"

echo "==> Converting to F16 GGUF..."
F16="$OUT_DIR/${BASE_NAME}-F16.gguf"
python3 "$LLAMA_DIR/convert_hf_to_gguf.py" "$HF_LOCAL" --outtype f16 --outfile "$F16"

echo "==> Quantising to $QUANT..."
"$LLAMA_DIR/build/bin/llama-quantize" "$F16" "$OUT_DIR/${BASE_NAME}-${QUANT}.gguf" "$QUANT"

echo ""
echo "✓ Done: $OUT_DIR/${BASE_NAME}-${QUANT}.gguf"
ls -lh "$OUT_DIR"/*.gguf
