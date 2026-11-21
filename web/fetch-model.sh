#!/bin/sh
# Download the free quantized bge-small-en-v1.5 model files into web/models/
# so the browser demo can run locally (they are not committed to git).
set -e

BASE="https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main"
DIR="$(dirname "$0")/models/Xenova/bge-small-en-v1.5"
mkdir -p "$DIR/onnx"

for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt; do
  curl -fsSL "$BASE/$f" -o "$DIR/$f"
done
curl -fsSL "$BASE/onnx/model_quantized.onnx" -o "$DIR/onnx/model_quantized.onnx"

echo "Model files saved to $DIR"
