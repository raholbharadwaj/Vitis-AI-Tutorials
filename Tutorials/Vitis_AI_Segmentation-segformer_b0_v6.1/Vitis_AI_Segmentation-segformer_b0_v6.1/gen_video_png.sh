#!/bin/bash

# ================================
# batch inference script
# ================================

# User updating below parameters
INPUT_DIR="../datasets/cityscapes/leftImg8bit/demoVideo/stuttgart_00"
OUTPUT_DIR="demoVideo"
MODEL_PATH="segformer_b0_cityscapes_256x512.onnx"
TARGET="cpu"
CACHE_KEY="segformer_b0_cityscapes_256x512"
# User updating above parameters

# Create output directory if not exists
mkdir -p "${OUTPUT_DIR}"

echo "Input dir: ${INPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Start processing..."

count=0

for img in "${INPUT_DIR}"/*.png; do
    # Skip if no file found
    [ -e "$img" ] || continue

    # Get filename only
    filename=$(basename "$img")

    output_path="${OUTPUT_DIR}/${filename}"

    echo "Processing: $filename"

    python  demo_segformer_b0_onnx.py \
        --image_path "$img" \
        --onnx_model "$MODEL_PATH" \
        --output_path "$output_path" \
        --target "$TARGET" \
        --cache_key "$CACHE_KEY" 

    count=$((count+1))
done

echo "Done. Processed ${count} images."
