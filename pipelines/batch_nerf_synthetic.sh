#!/bin/bash
# Batch process all NeRF synthetic scenes

SCENES=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship" "textureless")
SPLIT="train"  # trainセットのみ処理

K_LIST="200"
MAX_ITERATIONS=500
TOL=1e-4
MIN_ITERATIONS=5
INIT_MODE="grid"
OUTPUT_BASE_DIR="data/fitted_gs"

for scene in "${SCENES[@]}"; do
    input_dir="data/nerf_synthetic/${scene}/${SPLIT}"
    
    # Check if directory exists
    if [ ! -d "$input_dir" ]; then
        echo "Skipping ${input_dir} (directory not found)"
        continue
    fi
    
    echo "=========================================="
    echo "Processing: ${scene}/${SPLIT}"
    echo "=========================================="
    
    python pipelines/batch_image_to_gmm_em.py \
        --input_dir "$input_dir" \
        --output_base_dir "$OUTPUT_BASE_DIR" \
        --k_list "$K_LIST" \
        --max_iterations $MAX_ITERATIONS \
        --tol $TOL \
        --min_iterations $MIN_ITERATIONS \
        --init_mode "$INIT_MODE"
    
    echo ""
done

echo "All scenes processed!"

