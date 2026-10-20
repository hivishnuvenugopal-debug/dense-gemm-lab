#!/bin/bash

set -o pipefail

EXE=./build/register_tiled_gemm_template
M=1024
K=1024
N=1024
TILE_K=32
ITERS=100

CONFIGS=(
    "1 1 32 32"
    "1 2 16 32"
    "2 2 16 16"
    "4 2 16 8"
    "4 4 8 8"
)

for config in "${CONFIGS[@]}"; do
    read -r RM RN BLOCK_X BLOCK_Y <<< "$config"

    echo "=========================================="
    echo "Register tile: ${RM}x${RN}"
    echo "Block: ${BLOCK_X}x${BLOCK_Y}"
    echo "=========================================="

    "$EXE" \
        "$M" "$K" "$N" \
        "$RM" "$RN" "$TILE_K" \
        "$BLOCK_X" "$BLOCK_Y" \
        "$ITERS" || echo "WARNING: Configuration failed"

    echo
done 2>&1 | tee results_template.txt
