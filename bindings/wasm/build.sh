#!/bin/bash
# Build libpsam as WebAssembly module using Emscripten

set -e

echo "🔨 Building libpsam WASM module..."

# Check for emscripten
if ! command -v emcc &> /dev/null; then
    echo "❌ Error: Emscripten (emcc) not found"
    echo "Install from: https://emscripten.org/docs/getting_started/downloads.html"
    exit 1
fi

# Paths
CORE_DIR="../../core"
SRC_DIR="$CORE_DIR/src"
INCLUDE_DIR="$CORE_DIR/include"
BUILD_DIR="./build"
OUTPUT_DIR="$BUILD_DIR"

# Create build directory
mkdir -p "$BUILD_DIR"

# Source files
SOURCES=(
    "$SRC_DIR/core/model.c"
    "$SRC_DIR/core/csr.c"
    "$SRC_DIR/core/train.c"
    "$SRC_DIR/core/infer.c"
    "$SRC_DIR/composition/layers.c"
    "$SRC_DIR/io/serialize.c"
)

# Compiler flags
CFLAGS=(
    -O3
    -std=c11
    -I"$INCLUDE_DIR"
    -I"$SRC_DIR"
    -s WASM=1
    -s EXPORTED_RUNTIME_METHODS='["cwrap","ccall"]'
    -s ALLOW_MEMORY_GROWTH=1
    -s MODULARIZE=1
    -s 'EXPORT_NAME="createPSAMModule"'
    -s EXPORTED_FUNCTIONS='[
        "_psam_create",
        "_psam_create_with_config",
        "_psam_destroy",
        "_psam_train_token",
        "_psam_train_batch",
        "_psam_finalize_training",
        "_psam_predict",
        "_psam_explain",
        "_psam_add_layer",
        "_psam_remove_layer",
        "_psam_update_layer_weight",
        "_psam_save",
        "_psam_load",
        "_psam_get_stats",
        "_psam_error_string",
        "_psam_version",
        "_malloc",
        "_free"
    ]'
)

# Build
echo "Compiling C sources..."
emcc "${SOURCES[@]}" "${CFLAGS[@]}" -o "$OUTPUT_DIR/psam.js"

echo "✅ WASM build complete!"
echo ""
echo "Output files:"
echo "  - $BUILD_DIR/psam.js"
echo "  - $BUILD_DIR/psam.wasm"
echo ""
echo "Include in HTML:"
echo '  <script src="psam.js"></script>'
