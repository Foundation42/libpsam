# JavaScript Examples

Examples demonstrating libpsam usage in JavaScript/TypeScript environments.

## Node.js Example

Demonstrates native bindings for maximum performance.

### Prerequisites

```bash
# Build the native library
cd ../../
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
cd ../examples/javascript

# Install dependencies
npm install @foundation42/libpsam
```

### Running

```bash
# Set library path
export LIBPSAM_PATH=../../build/libpsam.so

# Run example
node node-example.js
```

## Browser Example

Demonstrates WASM usage for browser environments.

### Prerequisites

```bash
# Build WASM module
cd ../../bindings/wasm
./build.sh
cd ../../examples/javascript
```

### Running

```bash
# Serve the HTML file
python3 -m http.server 8000

# Open browser
open http://localhost:8000/browser-example.html
```

## What's Demonstrated

- ✅ Model creation and configuration
- ✅ Training on token sequences
- ✅ Batch training for efficiency
- ✅ Making predictions
- ✅ Temperature-based sampling
- ✅ Saving and loading models
- ✅ Model statistics
- ✅ Error handling

## Performance

### Native (Node.js/Bun)
- **20-200× faster** than pure JavaScript
- Uses libpsam.so via FFI (Bun) or N-API (Node.js)
- Direct memory access, no serialization overhead

### WASM (Browser)
- **5-20× faster** than pure JavaScript
- Runs entirely client-side
- No server required after initial load
- ~20-30 KB module size (compressed)

## See Also

- [API Documentation](../../bindings/javascript/README.md)
- [Python Examples](../python/)
- [C Examples](../c/)
