# Building libpsam

Comprehensive build instructions for all platforms and language bindings.

## Table of Contents

- [C Library](#c-library)
- [JavaScript/TypeScript Bindings](#javascripttypescript-bindings)
- [Python Bindings](#python-bindings)
- [WebAssembly](#webassembly)
- [Platform-Specific Notes](#platform-specific-notes)

---

## C Library

### CLI Quickstart (Sample Corpora)

Once the CLI is built (`psam` target), you can test-drive it using the bundled corpora:

```bash
# Train a toy model
../psam build --input ../corpora/text/CatSat.txt --out catsat.psam --vocab-out catsat.vocab

# Inspect predictions and explanations
../psam predict --model catsat.psam --context "the cat sat on" --top_k 5 --pretty
../psam explain --model catsat.psam --context "the cat sat on" --candidate "the" --topN 5 --pretty

# Stream a tiny completion
../psam generate --model catsat.psam --context "the cat sat on" --count 12 --seed 42 --pretty
```

Swap in `../corpora/text/Luna.txt` or `../corpora/text/TheAnomaly.txt` for longer-form material.

### Prerequisites

- C11 compiler (gcc, clang, MSVC)
- CMake ≥ 3.15
- POSIX threads (built-in on Linux/macOS, included with MinGW on Windows)

### Quick Build

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

Outputs:
- `libpsam.so` (Linux)
- `libpsam.dylib` (macOS)
- `libpsam.dll` (Windows)

### CMake Options

```bash
# Build shared library (default)
cmake -DBUILD_SHARED_LIBS=ON ..

# Build static library
cmake -DBUILD_SHARED_LIBS=OFF ..

# Build examples
cmake -DBUILD_EXAMPLES=ON ..

# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Install prefix
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
```

### Installation

```bash
# System-wide install
sudo cmake --install .

# Custom prefix
cmake --install . --prefix ~/.local
```

Installs:
- `lib/libpsam.so` - Shared library
- `include/psam.h` - Public header
- `lib/cmake/libpsam/` - CMake package config

### Running Examples

```bash
# Build with examples
cmake -DBUILD_EXAMPLES=ON ..
cmake --build .

# Run examples
./examples/c/basic_usage
./examples/c/layer_composition
```

---

## JavaScript/TypeScript Bindings

### Prerequisites

- Node.js ≥ 18 or Bun ≥ 1.0
- Built C library (libpsam.so)

### Install from npm

```bash
npm install @foundation42/libpsam
```

### Build from Source

```bash
cd bindings/javascript

# Install dependencies
npm install

# Build TypeScript
npm run build
```

### Running Examples

```bash
cd examples/javascript

# Set library path
export LIBPSAM_PATH=../../build/libpsam.so

# Run Node.js example
node node-example.js

# Or with Bun
bun node-example.js
```

### Browser (WASM)

See [WebAssembly](#webassembly) section below.

---

## Python Bindings

### Prerequisites

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- Built C library (libpsam.so)

### Install from PyPI

```bash
pip install libpsam
```

### Build from Source

```bash
cd bindings/python

# Install in development mode
pip install -e .

# Or build wheel
python -m build
pip install dist/libpsam-*.whl
```

### Running Examples

```bash
cd examples/python

# Set library path
export LIBPSAM_PATH=../../build/libpsam.so

# Run examples
python basic_usage.py
python domain_adaptation.py
```

---

## WebAssembly

### Prerequisites

- [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html)

### Install Emscripten

```bash
# Clone emsdk
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk

# Install and activate latest
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
```

### Build WASM

```bash
cd bindings/wasm
./build.sh
```

Outputs:
- `psam.js` - JavaScript loader
- `psam.wasm` - WebAssembly binary

### Using in Browser

```html
<!DOCTYPE html>
<html>
<head>
    <script src="psam.js"></script>
</head>
<body>
    <script>
        createPSAMModule().then(Module => {
            const create = Module.cwrap('psam_create', 'number', ['number', 'number', 'number']);
            const model = create(50000, 8, 32);
            console.log('Model created:', model);
        });
    </script>
</body>
</html>
```

Or use the high-level TypeScript wrapper:

```typescript
import { PSAMWASM } from '@foundation42/libpsam/wasm';

const psam = await PSAMWASM.create(50000, 8, 32);
psam.trainBatch([1, 2, 3, 4, 5]);
const predictions = psam.predict([1, 2, 3]);
```

---

## Platform-Specific Notes

### Linux

Standard build works out of the box:

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Library: `libpsam.so`

### macOS

Standard build works:

```bash
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

Library: `libpsam.dylib`

**Note:** On Apple Silicon, ensure Rosetta is not interfering if using x86_64 tools.

### Windows

#### With Visual Studio

```bash
mkdir build && cd build
cmake -G "Visual Studio 17 2022" ..
cmake --build . --config Release
```

Library: `libpsam.dll`

#### With MinGW

```bash
mkdir build && cd build
cmake -G "MinGW Makefiles" ..
cmake --build .
```

Library: `libpsam.dll`

**Notes:**
- Ensure pthreads are available (included with MinGW-w64)
- May need to set CMAKE_C_COMPILER to gcc explicitly

---

## Build Troubleshooting

### "CMake not found"

```bash
# Ubuntu/Debian
sudo apt install cmake

# macOS
brew install cmake

# Windows
# Download from https://cmake.org/download/
```

### "pthread not found" (Windows)

Use MinGW-w64 which includes pthreads, or install winpthreads:

```bash
# MSYS2
pacman -S mingw-w64-x86_64-winpthreads
```

### "libpsam.so not found" (Runtime)

Set library path:

```bash
# Linux
export LD_LIBRARY_PATH=/path/to/build:$LD_LIBRARY_PATH

# macOS
export DYLD_LIBRARY_PATH=/path/to/build:$DYLD_LIBRARY_PATH

# Or install system-wide
sudo cmake --install build
```

For language bindings:

```bash
export LIBPSAM_PATH=/path/to/libpsam.so
```

### "Emscripten not found"

```bash
# Install emsdk
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest

# Activate in current shell
source ./emsdk_env.sh

# Or add to shell profile
echo 'source /path/to/emsdk/emsdk_env.sh' >> ~/.bashrc
```

---

## Advanced Build Options

### Static Library

```bash
cmake -DBUILD_SHARED_LIBS=OFF ..
```

Outputs: `libpsam.a` (Linux/macOS) or `libpsam.lib` (Windows)

### Cross-Compilation

Example for ARM64:

```bash
cmake -DCMAKE_SYSTEM_NAME=Linux \
      -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
      -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
      ..
```

### Custom Compiler Flags

```bash
cmake -DCMAKE_C_FLAGS="-O3 -march=native" ..
```

### Verbose Build

```bash
cmake --build . --verbose
```

---

## Testing the Build

### C Library

```bash
# Run example
./build/examples/c/basic_usage

# Check library
nm -D build/libpsam.so | grep psam_create
```

### JavaScript

```bash
node -e "import('@foundation42/libpsam').then(m => console.log('OK'))"
```

### Python

```bash
python -c "from psam import PSAM; print('OK')"
```

---

## CI/CD Integration

### GitHub Actions

```yaml
name: Build

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: sudo apt install cmake build-essential

      - name: Build
        run: |
          mkdir build && cd build
          cmake -DCMAKE_BUILD_TYPE=Release ..
          cmake --build .

      - name: Test
        run: |
          cd build
          ./examples/c/basic_usage
```

### Docker

```dockerfile
FROM gcc:latest

RUN apt-get update && apt-get install -y cmake

WORKDIR /app
COPY . .

RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build .

CMD ["./build/examples/c/basic_usage"]
```

---

## See Also

- [README](../README.md)
- [API Reference](./API.md)
- [Examples](../examples/)
