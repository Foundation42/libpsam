# libpsam

**Fast, lightweight sequence prediction using learned token associations.**

A high-performance C library for PSAM (Position-Specific Association Memory) with bindings for JavaScript/TypeScript, Python, and WebAssembly. Perfect for next-token prediction, sequence generation, and domain adaptation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🚀 **20-200× faster** than pure JavaScript/Python (native C implementation)
- 💾 **Tiny models** - KB to MB, not GB
- 🌐 **Multi-language** - C, JavaScript/TypeScript, Python, WASM
- 🔥 **Hot-swappable layers** - Domain adaptation without retraining
- 📦 **Zero dependencies** - Pure C11 (just pthreads + math lib)
- ⚡ **Low latency** - 0.01-0.1ms per inference
- 🎯 **Simple API** - Train, predict, save, load

## Interactive Demo

🌐 **Try it live**: [https://foundation42.github.io/libpsam/](https://foundation42.github.io/libpsam/)

See PSAM in action with an interactive browser demo featuring:
- Real-time training visualization
- Live predictions and auto-generation
- Adjustable parameters (PPMI, IDF, temperature, etc.)
- No installation required!

## Quick Start

### C

```c
#include <psam.h>

// Create model
psam_model_t* model = psam_create(50000, 8, 32);

// Train
uint32_t tokens[] = {1, 2, 3, 4, 5};
psam_train_batch(model, tokens, 5);
psam_finalize_training(model);

// Predict
psam_prediction_t predictions[10];
int n = psam_predict(model, context, context_len, predictions, 10);

// Save/load
psam_save(model, "model.psam");
psam_model_t* loaded = psam_load("model.psam");
```

### JavaScript/TypeScript

```javascript
import { createPSAM } from '@foundation42/libpsam';

const psam = createPSAM(50000, 8, 32);
psam.trainBatch([1, 2, 3, 4, 5]);
psam.finalizeTraining();

const { ids, scores } = psam.predict([1, 2, 3], 10);
psam.save('model.psam');
```

### Python

```python
from psam import PSAM

psam = PSAM(vocab_size=50000, window=8, top_k=32)
psam.train_batch([1, 2, 3, 4, 5])
psam.finalize_training()

token_ids, scores = psam.predict([1, 2, 3], max_predictions=10)
psam.save('model.psam')
```

## Installation

### Building the C Library

```bash
# Clone repository
git clone https://github.com/Foundation42/libpsam.git
cd libpsam

# Build with CMake
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .

# Optional: Install system-wide
sudo cmake --install .
```

### Language Bindings

#### JavaScript/TypeScript

```bash
npm install @foundation42/libpsam
```

#### Python

```bash
pip install libpsam
```

Set the library path:
```bash
export LIBPSAM_PATH=/path/to/libpsam/build/libpsam.so
```

## Documentation

- **[How PSAM Works](./docs/PSAM.md)** - Theory, PPMI, IDF, architecture
- **[API Reference](./docs/API.md)** - Complete API documentation
- **[Build Guide](./docs/BUILDING.md)** - Build instructions
- **[Core C Library](./core/README.md)** - C API reference
- **[JavaScript Bindings](./bindings/javascript/README.md)** - Node.js, Bun, Browser (WASM)
- **[Python Bindings](./bindings/python/README.md)** - Python API
- **[Examples](./examples/)** - Code examples for all languages

## Examples

### Domain Adaptation

Hot-swap specialized domain layers without retraining:

```javascript
// Load base and domain models
const base = PSAMNative.load('models/general.psam');
const medical = PSAMNative.load('models/medical.psam');

// Add medical layer with 1.5× weight
base.addLayer('medical', medical, 1.5);

// Predictions now blend both models
const preds = base.predict(context, 10);

// Switch domains instantly
base.removeLayer('medical');
base.addLayer('legal', legalModel, 1.5);
```

### Sequence Generation

```python
def generate(psam, start_tokens, max_length=20):
    sequence = list(start_tokens)
    for _ in range(max_length):
        context = sequence[-psam.window:]
        next_token = psam.sample(context, temperature=1.0)
        sequence.append(next_token)
    return sequence

generated = generate(psam, [1, 2], max_length=10)
```

See [examples/](./examples/) for more.

## Performance

### Native C Library
- **Latency**: 0.01-0.1ms per inference
- **Throughput**: 10,000-100,000 inferences/sec
- **Memory**: Linear in vocabulary size + sparse edges

### JavaScript (Node.js/Bun)
- **20-200× faster** than pure JS (via FFI)
- Zero subprocess overhead
- Direct memory access

### Python
- **20-200× faster** than pure Python (via ctypes)
- Zero-copy NumPy integration
- Native C function calls

### WASM (Browser)
- **5-20× faster** than pure JS
- ~20-30 KB module (compressed)
- Runs entirely client-side

## Architecture

**Training:**
- Sliding window co-occurrence counting
- PPMI (Positive Pointwise Mutual Information) weighting
- IDF (Inverse Document Frequency) for rare tokens
- Distance decay with exponential falloff
- Top-K pruning and edge dropout

**Inference:**
- CSR (Compressed Sparse Row) matrix format
- Binary search for O(log n) row lookups
- Float to int16 quantization with per-row scaling
- Score accumulation with layer blending
- SIMD-ready implementation

**Storage:**
- Single binary file format
- Magic number and versioning
- Atomic writes, fast loading
- Compatible across languages

## Use Cases

- 🤖 **Chatbots** - Fast, lightweight language models
- 📝 **Text completion** - Code, prose, structured data
- 🔍 **Search** - Query understanding and expansion
- 🎯 **Recommendation** - Sequence-based suggestions
- 🧪 **Research** - Transformer-free language modeling
- 🎮 **Games** - NPC dialogue, procedural text

## Requirements

### Build Requirements
- C11 compiler (gcc, clang, MSVC)
- CMake ≥ 3.15
- pthreads (usually built-in)

### Optional
- Emscripten (for WASM builds)
- Node.js ≥ 18 or Bun ≥ 1.0 (for JS bindings)
- Python ≥ 3.8 (for Python bindings)

## Project Structure

```
libpsam/
├── core/                   # C library
│   ├── include/psam.h     # Public API
│   └── src/               # Implementation
├── bindings/              # Language bindings
│   ├── javascript/        # Node.js, Browser (WASM)
│   ├── python/            # Python ctypes bindings
│   └── wasm/              # Emscripten build
├── examples/              # Code examples
│   ├── c/
│   ├── javascript/
│   └── python/
└── docs/                  # Documentation
```

## Contributing

Contributions welcome! This library aims to stay simple, fast, and hackable.

**Areas of interest:**
- Novel weighting schemes beyond PPMI/IDF
- Alternative sparse matrix formats
- Hybrid architectures (PSAM + transformers)
- Additional language bindings (Rust, Go, etc.)
- Larger-scale experiments

## Authors

**Christian Beaumont** - Creator and Lead Developer

Built in collaboration with AI assistants:
- Claude Chat (Anthropic)
- GPT-5 (OpenAI)
- Codex (OpenAI)
- Claude Code (Anthropic)

## License

MIT License - see [LICENSE](./LICENSE) file.

## Links

- **GitHub**: https://github.com/Foundation42/libpsam
- **Issues**: https://github.com/Foundation42/libpsam/issues
- **npm**: https://www.npmjs.com/package/@foundation42/libpsam
- **PyPI**: https://pypi.org/project/libpsam

## Citation

If you use libpsam in your research, please cite:

```bibtex
@software{libpsam2025,
  title = {libpsam: Fast, lightweight sequence prediction using learned token associations},
  author = {Beaumont, Christian},
  year = {2025},
  url = {https://github.com/Foundation42/libpsam}
}
```

---

**Built with ❤️ by Christian Beaumont**
