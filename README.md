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

// Predict with temperature control
uint32_t context[] = {1, 2, 3};
psam_sampler_t sampler = {
    .transform = PSAM_LOGIT_ZSCORE,
    .temperature = 0.8f,
    .top_p = 0.95f,
    .seed = 42
};
psam_prediction_t predictions[10];
int n = psam_predict_with_sampler(model, context, 3, &sampler, predictions, 10);

// Save/load
psam_save(model, "model.psam");
psam_model_t* loaded = psam_load("model.psam");
if (!loaded) {
    fprintf(stderr, "Failed to load model\n");
    return 1;
}
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
from psam import PSAM  # Note: import module is 'psam', package name is 'libpsam'

psam = PSAM(vocab_size=50000, window=8, top_k=32)
psam.train_batch([1, 2, 3, 4, 5])
psam.finalize_training()

token_ids, scores = psam.predict([1, 2, 3], max_predictions=10)
psam.save('model.psam')
```

### Command Line Interface

A lightweight CLI is built alongside the library (executable name `psam`). It wraps the C API so you can train, inspect, and query models without writing code.

```
psam build   --input data.txt --out model.psam --vocab-out vocab.tsv
psam predict --model model.psam --prompt "the cat" --vocab vocab.tsv --top_k 5 --pretty
psam explain --model model.psam --ctx-ids 10,77,21 --candidate-id 42 --topN 8
psam generate --model model.psam --prompt "the cat" --vocab vocab.tsv --count 20 --temperature 1.0 --top_p 0.95 --seed 42
psam analyze --model model.psam
psam compose --out composite.psamc --layer base.psam --layer domain.psam
psam inspect --model composite.psamc
psam tokenize --vocab vocab.tsv --context "she sells sea shells"
psam ids      --vocab vocab.tsv --ids 12,44,77
```

Flags are consistent across commands (`--model`, `--prompt`, `--temperature`, `--top_k`, `--top_p`, `--pretty`, etc.). All commands emit JSON by default; pass `--pretty` for human-readable output. Exit codes follow `0` (ok), `2` (bad args), `3` (file missing), `4` (checksum failed), `5` (internal error).

When you build an **aligned composite** with `psam compose --from-vocabs ...`, the unified vocabulary
path is stored inside the `.psamc`, so prompt-based prediction no longer needs an extra `--vocab`:

```
psam predict --model tragedies_v1.psamc --prompt "When shall we three meet again" --top_k 10
```

**Temperature & Sampling (v1.1+):** The CLI supports intuitive temperature control (0.1-2.0 range) via z-score normalization. Use `--temperature 0.5` for deterministic output, `--temperature 1.5` for creative variation. See **[Sampler Guide](./docs/Sampler.md)** for details.

### Sample Corpora

The repository ships a few tiny corpora under `corpora/text/` so you can smoke-test the CLI end to end:

- `CatSat.txt` — the classic nursery-style sentences used by the web demo.
- `Luna.txt` — a short adventure story with richer vocabulary.
- `TheAnomaly.txt` — sci-fi vignette with longer-form structure.

You can train and query a model from these with only a couple of commands:

```bash
# 1. Train a PSAM model from the CatSat corpus
psam build \
  --input corpora/text/CatSat.txt \
  --out build/catsat.psam \
  --vocab-out build/catsat.vocab \
  --window 8 --top_k 32

# 2. Peek at the next-token prediction JSON
psam predict \
  --model build/catsat.psam \
  --context "the cat sat on" \
  --top_k 5 --pretty

# 3. Stream a short completion
psam generate \
  --model build/catsat.psam \
  --context "the cat sat on" \
  --count 12 --top_k 8 --top_p 0.95 --seed 17 --pretty

# 4. Explain why a token won
psam explain \
  --model build/catsat.psam \
  --context "the cat sat on" \
  --candidate "the" --topN 8 --pretty
```

Swap in `corpora/text/Luna.txt` or `TheAnomaly.txt` to exercise longer sequences or story-like text.

### Shakespeare Regression Harness

Want a richer sanity check? The repo ships Folger Shakespeare plays plus a harness that trains tragedy/comedy overlays, saves layered composites, and prints sample predictions so we can spot regressions quickly:

```bash
# Ensure libpsam is built and available (LIBPSAM_PATH if needed)
python scripts/shakespeare_harness.py \
  --prompt "to be or not to be" \
  --out-dir artifacts/shakespeare
```

The script caches overlay models under `artifacts/shakespeare/overlays/` and emits `.psamc` composites (tragedy, comedy, mixed) so you can inspect or reuse them with the CLI/bindings.

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

# (Installs `libpsam.so`, headers, the `psam` CLI, and `man 1 psam`.)

# Refresh linker cache on Linux if you install to /usr/local
sudo ldconfig

# Or add to LD_LIBRARY_PATH for a custom prefix:
# export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
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
- **[Sampler Guide](./docs/Sampler.md)** - Temperature control and sampling configuration
- **[.psamc Format](./docs/PSAMC_FORMAT.md)** - Composite model format with integrity verification
- **[Build Guide](./docs/BUILDING.md)** - Build instructions
- **[Core C Library](./core/README.md)** - C API reference
- **[JavaScript Bindings](./bindings/javascript/README.md)** - Node.js, Bun, Browser (WASM)
- **[Python Bindings](./bindings/python/README.md)** - Python API
- **[Examples](./examples/)** - Code examples for all languages

## Model Formats

### `.psam` - Single Model
Standard format for individual models. Compact binary format with CSR sparse matrix storage.

### `.psamc` - Composite Model
Extensible format for model composition with:
- ✅ **SHA-256 integrity checking** - Prevents "works on my machine" bugs
- ✅ **External references** - Reference models by URL with version checking
- ✅ **Hyperparameter storage** - α, K, IDF, PPMI stored for exact replay
- ✅ **Provenance tracking** - Creator, timestamp, source hash for reproducibility
- ✅ **Preset configurations** - FAST, BALANCED, ACCURATE, TINY presets
- ✅ **Layer composition** - Combine base + overlay models for domain adaptation

See **[.psamc Format Specification](./docs/PSAMC_FORMAT.md)** for full details.

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
