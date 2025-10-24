# @foundation42/libpsam

**Fast, lightweight sequence prediction using learned token associations.**

JavaScript/TypeScript bindings for [libpsam](https://github.com/Foundation42/libpsam) - a high-performance PSAM (Position-Specific Association Memory) library.

## Features

- 🚀 **20-200× faster** than pure JavaScript (native C library)
- 🌐 **Browser support** via WebAssembly
- 📦 **Tiny models** - KB to MB, not GB
- 🔥 **Hot-swappable layers** - Domain adaptation without retraining
- 💪 **TypeScript** - Full type safety

## Installation

```bash
npm install @foundation42/libpsam
```

For native performance, you'll also need to build the C library:

```bash
# Clone the repository
git clone https://github.com/Foundation42/libpsam.git
cd libpsam

# Build the native library
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .

# Set library path (or copy libpsam.so to your project)
export LIBPSAM_PATH=/path/to/libpsam/build/libpsam.so
```

## Quick Start

```typescript
import { createPSAM } from '@foundation42/libpsam';

// Create a model (auto-selects best implementation)
const psam = createPSAM(
  50000,  // vocab size
  8,      // context window
  32      // top-K predictions
);

// Train on token sequences
const tokens = [1, 2, 3, 4, 5, 2, 3, 6];
psam.trainBatch(tokens);
psam.finalizeTraining();

// Make predictions
const context = [1, 2, 3];
const predictions = psam.predict(context, 10);

console.log('Top predictions:', predictions.ids);
console.log('Scores:', predictions.scores);

// Sample from distribution
const sampled = psam.sample(context, 1.0);  // temperature = 1.0

// Save and load
psam.save('my-model.psam');
const loaded = PSAMNative.load('my-model.psam');
```

## API

### Creating Models

```typescript
import { createPSAM, PSAMNative } from '@foundation42/libpsam';

// Auto-select implementation (native > WASM)
const psam1 = createPSAM(vocabSize, window, topK);

// Force native implementation
const psam2 = new PSAMNative(vocabSize, window, topK);

// Force WASM implementation (browser)
import { PSAMWASM } from '@foundation42/libpsam/wasm';
const psam3 = await PSAMWASM.create(vocabSize, window, topK);
```

### Training

```typescript
// Train on batch
psam.trainBatch([1, 2, 3, 4, 5]);

// Train token-by-token
psam.trainToken(1);
psam.trainToken(2);

// Finalize (required before inference)
psam.finalizeTraining();
```

### Inference

```typescript
// Get top-K predictions
const result = psam.predict([1, 2, 3], 10);
// result.ids: TokenId[]
// result.scores: Float32Array

// Sample from distribution
const token = psam.sample([1, 2, 3], 1.0);  // temperature
```

### Layer Composition (Domain Adaptation)

```typescript
// Load base and domain-specific models
const base = PSAMNative.load('models/general.psam');
const medical = PSAMNative.load('models/medical.psam');

// Add medical layer with 1.5× weight
base.addLayer('medical', medical, 1.5);

// Predictions now blend both models
const preds = base.predict(context, 10);

// Switch domains
base.removeLayer('medical');
base.addLayer('legal', legalModel, 1.5);

// Update layer weight
base.updateLayerWeight('legal', 2.0);
```

### Persistence

```typescript
// Save to file
psam.save('my-model.psam');

// Load from file
const loaded = PSAMNative.load('my-model.psam');
```

### Statistics

```typescript
const stats = psam.stats();
console.log(stats);
// {
//   vocabSize: 50000,
//   rowCount: 1234,
//   edgeCount: 45678,
//   totalTokens: 100000,
//   memoryBytes: 524288
// }
```

## Implementation Selection

The library automatically selects the best available implementation:

```typescript
import { isNativeAvailable, isWASMAvailable } from '@foundation42/libpsam';

console.log('Native available:', isNativeAvailable());
console.log('WASM available:', isWASMAvailable());

// Force specific implementation
const psam = createPSAM(vocabSize, window, topK, 'native');  // or 'wasm', 'auto'
```

## Performance

### Native (FFI)
- **Latency**: 0.01-0.1ms per inference
- **Throughput**: 10,000-100,000 inferences/sec
- **20-200× faster** than pure JavaScript

### WASM
- **Latency**: 0.1-1ms per inference
- **Throughput**: 1,000-10,000 inferences/sec
- **5-20× faster** than pure JavaScript
- **Browser compatible**

## Requirements

### Node.js/Bun (Native)
- Node.js ≥ 18 or Bun ≥ 1.0
- libpsam.so (build from source)
- For Node.js: `ffi-napi` (optional, for native bindings)

### Browser (WASM)
- Modern browser with WebAssembly support
- psam.wasm module (build with Emscripten)

## Building from Source

See the [main repository](https://github.com/Foundation42/libpsam) for build instructions.

## License

MIT - see [LICENSE](../../LICENSE)

## Links

- [GitHub Repository](https://github.com/Foundation42/libpsam)
- [API Documentation](../../docs/API.md)
- [Examples](../../examples/javascript/)
