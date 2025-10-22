# libpsam Examples

Comprehensive examples demonstrating libpsam usage across all supported languages.

## Directory Structure

```
examples/
├── c/                    # C examples
│   ├── basic_usage.c
│   └── layer_composition.c
├── javascript/           # JavaScript/TypeScript examples
│   ├── node-example.js
│   └── browser-example.html
└── python/              # Python examples
    ├── basic_usage.py
    └── domain_adaptation.py
```

## Quick Start

### 1. Build the C Library

```bash
cd ..
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON ..
cmake --build .
```

### 2. Run Examples

#### C Examples

```bash
# From build directory
./examples/c/basic_usage
./examples/c/layer_composition
```

#### JavaScript Examples

```bash
cd ../examples/javascript

# Set library path
export LIBPSAM_PATH=../../build/libpsam.so

# Run with Node.js
node node-example.js

# Or with Bun
bun node-example.js

# Browser example
python3 -m http.server 8000
# Open http://localhost:8000/browser-example.html
```

#### Python Examples

```bash
cd ../examples/python

# Install bindings
pip install -e ../../bindings/python

# Set library path
export LIBPSAM_PATH=../../build/libpsam.so

# Run examples
python basic_usage.py
python domain_adaptation.py
```

## What Each Example Demonstrates

### C Examples

#### `basic_usage.c`

- ✅ Creating a model
- ✅ Training on token sequences
- ✅ Making predictions
- ✅ Getting model statistics
- ✅ Saving and loading models
- ✅ Error handling

**Key concepts:**
- Model lifecycle (create, use, destroy)
- Batch training for efficiency
- Top-K predictions
- Binary file persistence

#### `layer_composition.c`

- ✅ Creating multiple domain models
- ✅ Adding overlay layers
- ✅ Updating layer weights
- ✅ Removing layers
- ✅ Listing active layers

**Key concepts:**
- Domain adaptation without retraining
- Hot-swappable layers
- Weight-based blending
- Multi-domain composition

### JavaScript Examples

#### `node-example.js`

- ✅ Native FFI bindings (Node.js/Bun)
- ✅ Model creation and training
- ✅ Predictions and sampling
- ✅ Temperature-based sampling
- ✅ Model persistence

**Key concepts:**
- Auto-detection of native library
- TypeScript types
- Modern async/await patterns
- Error handling

#### `browser-example.html`

- ✅ WebAssembly in browser
- ✅ Client-side inference
- ✅ Interactive UI

**Key concepts:**
- WASM module loading
- Browser compatibility
- No server required
- Near-native performance

### Python Examples

#### `basic_usage.py`

- ✅ Creating models
- ✅ Batch training
- ✅ Predictions and sampling
- ✅ Model statistics
- ✅ Save/load

**Key concepts:**
- Pythonic API
- NumPy integration
- Exception handling
- Context managers (future)

#### `domain_adaptation.py`

- ✅ Multi-domain models
- ✅ Layer composition
- ✅ Weight adjustment
- ✅ Domain switching

**Key concepts:**
- Hot-swapping domains
- Weight-based blending
- Specialized vocabularies
- Performance optimization

## Common Patterns

### Building a Vocabulary

Most examples use raw token IDs. In practice, you'll need a vocabulary:

```python
# Build vocab from text
texts = ["the quick brown fox", "the lazy dog"]
vocab = {}
token_ids = []

for text in texts:
    for word in text.split():
        if word not in vocab:
            vocab[word] = len(vocab)
        token_ids.append(vocab[word])

# Reverse mapping
inv_vocab = {v: k for k, v in vocab.items()}

# Use with PSAM
psam = PSAM(vocab_size=len(vocab), window=8, top_k=10)
psam.train_batch(token_ids)
psam.finalize_training()

# Predict
context_words = ["the", "quick"]
context_ids = [vocab[w] for w in context_words]
pred_ids, scores = psam.predict(context_ids)

# Convert back
pred_words = [inv_vocab[tid] for tid in pred_ids]
```

### Sequence Generation

```python
def generate_sequence(psam, start_tokens, max_length=20, temperature=1.0):
    """Generate a sequence of tokens"""
    sequence = list(start_tokens)

    for _ in range(max_length):
        # Use last window tokens as context
        context = sequence[-psam.window:]

        # Sample next token
        next_token = psam.sample(context, temperature=temperature)
        sequence.append(next_token)

        # Optional: Stop on special token
        # if next_token == EOS_TOKEN:
        #     break

    return sequence

# Generate 10 tokens
generated = generate_sequence(psam, start_tokens=[1, 2], max_length=10)
```

### Domain-Specific Prediction

```python
# Load models
general = PSAM.load("models/general.psam")
medical = PSAM.load("models/medical.psam")
legal = PSAM.load("models/legal.psam")

# Function to get predictions for a domain
def predict_for_domain(base, domain_model, domain_name, context):
    # Add domain layer
    base.add_layer(domain_name, domain_model, weight=1.5)

    # Predict
    token_ids, scores = base.predict(context)

    # Remove layer
    base.remove_layer(domain_name)

    return token_ids, scores

# Use
medical_preds = predict_for_domain(general, medical, "medical", [1, 10, 11])
legal_preds = predict_for_domain(general, legal, "legal", [1, 20, 21])
```

## Performance Tips

1. **Batch Training**
   - Use `train_batch()` instead of multiple `train_token()` calls
   - 10-100× faster for large datasets

2. **Context Length**
   - Shorter contexts are faster
   - Use only as many tokens as needed
   - Max context = window size

3. **Top-K Selection**
   - Smaller top-K is faster
   - Balance accuracy vs speed
   - Typical values: 10-50

4. **Vocabulary Size**
   - Only allocate what you need
   - Larger vocab = more memory
   - Use vocab pruning for rare tokens

5. **Layer Composition**
   - Each layer adds ~10-20% overhead
   - Use 1-3 layers for best performance
   - Remove unused layers

## Benchmarking

Simple benchmark pattern:

```python
import time

# Create and train model
psam = PSAM(vocab_size=10000, window=8, top_k=32)
psam.train_batch(tokens)
psam.finalize_training()

# Warmup
for _ in range(100):
    psam.predict([1, 2, 3])

# Benchmark
context = [1, 2, 3, 4, 5]
iterations = 10000

start = time.perf_counter()
for _ in range(iterations):
    psam.predict(context)
end = time.perf_counter()

elapsed = end - start
per_inference = (elapsed / iterations) * 1000  # ms

print(f"Total: {elapsed:.3f}s")
print(f"Per inference: {per_inference:.3f}ms")
print(f"Throughput: {iterations / elapsed:.0f} inferences/sec")
```

## Next Steps

- Read the [API documentation](../docs/API.md)
- Check [building instructions](../docs/BUILDING.md)
- Explore language-specific READMEs:
  - [C Library](../core/README.md)
  - [JavaScript](../bindings/javascript/README.md)
  - [Python](../bindings/python/README.md)

## Getting Help

- **Issues**: https://github.com/Foundation42/libpsam/issues
- **Discussions**: https://github.com/Foundation42/libpsam/discussions
- **Documentation**: https://github.com/Foundation42/libpsam/tree/main/docs
