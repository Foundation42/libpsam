# Python Examples

Examples demonstrating libpsam usage in Python.

## Prerequisites

```bash
# Build the native library
cd ../../
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
cd ../examples/python

# Install Python package
pip install -e ../../bindings/python
```

## Running Examples

### Basic Usage

```bash
# Set library path
export LIBPSAM_PATH=../../build/libpsam.so

# Run example
python basic_usage.py
```

Output:
```
╔════════════════════════════════════════════════════════════╗
║          libpsam - Python Basic Usage Example             ║
╚════════════════════════════════════════════════════════════╝

✓ Native library loaded

📦 Creating PSAM model...
   - Vocabulary size: 100
   - Window: 8
   - Top-K: 10

📚 Training on sequence...
   Tokens: [1, 2, 3, 4, 5, 6, 1, 7, 8]

✓ Training complete!
...
```

### Domain Adaptation

```bash
python domain_adaptation.py
```

Demonstrates:
- Creating multiple domain-specific models
- Hot-swapping layers
- Adjusting layer weights
- Domain blending for predictions

## What's Demonstrated

### basic_usage.py
- ✅ Model creation
- ✅ Batch training
- ✅ Prediction generation
- ✅ Temperature-based sampling
- ✅ Model persistence (save/load)
- ✅ Statistics retrieval

### domain_adaptation.py
- ✅ Layer composition
- ✅ Domain-specific models
- ✅ Hot-swapping domains
- ✅ Weight adjustment
- ✅ Multi-domain blending

## Performance

Python bindings use ctypes to interface with the native C library:

- **20-200× faster** than pure Python
- **Zero-copy** array operations with NumPy
- **Efficient** - Direct C function calls, no subprocess overhead

## Common Patterns

### Building a Vocabulary

```python
texts = ["the quick brown fox", "the lazy dog"]
vocab = {}
tokens_list = []

for text in texts:
    for word in text.split():
        if word not in vocab:
            vocab[word] = len(vocab)
        tokens_list.append(vocab[word])

psam = PSAM(vocab_size=len(vocab), window=8, top_k=10)
psam.train_batch(tokens_list)
psam.finalize_training()
```

### Sequence Generation

```python
def generate(psam, start_tokens, max_length=20, temperature=1.0):
    sequence = list(start_tokens)
    for _ in range(max_length):
        context = sequence[-psam.window:]
        next_token = psam.sample(context, temperature)
        sequence.append(next_token)
    return sequence

generated = generate(psam, start_tokens=[1, 2], max_length=10)
```

## See Also

- [Python API Documentation](../../bindings/python/README.md)
- [JavaScript Examples](../javascript/)
- [C Examples](../c/)
