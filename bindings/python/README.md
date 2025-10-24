# libpsam - Python Bindings

**Fast, lightweight sequence prediction using learned token associations.**

Python bindings for [libpsam](https://github.com/Foundation42/libpsam) - a high-performance PSAM (Position-Specific Association Memory) library.

## Installation

```bash
pip install libpsam
```

You'll also need to build the native C library:

```bash
# Clone the repository
git clone https://github.com/Foundation42/libpsam.git
cd libpsam

# Build the C library
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .

# Set library path (or install system-wide)
export LIBPSAM_PATH=/path/to/libpsam/build/libpsam.so
```

## Quick Start

```python
from psam import PSAM

# Create a model
psam = PSAM(vocab_size=50000, window=8, top_k=32)

# Train on token sequences
tokens = [1, 2, 3, 4, 5, 2, 3, 6]
psam.train_batch(tokens)
psam.finalize_training()

# Make predictions
context = [1, 2, 3]
token_ids, scores = psam.predict(context, max_predictions=10)

print("Top predictions:", token_ids)
print("Scores:", scores)

# Sample from distribution
sampled = psam.sample(context, temperature=1.0)
print("Sampled token:", sampled)

# Save and load
psam.save("my-model.psam")
loaded = PSAM.load("my-model.psam")
```

## API Reference

### Creating Models

```python
from psam import PSAM

# Create new model
psam = PSAM(
    vocab_size=50000,  # Maximum vocabulary size
    window=8,          # Context window
    top_k=32           # Number of top predictions
)

# Load from file
psam = PSAM.load("model.psam")
```

### Training

```python
# Train on batch of tokens
psam.train_batch([1, 2, 3, 4, 5])

# Train token-by-token
psam.train_token(1)
psam.train_token(2)

# Finalize training (required before inference)
psam.finalize_training()
```

### Inference

```python
# Get top-K predictions
token_ids, scores = psam.predict([1, 2, 3], max_predictions=10)
# Returns: (List[int], np.ndarray)

# Sample from distribution
token = psam.sample([1, 2, 3], temperature=1.0)
# Returns: int
```

### Layer Composition (Domain Adaptation)

```python
# Load base and domain models
base = PSAM.load("models/general.psam")
medical = PSAM.load("models/medical.psam")

# Add medical layer with 1.5× weight
base.add_layer("medical", medical, weight=1.5)

# Predictions now blend both models
token_ids, scores = base.predict(context)

# Switch domains
base.remove_layer("medical")
base.add_layer("legal", legal_model, weight=1.5)

# Update layer weight
base.update_layer_weight("legal", new_weight=2.0)
```

### Persistence

```python
# Save to file
psam.save("my-model.psam")

# Load from file
loaded = PSAM.load("my-model.psam")
```

### Model Statistics

```python
from psam import ModelStats

stats = psam.stats()
print(f"Vocabulary: {stats.vocab_size}")
print(f"Rows: {stats.row_count}")
print(f"Edges: {stats.edge_count}")
print(f"Memory: {stats.memory_bytes / 1024:.1f} KB")
```

## Advanced Usage

### Building Vocabulary

```python
from collections import Counter

# Build vocabulary from text
texts = ["the quick brown fox", "the lazy dog"]
tokens_list = []
vocab = {}

for text in texts:
    for word in text.split():
        if word not in vocab:
            vocab[word] = len(vocab)
        tokens_list.append(vocab[word])

# Train model
psam = PSAM(vocab_size=len(vocab), window=8, top_k=10)
psam.train_batch(tokens_list)
psam.finalize_training()

# Predict
context_words = ["the", "quick"]
context_ids = [vocab[w] for w in context_words]
token_ids, scores = psam.predict(context_ids, max_predictions=5)

# Convert back to words
inv_vocab = {v: k for k, v in vocab.items()}
predicted_words = [inv_vocab[tid] for tid in token_ids]
print("Predictions:", predicted_words)
```

### Generate Sequences

```python
def generate_sequence(psam, start_tokens, max_length=20, temperature=1.0):
    """Generate a sequence of tokens"""
    sequence = list(start_tokens)

    for _ in range(max_length):
        context = sequence[-psam.window:]
        next_token = psam.sample(context, temperature=temperature)
        sequence.append(next_token)

        # Stop on end-of-sequence token (if applicable)
        # if next_token == EOS_TOKEN:
        #     break

    return sequence

# Usage
generated = generate_sequence(psam, start_tokens=[1, 2], max_length=10)
print("Generated:", generated)
```

## Performance

- **20-200× faster** than pure Python implementations
- **Low memory** - Linear in vocabulary size + sparse edges
- **Efficient** - Uses optimized C library with SIMD-ready kernels

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- libpsam.so (built from C source)

## Error Handling

```python
from psam import PSAM, PSAMError

try:
    psam = PSAM(vocab_size=50000, window=8, top_k=32)
    psam.train_batch(tokens)
    psam.finalize_training()
except PSAMError as e:
    print(f"PSAM error: {e}")
```

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black psam/

# Type checking
mypy psam/
```

## License

MIT - see [LICENSE](../../LICENSE)

## Links

- [GitHub Repository](https://github.com/Foundation42/libpsam)
- [API Documentation](../../docs/API.md)
- [Examples](../../examples/python/)
