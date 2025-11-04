# libpsam - Core C Library

**Fast, lightweight sequence prediction using learned token associations.**

This is the core C implementation of PSAM (Position-Specific Association Memory), designed to be embedded in applications across multiple languages.

## Features

- **Pure C11** - No external dependencies (just pthreads and math lib)
- **Thread-safe** - Read-write locks for concurrent inference
- **Efficient** - CSR sparse matrix format with SIMD-ready kernels
- **Portable** - Cross-platform (Linux, macOS, Windows)
- **Composable** - Layer system for domain adaptation

## Building

### Using CMake (Recommended)

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
cmake --build .

# Install (optional)
sudo cmake --install .
```

### Build Options

```bash
# Build shared library (default)
cmake -DBUILD_SHARED_LIBS=ON ..

# Build static library
cmake -DBUILD_SHARED_LIBS=OFF ..

# Include examples
cmake -DBUILD_EXAMPLES=ON ..

# Release build with optimizations
cmake -DCMAKE_BUILD_TYPE=Release ..
```

## API Overview

### Lifecycle

```c
#include <psam.h>

// Create model
psam_model_t* model = psam_create(50000, 8, 32);  // vocab_size, window, top_k

// Destroy when done
psam_destroy(model);
```

### Training

```c
// Train on token sequence
uint32_t tokens[] = {1, 2, 3, 4, 5};
psam_train_batch(model, tokens, 5);

// Finalize training (builds CSR matrix)
psam_finalize_training(model);
```

### Inference

```c
// Predict next tokens
uint32_t context[] = {1, 2, 3};
psam_prediction_t predictions[10];

int num_predictions = psam_predict(model, context, 3, predictions, 10);

for (int i = 0; i < num_predictions; i++) {
    printf("Token %u: score=%.3f raw=%.3f supports=%u\n",
           predictions[i].token,
           predictions[i].score,
           predictions[i].raw_strength,
           predictions[i].support_count);
}
```

### Layered Composites

```c
psam_model_t* base = psam_load("models/general.psam");
psam_model_t* medical = psam_load("models/medical.psam");
psam_model_t* legal = psam_load("models/legal.psam");

psam_composite_t* layered = psam_create_layered(base);
psam_composite_add_layer(layered, "medical", medical, 1.5f);

psam_prediction_t blended[5];
psam_composite_predict(layered, context, 3, blended, 5);

psam_composite_remove_layer(layered, "medical");
psam_composite_add_layer(layered, "legal", legal, 1.2f);

psam_composite_destroy(layered);
```

### Persistence

```c
// Save model to disk
psam_save(model, "my-model.psam");

// Load model from disk
psam_model_t* loaded = psam_load("my-model.psam");
```

## File Format

Models are saved as single binary files with this structure:

- Magic number and version
- Configuration (vocab size, window, top-k)
- CSR sparse matrix (row offsets, targets, quantized weights, scales)
- Bias vector
- IDF weights
- Unigram counts

See `src/io/serialize.c` for format details.

## Performance

- **Latency**: 0.01-0.1ms per inference
- **Throughput**: 10,000-100,000 inferences/sec (sequential)
- **Memory**: Linear in vocabulary size + sparse edges

## Thread Safety

- Multiple threads can call `psam_predict()` simultaneously (read lock)
- Composite builders maintain their own synchronization while calling into base/layer models
- Training and finalization must be called from a single thread

## Error Handling

Most functions return `psam_error_t`:

```c
psam_error_t err = psam_save(model, "model.psam");
if (err != PSAM_OK) {
    fprintf(stderr, "Error: %s\n", psam_error_string(err));
}
```

Inference functions return negative error codes or positive result count.

## License

MIT License - see LICENSE file in repository root.
