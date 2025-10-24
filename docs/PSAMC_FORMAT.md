# .psamc File Format Specification

**PSAM Composite Format** - Extensible model composition with integrity verification

## Overview

The `.psamc` format is designed for:
- **Model composition** - combine base + overlay models
- **Integrity verification** - SHA-256 checksums prevent "works on my machine" bugs
- **External references** - models can reference external files by URL
- **Hyperparameter storage** - preserve training configuration
- **Forward compatibility** - extensible section-based design

## File Structure

```
┌─────────────────────────────────────────┐
│ Header (48 bytes)                       │
│  - Magic: "PSMC" (0x434D5350)          │
│  - Version: 1                           │
│  - Flags: feature bitmask               │
│  - Section count                        │
│  - File size                            │
├─────────────────────────────────────────┤
│ Section Table                           │
│  - Array of section entries             │
│  - Each: type, flags, offset, size      │
├─────────────────────────────────────────┤
│ Section Data (variable order)           │
│  ├─ Manifest Section (optional)         │
│  ├─ Config Section                      │
│  ├─ Base Model Section                  │
│  ├─ Layer Sections (0+)                 │
│  └─ Extensions (reserved)               │
└─────────────────────────────────────────┘
```

## Header Format

```c
typedef struct {
    uint32_t magic;         // PSAMC_MAGIC (0x434D5350)
    uint32_t version;       // Format version (currently 1)
    uint32_t flags;         // Feature flags
    uint32_t num_sections;  // Number of sections
    uint64_t file_size;     // Total file size
    uint8_t reserved[16];   // Reserved for future use
} psamc_header_t;
```

### Feature Flags

| Flag | Value | Description |
|------|-------|-------------|
| `PSAMC_FLAG_COMPRESSED` | `1 << 0` | Model data is compressed |
| `PSAMC_FLAG_ENCRYPTED` | `1 << 1` | Model data is encrypted (reserved) |
| `PSAMC_FLAG_QUANTIZED` | `1 << 2` | Weights are quantized (reserved) |
| `PSAMC_FLAG_HAS_MANIFEST` | `1 << 3` | Has integrity manifest |

## Section Types

| Type | ID | Description |
|------|----|----|
| `MANIFEST` | 1 | Integrity manifest with SHA-256 checksums |
| `CONFIG` | 2 | Hyperparameters and presets |
| `BASE_MODEL` | 3 | Base model data (.psam format) |
| `LAYER` | 4 | Overlay layer (repeatable) |
| `METADATA` | 5 | Human-readable metadata |
| `EXTENSIONS` | 99 | Reserved for future features |

## Manifest Section

Ensures integrity of external model references and provides provenance tracking:

```c
typedef struct {
    char url[256];          // URL or path to external model
    uint8_t sha256[32];     // Expected SHA-256 hash
    uint64_t size;          // Expected size in bytes
    semver_t version;       // Semantic version (major.minor.patch)
    char model_id[64];      // Unique identifier
} psamc_model_ref_t;

typedef struct {
    uint32_t num_references;    // Number of external model references
    psamc_model_ref_t* refs;    // Array of references
    sha256_hash_t self_hash;    // Hash of this composite (excluding this field)
    sha256_hash_t source_hash;  // Hash of training data source
    uint64_t created_timestamp; // Unix timestamp
    char created_by[128];       // Creator/tool identification
} psamc_manifest_t;
```

**Integrity Checking:**
- Computes SHA-256 of referenced file
- Compares against manifest
- Refuses to load on mismatch
- Prevents "works on my machine" bugs

**Provenance Tracking:**
- `created_by`: Tracks which tool/user created the model (e.g., "psam-trainer v1.2.3", "john@example.com")
- `created_timestamp`: Unix timestamp for creation time
- `source_hash`: SHA-256 of training data for exact reproducibility
- `self_hash`: Hash of the composite file itself (excluding this field)

**Example manifest entry:**
```
URL: https://models.example.com/medical-psam-v2.1.0.psam
SHA-256: a3f5... (32 bytes)
Size: 1048576 bytes
Version: 2.1.0
Model ID: medical-specialty-2024
Created By: psam-trainer v1.2.3 (john@example.com)
Created: 2025-10-24 14:30:00 UTC
Source Hash: b4e6c3d2... (training corpus SHA-256)
```

## Config Section

Stores hyperparameters:

```c
typedef struct {
    psamc_preset_t preset;  // fast/balanced/accurate/tiny

    float alpha;            // Distance decay (default: 0.1)
    float min_evidence;     // Min edge count (default: 1.0)
    uint32_t top_k;         // Top-K predictions (default: 32)

    bool enable_ppmi;       // PPMI transform (default: true)
    bool enable_idf;        // IDF weighting (default: true)
    float idf_smoothing;    // IDF smoothing (default: 1.0)

    float edge_dropout;     // Dropout rate 0-1 (default: 0.0)
} psamc_hyperparams_t;
```

### Presets

| Preset | Alpha | Min Ev. | Top-K | IDF | Dropout | Use Case |
|--------|-------|---------|-------|-----|---------|----------|
| **FAST** | 0.15 | 2.0 | 16 | ✗ | 0.1 | Real-time inference |
| **BALANCED** | 0.1 | 1.0 | 32 | ✓ | 0.0 | Default (recommended) |
| **ACCURATE** | 0.05 | 0.5 | 64 | ✓ | 0.0 | High-quality predictions |
| **TINY** | 0.2 | 3.0 | 8 | ✗ | 0.2 | Minimal memory |

## Base Model Section

Contains embedded `.psam` file or reference to external model:

```c
typedef struct {
    bool is_embedded;       // true = embedded, false = reference
    union {
        uint8_t* data;      // Embedded .psam data
        psamc_model_ref_t ref;  // External reference
    };
} psamc_base_model_t;
```

## Layer Section

Overlay layers for domain adaptation:

```c
typedef struct {
    char layer_id[64];      // Unique identifier (e.g., "medical")
    float weight;           // Blending weight (e.g., 1.5)
    bool is_embedded;       // Embedded or reference
    union {
        psamc_model_ref_t ref;
        uint64_t data_offset;
    };
} psamc_layer_meta_t;
```

## Usage Examples

### Creating a Composite

```c
#include <psam_composite.h>
#include <time.h>

// Set up hyperparameters (stored in header for exact replay)
psamc_hyperparams_t config = PSAMC_PRESET_BALANCED_CONFIG;
config.alpha = 0.12;  // Custom tweak

// Create manifest with external reference
psamc_model_ref_t medical_ref = {
    .url = "https://models.example.com/medical.psam",
    .size = 1048576,
    .version = {.major = 2, .minor = 1, .patch = 0},
    .model_id = "medical-specialty-2024"
};

// Compute SHA-256 of external model
psamc_sha256_file(medical_ref.url, &medical_ref.sha256);

// Create manifest with provenance tracking
psamc_manifest_t manifest = {
    .num_references = 1,
    .refs = &medical_ref,
    .created_timestamp = (uint64_t)time(NULL),
    .created_by = "psam-trainer v1.0.0 (john@example.com)"
};

// Hash the training data source for reproducibility
psamc_sha256_file("training_corpus.txt", &manifest.source_hash);

// Save composite (hyperparameters stored in header for exact replay)
psamc_save("my_composite.psamc", base_model, &config, &manifest);
```

### Loading with Verification

```c
// Load with integrity checking
void* model = psamc_load("my_composite.psamc", true);  // verify=true

if (!model) {
    // Verification failed - SHA-256 mismatch or size mismatch
    fprintf(stderr, "Integrity check failed!\n");
}
```

## Integrity Guarantees

When `verify_integrity=true`:

1. ✅ **SHA-256 verification** - Detects corrupted files
2. ✅ **Size verification** - Catches truncated downloads
3. ✅ **Semver checking** - Ensures compatible versions
4. ✅ **Deterministic builds** - Same inputs → same outputs
5. ✅ **Exact replay** - Hyperparameters (α, K, IDF, PPMI) stored in header
6. ✅ **Provenance tracking** - Source hash + creator + timestamp for reproducibility

**Error messages:**
```
ERROR: SHA-256 mismatch for medical.psam
  This prevents 'works on my machine' bugs.
  Expected: a3f5b2c1...
  Actual:   b4e6c3d2...
```

## Extension Points

### Future Features (Reserved)

- **Compression** - zstd/lz4 for smaller files
- **Encryption** - AES for sensitive models
- **Quantization metadata** - int8/int4 weight info
- **POS-specific Top-K** - Different K for function/content/rare words
- **PPMI smoothing** - Laplace smoothing constant
- **Custom sections** - Application-specific data

### Adding New Sections

1. Define new `psamc_section_type_t` enum value
2. Document section format
3. Increment minor version if backward-compatible
4. Increment major version if breaking change

Old readers will skip unknown sections gracefully.

## Best Practices

1. **Always verify integrity** in production
2. **Use semantic versioning** for model releases
3. **Store models on CDN** with SHA-256 in manifest
4. **Version hyperparameters** with the model for exact replay
5. **Test with different presets** (fast/balanced/accurate)
6. **Document custom configurations** in metadata section
7. **Track provenance** - always set `created_by`, `created_timestamp`, and `source_hash`
8. **Hash training data** to enable exact reproducibility of model behavior

## File Extension

- `.psam` - Single model (Position-Specific Association Memory)
- `.psamc` - Composite model (PSAM Composite)

## Compatibility

- **C11** or later
- **No external dependencies** (SHA-256 is built-in)
- **Platform independent** (little-endian format with byte swapping)
- **Forward compatible** (unknown sections are skipped)

## Security Considerations

- SHA-256 provides cryptographic integrity
- NOT designed for authentication (no signatures)
- For tamper-proof models, add HMAC or Ed25519 signatures in Extensions section
- Encryption support reserved for future (PSAMC_FLAG_ENCRYPTED)

---

**Version:** 1.0
**Status:** Draft Specification
**Last Updated:** 2025-10-24
