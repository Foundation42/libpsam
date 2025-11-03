# libpsam API Reference

Complete API documentation for all language bindings.

## Table of Contents

- [C API](#c-api)
- [JavaScript/TypeScript API](#javascripttypescript-api)
- [Python API](#python-api)
- [Common Concepts](#common-concepts)

---

## C API

### Header

```c
#include <psam.h>
```

### Types

#### `psam_model_t`

Opaque handle to a PSAM model. Created with `psam_create()` or `psam_load()`, destroyed with `psam_destroy()`.

#### `psam_config_t`

```c
typedef struct {
    uint32_t vocab_size;      // Maximum vocabulary size
    uint32_t window;          // Context window size
    uint32_t top_k;           // Number of top predictions to keep
    float alpha;              // Laplace smoothing (default: 0.1)
    float gamma;              // Distance decay (default: 0.05)
    float eta;                // IDF smoothing (default: 1.0)
    bool enable_ppmi;         // Enable PPMI transformation (default: true)
    bool enable_idf;          // Enable IDF weighting (default: true)
} psam_config_t;
```

#### `psam_prediction_t`

```c
typedef struct {
    uint32_t token_id;        // Predicted token ID
    float score;              // Raw score
    float calibrated_prob;    // Calibrated probability (if available)
} psam_prediction_t;
```

#### `psam_explain_term_t`

```c
typedef struct {
    uint32_t source_token;    // Context token that contributed
    int16_t  rel_offset;      // Relative position delta (e.g. -3)
    float    weight_ppmi;     // Base association weight (PPMI-adjusted)
    float    idf;             // IDF weighting factor
    float    decay;           // Distance decay factor
    float    contribution;    // weight_ppmi * idf * decay
} psam_explain_term_t;
```

#### `psam_explain_result_t`

```c
typedef struct {
    uint32_t candidate;       // Token being explained
    float    total_score;     // Final sampler score (bias + contributions)
    float    bias_score;      // Baseline bias score for the candidate
    int32_t  term_count;      // Total contributing terms discovered
    float    z_logit;         // Z-score normalized logit (0.0 if unavailable)
    float    prob;            // Post-softmax probability (0.0 if unavailable)
} psam_explain_result_t;
```

#### `psam_logit_transform_t`

```c
typedef enum {
    PSAM_LOGIT_RAW = 0,        // Raw scores (max subtraction only)
    PSAM_LOGIT_ZSCORE = 1,     // Z-score normalization (default)
    PSAM_LOGIT_CALIBRATED = 2, // Use composite calibration
    PSAM_LOGIT_LEGACY = 3      // Legacy behavior (high temps 10-100)
} psam_logit_transform_t;
```

Controls how raw prediction scores are normalized before applying temperature and softmax.

- **ZSCORE** (default): Applies per-step z-score normalization `(score - mean) / std`, making temperature in the range 0.1-2.0 work intuitively across different models and contexts
- **RAW**: No normalization, just numerical stability (max subtraction)
- **LEGACY**: Preserves pre-1.1 behavior where temperatures of 10-100 were needed
- **CALIBRATED**: Reserved for future composite calibration features

#### `psam_sampler_t`

```c
typedef struct {
    psam_logit_transform_t transform; // Logit preprocessing (default: ZSCORE)
    float temperature;                // Sampling temperature (default: 1.0)
    int   top_k;                      // Top-k filtering (default: 0 = model default)
    float top_p;                      // Nucleus sampling (default: 0.95)
    uint64_t seed;                    // Random seed (default: 42)
} psam_sampler_t;
```

Configuration for inference sampling. When passed to `psam_predict_with_sampler()`, controls how scores are transformed into probabilities.

#### `psam_stats_t`

```c
typedef struct {
    uint32_t vocab_size;      // Vocabulary size
    uint32_t row_count;       // Number of CSR rows
    uint64_t edge_count;      // Total number of edges
    uint64_t total_tokens;    // Tokens processed during training
    size_t memory_bytes;      // Memory usage
} psam_stats_t;
```

#### `psam_error_t`

```c
typedef enum {
    PSAM_OK = 0,              // Success
    PSAM_NULL_PARAM = -1,     // Null parameter
    PSAM_INVALID_CONFIG = -2, // Invalid configuration
    PSAM_OUT_OF_MEMORY = -3,  // Out of memory
    PSAM_IO = -4,             // I/O error
    PSAM_INVALID_MODEL = -5,  // Invalid model file
    PSAM_NOT_TRAINED = -6,    // Model not trained
    PSAM_LAYER_NOT_FOUND = -7 // Layer not found
} psam_error_t;
```

### Functions

#### Lifecycle

##### `psam_create`

```c
psam_model_t* psam_create(uint32_t vocab_size, uint32_t window, uint32_t top_k);
```

Create a new PSAM model with default configuration.

**Parameters:**
- `vocab_size`: Maximum vocabulary size
- `window`: Context window size
- `top_k`: Number of top predictions to keep

**Returns:** Model handle, or NULL on failure

##### `psam_create_with_config`

```c
psam_model_t* psam_create_with_config(const psam_config_t* config);
```

Create a model with custom configuration.

##### `psam_destroy`

```c
void psam_destroy(psam_model_t* model);
```

Destroy model and free all resources. Safe to call with NULL.

##### `psam_get_provenance`

```c
psam_error_t psam_get_provenance(const psam_model_t* model, psam_provenance_t* out_provenance);
```

Retrieve provenance metadata (creation timestamp, creator string, dataset hash) stored with the model.

##### `psam_set_provenance`

```c
psam_error_t psam_set_provenance(psam_model_t* model, const psam_provenance_t* provenance);
```

Override provenance metadata before calling `psam_save()`.

#### Training

##### `psam_train_token`

```c
psam_error_t psam_train_token(psam_model_t* model, uint32_t token);
```

Process a single token during training.

##### `psam_train_batch`

```c
psam_error_t psam_train_batch(psam_model_t* model, const uint32_t* tokens, size_t num_tokens);
```

Process a batch of tokens (more efficient).

##### `psam_finalize_training`

```c
psam_error_t psam_finalize_training(psam_model_t* model);
```

Finalize training by computing PPMI/IDF and building CSR storage. Must be called before inference.

#### Inference

##### `psam_predict`

```c
int psam_predict(
    psam_model_t* model,
    const uint32_t* context,
    size_t context_len,
    psam_prediction_t* out_preds,
    size_t max_preds
);
```

Generate predictions for a given context.

**Returns:** Number of predictions (≥0), or negative error code

##### `psam_predict_with_sampler`

```c
int psam_predict_with_sampler(
    psam_model_t* model,
    const uint32_t* context,
    size_t context_len,
    const psam_sampler_t* sampler,
    psam_prediction_t* out_preds,
    size_t max_preds
);
```

Generate predictions with explicit sampler configuration. Applies logit normalization (z-score by default) and populates the `calibrated_prob` field with post-softmax probabilities.

**Parameters:**
- `model` - Trained model (must be finalized)
- `context` - Array of context token IDs
- `context_len` - Number of tokens in context
- `sampler` - Sampler configuration (`NULL` uses defaults: z-score, temp=1.0, top_p=0.95)
- `out_preds` - Output buffer for predictions
- `max_preds` - Maximum predictions to return

**Returns:** Number of predictions (≥0), or negative error code

**Example:**
```c
psam_sampler_t sampler = {
    .transform = PSAM_LOGIT_ZSCORE,
    .temperature = 0.7f,
    .top_k = 0,      // use model default
    .top_p = 0.9f,
    .seed = 42
};

psam_prediction_t preds[32];
int n = psam_predict_with_sampler(model, context, 3, &sampler, preds, 32);
for (int i = 0; i < n; i++) {
    printf("Token %u: score=%.2f prob=%.4f\n",
           preds[i].token, preds[i].score, preds[i].calibrated_prob);
}
```

**See also:** [Sampler Guide](Sampler.md) for details on temperature ranges and transform modes.

##### `psam_explain`

```c
psam_error_t psam_explain(
    psam_model_t* model,
    const uint32_t* context,
    size_t context_len,
    uint32_t candidate_token,
    psam_explain_term_t* out_terms,
    int max_terms,
    psam_explain_result_t* result
);
```

Explain why a specific token was predicted for the given context. Returns the top contributing association terms with full traceability.

This exposes PSAM's interpretability superpower: every prediction can be traced back to specific `(source_token, offset, weight × idf × decay)` associations.

**Parameters:**
- `model` - Trained model (must be finalized)
- `context` - Array of context token IDs
- `context_len` - Number of tokens in context
- `candidate_token` - Token ID to explain
- `out_terms` - Output buffer for explanation terms (caller-allocated, can be `NULL` if probing size)
- `max_terms` - Size of output buffer (`0` to query required capacity)
- `result` - Metadata describing the explanation

**Returns:** `PSAM_OK` on success, or negative error code

**Example:**
```c
uint32_t context[] = {10, 20, 30};
psam_explain_term_t terms[16];
psam_explain_result_t info;

psam_error_t err = psam_explain(model, context, 3, 42, terms, 16, &info);
if (err == PSAM_OK) {
    int written = info.term_count < 16 ? info.term_count : 16;
    printf("candidate=%u bias=%.4f total=%.4f terms=%d\n",
           info.candidate, info.bias_score, info.total_score, info.term_count);

    for (int i = 0; i < written; i++) {
        printf("  token=%u offset=%+d weight=%.3f idf=%.3f decay=%.3f contribution=%.4f\n",
               terms[i].source_token, terms[i].rel_offset,
               terms[i].weight_ppmi, terms[i].idf, terms[i].decay,
               terms[i].contribution);
    }

    if (info.term_count > written) {
        // Allocate a larger buffer and call again if you need the full list.
    }
} else {
    fprintf(stderr, "Explain failed: %d\n", err);
}
```

**JSON log format**

```json
{
  "candidate": 1234,
  "total": 2.7183,
  "bias": 0.4200,
  "term_count": 5,
  "terms": [
    {"source": 77, "offset": -2, "weight": 0.84, "idf": 1.09, "decay": 0.61, "contribution": 0.56}
  ]
}
```

#### Composite Builders

Layered composites allow you to blend a finalized base model with additional domain-specific overlays at runtime. Composites are opaque handles that orchestrate multiple `psam_model_t` instances without copying their weights.

##### `psam_create_layered`

```c
psam_composite_t* psam_create_layered(psam_model_t* base_model);
```

Create a layered composite around a finalized base model. Returns NULL if the base is missing or not finalized.

##### `psam_composite_add_layer`

```c
psam_error_t psam_composite_add_layer(
    psam_composite_t* composite,
    const char* layer_id,
    psam_model_t* layer_model,
    float weight
);
```

Attach a named layer with the provided blending weight. Models must share the same vocabulary size.

##### `psam_composite_remove_layer`

```c
psam_error_t psam_composite_remove_layer(psam_composite_t* composite, const char* layer_id);
```

Detach a layer by ID.

##### `psam_composite_update_layer_weight`

```c
psam_error_t psam_composite_update_layer_weight(
    psam_composite_t* composite,
    const char* layer_id,
    float new_weight
);
```

Adjust the weight of an attached layer.

##### `psam_composite_list_layers`

```c
int psam_composite_list_layers(
    const psam_composite_t* composite,
    psam_composite_layer_info_t* out_layers,
    size_t max_layers
);
```

Copy up to `max_layers` layer descriptors (ID + weight) into `out_layers`. Returns the number of copies or a negative error code.

##### `psam_composite_predict`

```c
int psam_composite_predict(
    psam_composite_t* composite,
    const uint32_t* context,
    size_t context_len,
    psam_prediction_t* out_preds,
    size_t max_preds
);
```

Blend predictions from the base model and every attached layer using their configured weights. Returns the number of predictions written.

##### `psam_composite_predict_with_sampler`

```c
int psam_composite_predict_with_sampler(
    psam_composite_t* composite,
    const uint32_t* context,
    size_t context_len,
    const psam_sampler_t* sampler,
    psam_prediction_t* out_preds,
    size_t max_preds
);
```

Blend predictions from multiple layers with per-layer weight/bias, then apply logit normalization and temperature sampling. If `sampler` is `NULL`, uses the composite's default sampler configuration (loaded from `.psamc` or initialized to zscore defaults).

**Returns:** Number of predictions (≥0), or negative error code

##### `psam_composite_update_layer_bias`

```c
psam_error_t psam_composite_update_layer_bias(
    psam_composite_t* composite,
    const char* layer_id,
    float new_bias
);
```

Update the bias offset for a specific layer in the composite. The bias is added to scores after weight scaling: `final_score = weight * score + bias`.

**Returns:** `PSAM_OK` on success, `PSAM_ERR_LAYER_NOT_FOUND` if layer ID not found

##### `psam_composite_save_file`

```c
int psam_composite_save_file(
    const char* path,
    const char* created_by,
    const psamc_hyperparams_t* hyperparams,
    float base_weight,
    const char* base_model_path,
    size_t layer_count,
    const psam_composite_layer_file_t* layers
);
```

Create a `.psamc` file that references an on-disk base model plus any number of overlay models. Each `layers[i]` entry specifies an ID, weight, and file path for the overlay. Paths are hashed at save time so integrity checks remain reproducible.

##### `psam_composite_load_file`

```c
psam_composite_t* psam_composite_load_file(const char* path, bool verify_integrity);
```

Load a `.psamc` manifest, verify referenced hashes (if requested), and instantiate a layered composite with all referenced models opened and owned by the composite handle.

> **Harness:** `scripts/shakespeare_harness.py` shows these functions in context—training tragedy/comedy overlays, emitting blended `.psamc` files with `psam_composite_save_file`, and reloading them via `psam_composite_load_file` for regression testing.

#### Vocabulary Alignment (Mixed-Vocabulary Composites)

PSAM supports composites with different vocabularies through automatic token ID remapping. This enables blending models trained on different corpora without requiring unified vocabularies during training.

```c
#include <psam_vocab_alignment.h>
```

##### Types

###### `psam_vocab_alignment_t`

Opaque handle containing:
- **Unified vocabulary**: Superset of all layer vocabularies (sorted lexicographically)
- **Per-layer remapping tables**: Bidirectional mappings between local and unified ID spaces
- **Coverage statistics**: Percentage of unified vocabulary known to each layer

###### `psam_vocab_remap_t`

Per-layer remapping structure:
- **Dense local→unified map**: O(1) array lookup, `local_vocab_size × 4 bytes`
- **Sparse unified→local map**: O(log N) binary search, `~8 bytes per local token`

###### `psam_unknown_policy_t`

```c
typedef enum {
    PSAM_UNKNOWN_SKIP = 0,      // Drop unknown tokens from layer's context
    PSAM_UNKNOWN_MAP_UNK = 1,   // Map to reserved UNK token (ID = vocab_size - 1)
    PSAM_UNKNOWN_COVERAGE = 2   // Weight layer by coverage: known_tokens / total_tokens
} psam_unknown_policy_t;
```

##### Building Alignment

###### `psam_build_vocab_alignment_from_files`

```c
psam_vocab_alignment_t* psam_build_vocab_alignment_from_files(
    const char** vocab_paths,
    size_t n_vocabs,
    const char** layer_ids,
    uint32_t* out_unified_vocab_size
);
```

Build vocabulary alignment from TSV vocabulary files.

**Parameters:**
- `vocab_paths`: Array of vocabulary file paths (TSV format: "id\ttoken\n")
- `n_vocabs`: Number of vocabulary files (typically n_layers + 1 for base)
- `layer_ids`: Optional layer identifiers for debugging (may be NULL)
- `out_unified_vocab_size`: [out] Size of unified vocabulary

**Returns:** Alignment structure, or NULL on error

**Example:**
```c
const char* vocabs[] = {"hamlet.tsv", "macbeth.tsv"};
uint32_t unified_size;
psam_vocab_alignment_t* align = psam_build_vocab_alignment_from_files(
    vocabs, 2, NULL, &unified_size);

// Result: Unified vocab contains all unique tokens from both plays
// Hamlet: 7,603 tokens → 71% coverage of 10,682 unified tokens
// Macbeth: 5,300 tokens → 50% coverage of 10,682 unified tokens
```

###### `psam_vocab_alignment_destroy`

```c
void psam_vocab_alignment_destroy(psam_vocab_alignment_t* alignment);
```

Free alignment structure and all associated memory.

##### Remapping Functions

###### `psam_vocab_remap_local_to_unified`

```c
uint32_t psam_vocab_remap_local_to_unified(
    const psam_vocab_remap_t* remap,
    uint32_t local_id
);
```

Map token ID from layer's local vocabulary to unified vocabulary space.

**Returns:** Unified token ID, or `PSAM_VOCAB_INVALID_ID` if invalid

**Performance:** O(1) array lookup

###### `psam_vocab_remap_unified_to_local`

```c
uint32_t psam_vocab_remap_unified_to_local(
    const psam_vocab_remap_t* remap,
    uint32_t unified_id
);
```

Map token ID from unified vocabulary to layer's local vocabulary space.

**Returns:** Local token ID, or `PSAM_VOCAB_INVALID_ID` if layer doesn't know this token

**Performance:** O(log N) binary search where N = tokens in layer

###### `psam_vocab_alignment_get_coverage`

```c
float psam_vocab_alignment_get_coverage(
    const psam_vocab_alignment_t* alignment,
    uint32_t layer_index
);
```

Get vocabulary coverage for a specific layer.

**Returns:** Coverage ratio (0.0 to 1.0), or -1.0 on error

**Example:**
```c
float coverage = psam_vocab_alignment_get_coverage(align, 0);
// coverage = 0.712 means layer knows 71.2% of unified vocabulary
```

##### Usage Pattern

**Step 1: Train models with individual vocabularies**
```bash
psam build --input hamlet.txt --out hamlet.psam --vocab-out hamlet.tsv
psam build --input macbeth.txt --out macbeth.psam --vocab-out macbeth.tsv
```

**Step 2: Build alignment from vocabulary files**
```c
const char* vocabs[] = {"hamlet.tsv", "macbeth.tsv"};
uint32_t unified_size;
psam_vocab_alignment_t* align = psam_build_vocab_alignment_from_files(
    vocabs, 2, NULL, &unified_size);
```

**Step 3: Create aligned composite and add overlays**
```c
psam_model_t* hamlet = psam_load("hamlet.psam");
psam_model_t* macbeth = psam_load("macbeth.psam");

psam_composite_aligned_t* comp = psam_create_composite_aligned(
    hamlet,          /* base model */
    align,           /* alignment from Step 2 */
    true,            /* composite takes ownership of alignment */
    false            /* caller keeps the base model handle */
);

psam_composite_aligned_set_unknown_policy(comp, PSAM_UNKNOWN_SKIP);
psam_composite_aligned_set_coverage_rule(comp, PSAM_COVER_LINEAR);
psam_composite_aligned_add_layer(comp, "macbeth", macbeth, 0.8f, false);
```

**Step 4: Save / load via `.psamc` (v1)**
```c
/* Extract alignment buffers if you want to call the low-level save helper */
const psam_vocab_alignment_t* alignment = psam_composite_aligned_get_alignment(comp);

psam_composite_save_v1(
    "tragedies_v1.psamc",
    "libpsam-cli",
    "unified.tsv",
    PSAM_UNKNOWN_SKIP,
    PSAM_COVER_LINEAR,
    NULL,                      /* sampler defaults (NULL = current runtime defaults) */
    alignment->num_layers,
    layer_ids, layer_model_paths,
    weights, biases,
    local_vocab_sizes,
    local_to_unified_arrays,
    unified_to_local_pairs,
    unified_to_local_counts,
    l2u_paths,
    u2l_paths,
    alignment->unified_vocab_size
);

/* Or simply let the loader reconstruct everything */
psam_composite_aligned_t* restored =
    psam_composite_load_aligned("tragedies_v1.psamc", /*verify_integrity=*/true);

psam_prediction_t preds[16];
int n = psam_composite_aligned_predict_with_sampler(
    restored,
    ctx_ids,
    ctx_len,
    NULL,   /* NULL => use sampler defaults saved in the file */
    preds,
    16
);
```

##### Design Notes

- **Deterministic IDs**: Unified vocabulary is sorted lexicographically, ensuring consistent token IDs across runs
- **Memory efficient**: ~12 bytes per token (4 bytes dense + ~8 bytes sparse)
- **Fast lookups**: O(1) for local→unified (hot path during prediction), O(log N) for unified→local
- **Coverage-aware mixing**: Future support for weighting layers by vocabulary overlap with context

#### Persistence

##### `psam_save`

```c
psam_error_t psam_save(const psam_model_t* model, const char* path);
```

Save model to binary file. Persists provenance metadata (timestamp, creator, dataset hash) along with hyperparameters so the run can be exactly replayed later.

##### `psam_load`

```c
psam_model_t* psam_load(const char* path);
```

Load model from binary file.

**Returns:** Model handle, or NULL on failure

#### Introspection

##### `psam_get_stats`

```c
psam_error_t psam_get_stats(const psam_model_t* model, psam_stats_t* out_stats);
```

Get model statistics.

##### `psam_error_string`

```c
const char* psam_error_string(psam_error_t error);
```

Get human-readable error message.

##### `psam_version`

```c
const char* psam_version(void);
```

Get library version string.

---

## JavaScript/TypeScript API

### Installation

```bash
npm install @foundation42/libpsam
```

### Types

```typescript
type TokenId = number;

interface TrainOptions {
  window: number;
  alpha: number;
  gamma: number;
  eta: number;
  kFunction: number;
  kContent: number;
  kRare: number;
  edgeDropout: number;
  minEvidence: number;
  enableIDF: boolean;
  enablePPMI: boolean;
}

interface InferenceResult {
  ids: TokenId[];
  scores: Float32Array;
  probabilities?: Float32Array;
}

interface ExplainTerm {
  source: TokenId;
  offset: number;
  weight: number;
  idf: number;
  decay: number;
  contribution: number;
}

interface ExplainResult {
  candidate: TokenId;
  total: number;
  bias: number;
  termCount: number;
  terms: ExplainTerm[];
}

interface ModelStats {
  vocabSize: number;
  rowCount: number;
  edgeCount: number;
  totalTokens?: number;
  memoryBytes: number;
}
```

### Functions

#### `createPSAM`

```typescript
function createPSAM(
  vocabSize: number,
  window: number,
  topK: number,
  prefer?: 'native' | 'wasm' | 'auto'
): TrainablePSAM;
```

Create a PSAM model using the best available implementation.

#### `isNativeAvailable`

```typescript
function isNativeAvailable(): boolean;
```

Check if native FFI implementation is available.

#### `isWASMAvailable`

```typescript
function isWASMAvailable(): boolean;
```

Check if WASM implementation is available.

### Classes

#### `PSAMNative`

Native implementation (Node.js, Bun).

```typescript
class PSAMNative implements TrainablePSAM {
  constructor(vocabSize: number, window: number, topK: number);

  trainToken(token: TokenId): void;
  trainBatch(tokens: TokenId[] | Uint32Array): void;
  finalizeTraining(): void;

  predict(context: TokenId[], maxPredictions?: number): InferenceResult;
  explain(context: TokenId[], candidateToken: TokenId, maxTerms?: number): ExplainResult;
  sample(context: TokenId[], temperature?: number): TokenId;

  addLayer(layerId: string, overlay: PSAMNative, weight: number): void;
  removeLayer(layerId: string): void;
  updateLayerWeight(layerId: string, newWeight: number): void;

  save(path: string): void;
  static load(path: string): PSAMNative;

  stats(): ModelStats;
  destroy(): void;

  static version(): string;
}
```

#### `PSAMWASM`

WASM implementation (browser).

```typescript
class PSAMWASM implements TrainablePSAM {
  static async create(vocabSize: number, window: number, topK: number): Promise<PSAMWASM>;

  // Same methods as PSAMNative
}
```

---

## Python API

### Installation

```bash
pip install libpsam
```

### Types

```python
from dataclasses import dataclass

@dataclass
class ModelStats:
    vocab_size: int
    row_count: int
    edge_count: int
    total_tokens: int
    memory_bytes: int

@dataclass
class ExplainTerm:
    source: int
    offset: int
    weight: float
    idf: float
    decay: float
    contribution: float


@dataclass
class ExplainResult:
    candidate: int
    total: float
    bias: float
    term_count: int
    terms: List[ExplainTerm]
```

### Classes

#### `PSAM`

```python
class PSAM:
    def __init__(self, vocab_size: int, window: int, top_k: int):
        """Create a new PSAM model"""

    def train_token(self, token: int) -> None:
        """Process a single token during training"""

    def train_batch(self, tokens: List[int]) -> None:
        """Process a batch of tokens"""

    def finalize_training(self) -> None:
        """Finalize training (required before inference)"""

    def predict(
        self,
        context: List[int],
        max_predictions: Optional[int] = None
    ) -> Tuple[List[int], np.ndarray]:
        """Generate predictions. Returns (token_ids, scores)"""

    def explain(
        self,
        context: List[int],
        candidate_token: int,
        max_terms: Optional[int] = None
    ) -> ExplainResult:
        """Explain why a token was predicted. Returns scores and top terms."""

    def sample(self, context: List[int], temperature: float = 1.0) -> int:
        """Sample a single token from distribution"""

    def add_layer(self, layer_id: str, overlay: PSAM, weight: float) -> None:
        """Add overlay layer for domain adaptation"""

    def remove_layer(self, layer_id: str) -> None:
        """Remove layer by ID"""

    def update_layer_weight(self, layer_id: str, new_weight: float) -> None:
        """Update layer weight"""

    def save(self, path: str) -> None:
        """Save model to file"""

    @classmethod
    def load(cls, path: str) -> PSAM:
        """Load model from file"""

    def stats(self) -> ModelStats:
        """Get model statistics"""

    def destroy(self) -> None:
        """Cleanup resources"""

    @staticmethod
    def version() -> str:
        """Get library version"""

    @property
    def vocab_size(self) -> int:
        """Vocabulary size"""

    @property
    def window(self) -> int:
        """Context window size"""

    @property
    def top_k(self) -> int:
        """Top-K predictions"""
```

### Functions

#### `is_library_available`

```python
def is_library_available() -> bool:
    """Check if native library is available"""
```

### Exceptions

#### `PSAMError`

```python
class PSAMError(Exception):
    """PSAM library error"""
```

---

## Common Concepts

### Token IDs

All APIs use integer token IDs (0 to vocab_size-1). You must maintain your own vocabulary mapping:

```python
# Build vocabulary
vocab = {"the": 0, "quick": 1, "brown": 2, "fox": 3}
inv_vocab = {v: k for k, v in vocab.items()}

# Convert to IDs
text = "the quick brown"
token_ids = [vocab[word] for word in text.split()]

# Train
psam.train_batch(token_ids)

# Predict
predictions = psam.predict(token_ids)

# Convert back
words = [inv_vocab[tid] for tid in predictions.ids]
```

### Context Window

The model uses a sliding window during training to capture positional associations. During inference, provide up to `window` tokens as context:

```python
# Window size: 8
psam = PSAM(vocab_size=1000, window=8, top_k=10)

# Context can be 1 to 8 tokens
psam.predict([1, 2, 3])           # 3 tokens
psam.predict([1, 2, 3, 4, 5, 6])  # 6 tokens

# If more than window tokens, only last `window` are used
psam.predict([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Uses last 8
```

### Training Workflow

1. **Create** model with vocabulary size
2. **Train** on token sequences
3. **Finalize** training (builds CSR matrix)
4. **Infer** predictions

```python
psam = PSAM(vocab_size=100, window=8, top_k=10)
psam.train_batch(tokens)        # Can call multiple times
psam.finalize_training()         # Required before inference
predictions = psam.predict(ctx)  # Now ready for inference
```

### Layer Composition

Hot-swap domain-specific layers without retraining:

```python
base = PSAM.load("general.psam")
medical = PSAM.load("medical.psam")

# Add medical knowledge with 1.5× weight
base.add_layer("medical", medical, weight=1.5)

# Predictions now blend both models
predictions = base.predict(context)

# Switch domains
base.remove_layer("medical")
base.add_layer("legal", legal_model, weight=2.0)
```

**Notes:**
- Overlay models must have same vocab size
- Overlay models are NOT owned by base (destroy separately)
- Multiple layers can be active simultaneously
- Layer weights are multiplicative

### File Formats

#### `.psam` - Single Model Format

Binary format for individual models:
- Magic number + version
- Provenance metadata (timestamp, creator, dataset hash)
- Configuration (vocab size, window, hyperparameters)
- CSR sparse matrix
- Bias, IDF, unigram counts

Compatible across all language bindings and platforms.

#### `.psamc` - Composite Model Format

Extensible format for model composition with integrity verification (C API only currently):

```c
#include <psam_composite.h>
#include <sys/stat.h>

// Create composite with hyperparameters and manifest
psamc_hyperparams_t config = PSAMC_PRESET_BALANCED_CONFIG;
config.alpha = 0.12;  // Custom tweak

psamc_model_ref_t refs[] = {
    {
        .url = "./models/base.psam",
        .size = 0,  // filled after hashing
    }
};

psamc_source_t sources[] = {
    {
        .label = "training-corpus",
        .uri = "s3://datasets/corpus-v1",
        .license = "CC-BY-4.0"
    }
};

psamc_manifest_t manifest = {
    .num_references = 1,
    .refs = refs,
    .source_count = 1,
    .sources = sources,
    .created_timestamp = time(NULL),
    .created_by = "my-trainer v1.0.0",
};

psamc_sha256_file("training_data.txt", &manifest.source_hash);
psamc_sha256_file(refs[0].url, &refs[0].sha256);
#ifdef _POSIX_VERSION
struct stat st;
if (stat(refs[0].url, &st) == 0) {
    refs[0].size = (uint64_t)st.st_size;
}
#endif

psamc_save("model.psamc", base_model, &config, &manifest);

// Load with integrity verification
psamc_composite_t* composite = psamc_load("model.psamc", true);
if (composite) {
    // Access composite->hyperparams / composite->manifest
    psamc_free(composite);
}
```

**Features:**
- SHA-256 integrity checking for external model references
- Hyperparameter storage (α, K, IDF, PPMI) for exact replay
- Provenance tracking (creator, timestamp, source hash)
- Preset configurations (FAST, BALANCED, ACCURATE, TINY)
- External references with semver version checking
- Optional dataset/source metadata (`meta.sources[]`)
- Layer composition for domain adaptation

See **[.psamc Format Specification](./PSAMC_FORMAT.md)** for complete details.

### Thread Safety

**C library:**
- Multiple threads can call `psam_predict()` simultaneously (read lock)
- Layer operations are exclusive (write lock)
- Training must be single-threaded

**Language bindings:**
- Follow C library semantics
- GIL in Python may limit parallelism
- Use process pools for parallel inference in Python

### Error Handling

**C:**
```c
psam_error_t err = psam_save(model, "model.psam");
if (err != PSAM_OK) {
    fprintf(stderr, "Error: %s\n", psam_error_string(err));
}
```

**JavaScript:**
```javascript
try {
    psam.save('model.psam');
} catch (error) {
    console.error('Error:', error.message);
}
```

**Python:**
```python
from psam import PSAM, PSAMError

try:
    psam.save('model.psam')
except PSAMError as e:
    print(f'Error: {e}')
```

---

## Performance Tips

1. **Batch training** - Use `train_batch()` instead of repeated `train_token()` calls
2. **Vocabulary size** - Use only as large as needed (affects memory and speed)
3. **Context length** - Shorter contexts are faster to predict
4. **Top-K** - Smaller top-K is faster
5. **Layer composition** - Each layer adds overhead; use judiciously

## See Also

- [C Library README](../core/README.md)
- [JavaScript Bindings](../bindings/javascript/README.md)
- [Python Bindings](../bindings/python/README.md)
- [Examples](../examples/)
