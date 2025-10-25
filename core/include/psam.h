/**
 * libpsam - Position-Specific Association Memory
 *
 * High-performance C library for sequence prediction using sparse associative memory.
 * Designed for fast FFI integration with Node.js and other language runtimes.
 *
 * Architecture:
 * - CSR (Compressed Sparse Row) storage for memory efficiency
 * - Multi-layer composition for domain adaptation ("HAL memory cartridge")
 * - Thread-safe inference with read-write locks
 * - Batch API for high-throughput scenarios
 *
 * Performance Targets:
 * - Latency: 0.01-0.1ms per inference (vs 2ms subprocess spawn)
 * - Throughput: 10,000-100,000 inferences/sec sequential
 * - Batch throughput: 100,000-1M inferences/sec
 */

#ifndef PSAM_H
#define PSAM_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "psam_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================ Types ============================ */

/**
 * Opaque handle to a PSAM model.
 * Internal structure is hidden to allow implementation flexibility.
 */
typedef struct psam_model psam_model_t;

/**
 * Opaque handle to a composite model builder/runtime.
 */
typedef struct psam_composite psam_composite_t;

/**
 * Configuration for model creation.
 */
typedef struct {
    uint32_t vocab_size;      /* Total vocabulary size */
    uint32_t window;          /* Context window size */
    uint32_t top_k;           /* Number of predictions to return */
    float alpha;              /* Distance decay parameter (default: 0.1) */
    float min_evidence;       /* Minimum edge count threshold (default: 1) */
    bool enable_idf;          /* Enable IDF weighting (default: true) */
    bool enable_ppmi;         /* Enable PPMI transformation (default: true) */
    float edge_dropout;       /* Edge dropout rate 0-1 (default: 0) */
} psam_config_t;

/**
 * Provenance metadata stored with serialized models (.psam/.psamc).
 * Enables exact replay and audit of training sources.
 */
#define PSAM_CREATED_BY_MAX 128
#define PSAM_SOURCE_HASH_SIZE 32
#define PSAM_LAYER_ID_MAX 64

/**
 * Supported composite topologies. Currently only layered builders are implemented,
 * but the enum leaves room for sequenced/routed configurations.
 */
typedef enum {
    PSAM_COMPOSITE_LAYERED = 1,
} psam_composite_topology_t;

/**
 * Lightweight descriptor for inspecting layers attached to a composite.
 */
typedef struct {
    char id[PSAM_LAYER_ID_MAX];
    float weight;
} psam_composite_layer_info_t;

typedef struct {
    uint64_t created_timestamp;                /* Unix timestamp of model creation */
    char created_by[PSAM_CREATED_BY_MAX];      /* Tool/user identifier */
    uint8_t source_hash[PSAM_SOURCE_HASH_SIZE];/* SHA-256 of training data (optional) */
} psam_provenance_t;

/**
 * Single prediction result with token ID, raw score, and calibrated probability.
 */
typedef struct {
    uint32_t token;           /* Predicted token ID */
    float score;              /* Raw association score */
    float calibrated_prob;    /* Calibrated probability (0-1 range) */
} psam_prediction_t;

/**
 * Explanation term showing contributing factors for a prediction.
 * Provides full traceability of why a token was predicted.
 */
typedef struct {
    uint32_t source_token;    /* Context token that contributed */
    int16_t  rel_offset;      /* Relative position delta (e.g., -3 means three tokens back) */
    float    weight_ppmi;     /* Base association weight (PPMI-adjusted) */
    float    idf;             /* IDF weighting factor */
    float    decay;           /* Distance decay factor */
    float    contribution;    /* Final contribution (weight × idf × decay) */
} psam_explain_term_t;

/**
 * Aggregate result metadata for an explanation request.
 */
typedef struct {
    uint32_t candidate;       /* Token being explained */
    float    total_score;     /* Final score used by the sampler (bias + contributions) */
    float    bias_score;      /* Baseline bias score for the candidate */
    int32_t  term_count;      /* Total number of contributing terms discovered */
} psam_explain_result_t;

/**
 * Training statistics for monitoring.
 */
typedef struct {
    uint32_t vocab_size;      /* Active vocabulary size */
    uint32_t row_count;       /* Number of CSR rows */
    uint64_t edge_count;      /* Total number of associations */
    uint64_t total_tokens;    /* Total tokens processed */
    size_t memory_bytes;      /* Approximate memory usage */
} psam_stats_t;

/**
 * Error codes returned by libpsam functions.
 */
typedef enum {
    PSAM_OK = 0,              /* Success */
    PSAM_ERR_NULL_PARAM = -1, /* Null parameter provided */
    PSAM_ERR_INVALID_CONFIG = -2, /* Invalid configuration */
    PSAM_ERR_OUT_OF_MEMORY = -3,  /* Memory allocation failed */
    PSAM_ERR_IO = -4,         /* File I/O error */
    PSAM_ERR_INVALID_MODEL = -5,  /* Corrupted or invalid model */
    PSAM_ERR_NOT_TRAINED = -6,    /* Model not finalized for inference */
    PSAM_ERR_LAYER_NOT_FOUND = -7 /* Layer ID not found */
} psam_error_t;

/* ============================ Lifecycle ============================ */

/**
 * Create a new PSAM model with default configuration.
 *
 * @param vocab_size Maximum vocabulary size
 * @param window Context window size (number of preceding tokens)
 * @param top_k Number of top predictions to return
 * @return Pointer to new model, or NULL on allocation failure
 *
 * Example:
 *   psam_model_t* model = psam_create(50000, 8, 32);
 *   if (!model) {
 *     fprintf(stderr, "Failed to create model\n");
 *     return 1;
 *   }
 */
PSAM_API psam_model_t* psam_create(uint32_t vocab_size, uint32_t window, uint32_t top_k);

/**
 * Create a new PSAM model with custom configuration.
 *
 * @param config Configuration structure
 * @return Pointer to new model, or NULL on allocation failure
 *
 * Example:
 *   psam_config_t config = {
 *     .vocab_size = 50000,
 *     .window = 8,
 *     .top_k = 32,
 *     .alpha = 0.1,
 *     .min_evidence = 1,
 *     .enable_idf = true,
 *     .enable_ppmi = true,
 *     .edge_dropout = 0.0
 *   };
 *   psam_model_t* model = psam_create_with_config(&config);
 */
PSAM_API psam_model_t* psam_create_with_config(const psam_config_t* config);

/**
 * Destroy a PSAM model and free all associated memory.
 * Safe to call with NULL pointer.
 *
 * @param model Model to destroy
 *
 * Example:
 *   psam_destroy(model);
 *   model = NULL;
 */
PSAM_API void psam_destroy(psam_model_t* model);

/**
 * Get provenance metadata for a model (thread-safe).
 *
 * @param model Model handle
 * @param out_provenance Output structure to populate
 * @return PSAM_OK on success, error code otherwise
 */
PSAM_API psam_error_t psam_get_provenance(const psam_model_t* model, psam_provenance_t* out_provenance);

/**
 * Set provenance metadata for a model (thread-safe).
 * Call before psam_save() to embed metadata in the file header.
 *
 * @param model Model handle
 * @param provenance Provenance data to store
 * @return PSAM_OK on success, error code otherwise
 */
PSAM_API psam_error_t psam_set_provenance(psam_model_t* model, const psam_provenance_t* provenance);

/* ============================ Training ============================ */

/**
 * Process a single token during training.
 * Builds co-occurrence statistics within the configured window.
 *
 * @param model Model to train
 * @param token Token ID to process (must be < vocab_size)
 * @return PSAM_OK on success, error code otherwise
 *
 * Example:
 *   for (size_t i = 0; i < num_tokens; i++) {
 *     psam_error_t err = psam_train_token(model, tokens[i]);
 *     if (err != PSAM_OK) {
 *       fprintf(stderr, "Training failed: %d\n", err);
 *       return err;
 *     }
 *   }
 */
PSAM_API psam_error_t psam_train_token(psam_model_t* model, uint32_t token);

/**
 * Process multiple tokens in a batch during training.
 * More efficient than individual calls to psam_train_token().
 *
 * @param model Model to train
 * @param tokens Array of token IDs
 * @param num_tokens Number of tokens in array
 * @return PSAM_OK on success, error code otherwise
 *
 * Example:
 *   uint32_t tokens[] = {1, 2, 3, 4, 5};
 *   psam_train_batch(model, tokens, 5);
 */
PSAM_API psam_error_t psam_train_batch(psam_model_t* model, const uint32_t* tokens, size_t num_tokens);

/**
 * Finalize training by computing PPMI/IDF transformations and building CSR storage.
 * Must be called before inference. Model becomes read-only after finalization.
 *
 * @param model Model to finalize
 * @return PSAM_OK on success, error code otherwise
 *
 * Example:
 *   psam_finalize_training(model);
 *   // Now ready for inference
 *   psam_predict(model, context, context_len, preds, top_k);
 */
PSAM_API psam_error_t psam_finalize_training(psam_model_t* model);

/* ============================ Inference ============================ */

/**
 * Generate predictions for a given context.
 * Returns top-K predictions sorted by score (descending).
 * Thread-safe: multiple threads can call this simultaneously.
 *
 * @param model Trained model (must be finalized)
 * @param context Array of context token IDs
 * @param context_len Number of tokens in context
 * @param out_preds Output buffer for predictions (caller-allocated)
 * @param max_preds Size of output buffer (typically model's top_k)
 * @return Number of predictions written (0 to max_preds), or negative error code
 *
 * Example:
 *   uint32_t context[] = {10, 20, 30};
 *   psam_prediction_t preds[32];
 *   int n = psam_predict(model, context, 3, preds, 32);
 *   if (n < 0) {
 *     fprintf(stderr, "Prediction failed: %d\n", n);
 *   } else {
 *     for (int i = 0; i < n; i++) {
 *       printf("Token %u: score=%.4f prob=%.4f\n",
 *              preds[i].token, preds[i].score, preds[i].calibrated_prob);
 *     }
 *   }
 */
PSAM_API int psam_predict(
    psam_model_t* model,
    const uint32_t* context,
    size_t context_len,
    psam_prediction_t* out_preds,
    size_t max_preds
);

/**
 * Batch inference: process multiple contexts in one call.
 * Internally parallelized for high throughput.
 *
 * @param model Trained model (must be finalized)
 * @param contexts Array of context arrays
 * @param context_lens Length of each context
 * @param batch_size Number of contexts
 * @param out_preds Array of prediction buffers (batch_size arrays)
 * @param max_preds_per_context Size of each prediction buffer
 * @param out_counts Array to receive number of predictions per context (batch_size elements)
 * @return PSAM_OK on success, error code otherwise
 *
 * Example:
 *   uint32_t ctx1[] = {1, 2, 3};
 *   uint32_t ctx2[] = {4, 5, 6};
 *   const uint32_t* contexts[] = {ctx1, ctx2};
 *   size_t lens[] = {3, 3};
 *   psam_prediction_t preds[2][32];
 *   int counts[2];
 *
 *   psam_predict_batch(model, contexts, lens, 2,
 *                      (psam_prediction_t**)preds, 32, counts);
 */
PSAM_API psam_error_t psam_predict_batch(
    psam_model_t* model,
    const uint32_t** contexts,
    const size_t* context_lens,
    size_t batch_size,
    psam_prediction_t** out_preds,
    size_t max_preds_per_context,
    int* out_counts
);

/**
 * Explain why a specific token was predicted for the given context.
 * Returns the top contributing association terms with full traceability.
 * This exposes PSAM's interpretability superpower: every prediction can be
 * traced back to specific (source_token, offset, weight) associations.
 *
 * @param model Trained model (must be finalized)
 * @param context Array of context token IDs
 * @param context_len Number of tokens in context
 * @param candidate_token Token ID to explain
 * @param out_terms Output buffer for explanation terms (caller-allocated, can be NULL if max_terms == 0)
 * @param max_terms Size of output buffer (0 to probe required size)
 * @param result   Metadata about the explanation (required)
 * @return PSAM_OK on success, negative error code otherwise
 *
 * Example:
 *   uint32_t context[] = {10, 20, 30};
 *   psam_explain_term_t terms[16];
 *   psam_explain_result_t info;
 *   psam_error_t err = psam_explain(model, context, 3, 42, terms, 16, &info);
 *   if (err == PSAM_OK) {
 *     int written = (info.term_count < 16) ? info.term_count : 16;
 *     for (int i = 0; i < written; i++) {
 *       printf("  Token %u (offset %+d): weight=%.3f × idf=%.3f × decay=%.3f = %.4f\n",
 *              terms[i].source_token, terms[i].rel_offset,
 *              terms[i].weight_ppmi, terms[i].idf, terms[i].decay,
 *              terms[i].contribution);
 *     }
 *     printf("Bias=%.4f ContributionSum=%.4f Total=%.4f\n",
 *            info.bias_score, info.total_score - info.bias_score, info.total_score);
 *   } else if (err == PSAM_OK && info.term_count > 16) {
 *     // allocate a larger buffer (info.term_count entries) and call again
 *   }
 */
PSAM_API psam_error_t psam_explain(
    psam_model_t* model,
    const uint32_t* context,
    size_t context_len,
    uint32_t candidate_token,
    psam_explain_term_t* out_terms,
    int max_terms,
psam_explain_result_t* result
);

/* ============================ Composite Model Builders ============================ */

/**
 * Create a layered composite builder around a finalized base model.
 * The composite does not take ownership of the base model; call psam_destroy
 * separately if needed.
 *
 * @param base_model Finalized model that supplies baseline predictions
 * @return Composite handle or NULL on allocation/validation failure
 */
PSAM_API psam_composite_t* psam_create_layered(psam_model_t* base_model);

/**
 * Destroy a composite builder/runtime. Does not destroy referenced models.
 */
PSAM_API void psam_composite_destroy(psam_composite_t* composite);

/**
 * Set the blending weight for the base model inside a composite.
 * Default weight is 1.0.
 */
PSAM_API psam_error_t psam_composite_set_base_weight(psam_composite_t* composite, float weight);

/**
 * Add a named layer to the composite.
 *
 * @param composite Composite handle
 * @param layer_id Unique identifier for the layer (up to PSAM_LAYER_ID_MAX chars)
 * @param layer_model Finalized model to blend
 * @param weight Relative blending weight for this layer
 */
PSAM_API psam_error_t psam_composite_add_layer(
    psam_composite_t* composite,
    const char* layer_id,
    psam_model_t* layer_model,
    float weight
);

/**
 * Remove a layer by identifier.
 */
PSAM_API psam_error_t psam_composite_remove_layer(psam_composite_t* composite, const char* layer_id);

/**
 * Update the blending weight for an existing layer.
 */
PSAM_API psam_error_t psam_composite_update_layer_weight(
    psam_composite_t* composite,
    const char* layer_id,
    float new_weight
);

/**
 * List active layers in a composite. Copies up to max_layers entries into out_layers.
 *
 * @return Number of layers written, or negative error code.
 */
PSAM_API int psam_composite_list_layers(
    const psam_composite_t* composite,
    psam_composite_layer_info_t* out_layers,
    size_t max_layers
);

/**
 * Run inference through a composite topology and return blended predictions.
 */
PSAM_API int psam_composite_predict(
    psam_composite_t* composite,
    const uint32_t* context,
    size_t context_len,
    psam_prediction_t* out_preds,
    size_t max_preds
);

/* ============================ Persistence ============================ */

/**
 * Save model to binary file.
 * Format is compact and optimized for fast loading.
 *
 * @param model Model to save
 * @param path File path
 * @return PSAM_OK on success, PSAM_ERR_IO on file error
 *
 * Example:
 *   psam_save(model, "my_model.psam");
 */
PSAM_API psam_error_t psam_save(const psam_model_t* model, const char* path);

/**
 * Load model from binary file.
 *
 * @param path File path
 * @return Pointer to loaded model, or NULL on error
 *
 * Example:
 *   psam_model_t* model = psam_load("my_model.psam");
 *   if (!model) {
 *     fprintf(stderr, "Failed to load model\n");
 *     return 1;
 *   }
 */
PSAM_API psam_model_t* psam_load(const char* path);

/* ============================ Introspection ============================ */

/**
 * Get model statistics (vocabulary size, edge count, memory usage, etc.).
 *
 * @param model Model to query
 * @param out_stats Output structure (caller-allocated)
 * @return PSAM_OK on success, error code otherwise
 *
 * Example:
 *   psam_stats_t stats;
 *   psam_get_stats(model, &stats);
 *   printf("Vocab: %u, Edges: %lu, Memory: %zu bytes\n",
 *          stats.vocab_size, stats.edge_count, stats.memory_bytes);
 */
PSAM_API psam_error_t psam_get_stats(const psam_model_t* model, psam_stats_t* out_stats);

/**
 * Get human-readable error message for an error code.
 *
 * @param error Error code
 * @return Error message string (static, do not free)
 */
PSAM_API const char* psam_error_string(psam_error_t error);

/**
 * Get libpsam version string.
 *
 * @return Version string (e.g., "1.0.0")
 */
PSAM_API const char* psam_version(void);

#ifdef __cplusplus
}
#endif

#endif /* PSAM_H */
