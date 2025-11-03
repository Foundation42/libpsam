/**
 * psam_vocab_alignment.h - Vocabulary alignment for composite models
 *
 * This module enables composites to combine models with different vocabularies
 * by providing bidirectional token ID mapping between local and unified vocab spaces.
 *
 * Architecture:
 * ┌──────────────────────────────────────────────────────────┐
 * │ Unified Vocabulary (superset of all layers)              │
 * │ ID Space: [0..unified_vocab_size)                        │
 * └──────────────────────────────────────────────────────────┘
 *           ▲                    ▲                    ▲
 *           │                    │                    │
 *    ┌──────┴──────┐      ┌──────┴──────┐      ┌──────┴──────┐
 *    │  Layer 0    │      │  Layer 1    │      │  Layer 2    │
 *    │  Vocab      │      │  Vocab      │      │  Vocab      │
 *    │  [0..N0)    │      │  [0..N1)    │      │  [0..N2)    │
 *    └─────────────┘      └─────────────┘      └─────────────┘
 *
 * Each layer has:
 * - Dense local→unified map: O(1) lookup, local_vocab_size * 4 bytes
 * - Sparse unified→local map: O(log N) binary search or O(1) hash
 *
 * Usage:
 *   // Build alignment from models
 *   psam_vocab_alignment_t* align = psam_build_vocab_alignment(
 *       base_model, overlay_models, layer_ids, n_layers, &unified_size);
 *
 *   // Create aligned composite
 *   psam_composite_aligned_t* comp = psam_create_composite_aligned(
 *       base_model, align, true);
 *
 *   // Add layers (context automatically remapped)
 *   psam_composite_aligned_add_layer(comp, "hamlet", hamlet_model, 1.0, true);
 *
 *   // Predict (all remapping handled internally)
 *   psam_composite_aligned_predict(comp, context, ctx_len, preds, max_preds);
 */

#ifndef PSAM_VOCAB_ALIGNMENT_H
#define PSAM_VOCAB_ALIGNMENT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "psam_export.h"
#include "psam.h"

/* Forward declarations */
typedef struct psam_model psam_model_t;
typedef struct psam_composite psam_composite_t;

#ifdef __cplusplus
extern "C" {
#endif

/* Invalid token ID sentinel (used when token doesn't exist in vocab) */
#define PSAM_VOCAB_INVALID_ID UINT32_MAX

/**
 * Policy for handling tokens unknown to a layer
 */
typedef enum {
    PSAM_UNKNOWN_SKIP = 0,      /* Drop unknown tokens from layer's context */
    PSAM_UNKNOWN_MAP_UNK = 1,   /* Map to reserved UNK token (ID = vocab_size - 1) */
    PSAM_UNKNOWN_COVERAGE = 2   /* Weight layer by coverage: known_tokens / total_tokens */
} psam_unknown_policy_t;

/**
 * Coverage weighting rules for aligned composites
 */
typedef enum {
    PSAM_COVER_NONE = 0,      /* No coverage weighting */
    PSAM_COVER_LINEAR = 1,    /* Linear coverage: f(c) = c */
    PSAM_COVER_SQRT = 2       /* Sqrt coverage: f(c) = sqrt(c) */
} psam_coverage_rule_t;

/**
 * Sparse entry for unified→local mapping
 * Sorted by unified_id for binary search
 */
typedef struct {
    uint32_t unified_id;
    uint32_t local_id;
} psam_vocab_sparse_entry_t;

/**
 * Vocabulary remapping for a single layer
 *
 * Memory:
 *   Dense map: local_vocab_size × 4 bytes (typically 10-100KB)
 *   Sparse map: (tokens in layer) × 8 bytes
 *
 * Performance:
 *   local→unified: O(1) array lookup
 *   unified→local: O(log N) binary search (N = tokens in layer)
 */
typedef struct {
    uint32_t local_vocab_size;              /* Layer's original vocabulary size */
    uint32_t* local_to_unified;             /* Dense: [local_id] → unified_id */

    /* Sparse unified→local map (sorted by unified_id for binary search) */
    psam_vocab_sparse_entry_t* unified_to_local_sparse;
    uint32_t unified_to_local_count;        /* Number of entries in sparse map */

    /* Statistics for debugging */
    float coverage;                         /* Fraction: local_vocab / unified_vocab */
} psam_vocab_remap_t;

/**
 * Complete vocabulary alignment for a composite
 *
 * Contains:
 * - Unified vocabulary (superset of all layer vocabularies)
 * - Per-layer remapping tables (one for base + one per overlay)
 */
typedef struct {
    uint32_t unified_vocab_size;            /* Size of unified vocabulary */
    char** unified_tokens;                  /* Token strings [0..unified_vocab_size) */

    /* Per-layer remapping (index 0 = base, 1..n = overlays) */
    uint32_t num_layers;                    /* Total layers (base + overlays) */
    psam_vocab_remap_t* layer_remaps;       /* Remapping for each layer */

    /* Policy for unknown tokens */
    psam_unknown_policy_t unknown_policy;

    /* Memory ownership */
    bool owns_unified_tokens;               /* Whether to free unified_tokens on destroy */
} psam_vocab_alignment_t;

typedef struct psam_aligned_layer {
    char* id;                               /* Layer identifier */
    psam_model_t* model;                    /* Model pointer */
    float weight;                           /* Layer blending weight */
    float bias;                             /* Layer bias offset */
    bool owns_model;                        /* Ownership flag */
    uint32_t alignment_index;               /* Index into alignment->layer_remaps */
} psam_aligned_layer_t;

/**
 * Composite with vocabulary alignment
 *
 * Wraps a standard psam_composite_t and adds vocabulary remapping.
 * All predictions automatically remap between unified and local vocab spaces.
 */
typedef struct {
    psam_composite_t* composite;            /* Underlying composite (same-vocab assumption) */
    psam_vocab_alignment_t* alignment;      /* Vocabulary remapping */
    psam_unknown_policy_t unknown_policy;   /* Unknown token handling */
    psam_coverage_rule_t coverage_rule;     /* Coverage weighting rule */
    psam_model_t* base_model;               /* Base model reference */
    float base_weight;                      /* Base model weight */
    bool owns_base_model;                   /* Whether to free base model on destroy */
    psam_aligned_layer_t* layers;           /* Dynamic array of aligned layers */
    size_t layer_count;                     /* Number of aligned overlay layers */
    size_t layer_capacity;                  /* Allocated capacity for layers */
    bool owns_composite;                    /* Whether to free composite on destroy */
    bool owns_alignment;                    /* Whether to free alignment on destroy */
} psam_composite_aligned_t;

/**
 * Build vocabulary alignment from vocabulary files
 *
 * Creates unified vocabulary (superset) and bidirectional remapping tables.
 *
 * @param vocab_paths   Array of vocabulary TSV file paths
 * @param n_vocabs      Number of vocabulary files (typically n_layers + 1 for base)
 * @param layer_ids     Layer identifiers (for debugging, may be NULL)
 * @param out_unified_vocab_size  [out] Size of unified vocabulary
 * @return Alignment structure, or NULL on error
 *
 * Vocabulary files must be in TSV format: "id\ttoken\n"
 * IDs must be sequential starting at 0.
 *
 * Memory: Caller must free with psam_vocab_alignment_destroy()
 */
PSAM_API psam_vocab_alignment_t* psam_build_vocab_alignment_from_files(
    const char** vocab_paths,
    size_t n_vocabs,
    const char** layer_ids,
    uint32_t* out_unified_vocab_size
);

/**
 * Free vocabulary alignment structure
 */
PSAM_API void psam_vocab_alignment_destroy(psam_vocab_alignment_t* alignment);

/**
 * Create aligned composite from base model and alignment
 *
 * @param base          Base model (must match alignment's base layer)
 * @param alignment     Pre-built vocabulary alignment
 * @param owns_alignment Whether composite takes ownership of alignment
 * @param owns_base      Whether composite takes ownership of base model
 * @return Aligned composite, or NULL on error
 */
PSAM_API psam_composite_aligned_t* psam_create_composite_aligned(
    psam_model_t* base,
    psam_vocab_alignment_t* alignment,
    bool owns_alignment,
    bool owns_base
);

/**
 * Add overlay layer to aligned composite
 *
 * @param composite     Aligned composite
 * @param layer_id      Layer identifier
 * @param model         Overlay model (must match alignment's layer)
 * @param weight        Layer weight
 * @param owns_model    Whether composite takes ownership of model
 * @return 0 on success, negative on error
 */
PSAM_API int psam_composite_aligned_add_layer(
    psam_composite_aligned_t* composite,
    const char* layer_id,
    psam_model_t* model,
    float weight,
    bool owns_model
);

/**
 * Predict next tokens with aligned composite
 *
 * Context is in unified vocabulary space. Predictions are returned in unified space.
 * All remapping (unified→local for context, local→unified for predictions) is handled internally.
 *
 * @param composite     Aligned composite
 * @param context       Context token IDs (unified vocab space)
 * @param context_len   Number of context tokens
 * @param out_preds     Output predictions (unified vocab space)
 * @param max_preds     Maximum predictions to return
 * @return Number of predictions, or negative on error
 */
PSAM_API int psam_composite_aligned_predict(
    psam_composite_aligned_t* composite,
    const uint32_t* context,
    size_t context_len,
    void* out_preds,  /* psam_prediction_t* but avoiding circular include */
    size_t max_preds
);

PSAM_API int psam_composite_aligned_predict_with_sampler(
    psam_composite_aligned_t* composite,
    const uint32_t* context,
    size_t context_len,
    const psam_sampler_t* sampler,
    void* out_preds,  /* psam_prediction_t* */
    size_t max_preds
);

/**
 * Destroy aligned composite
 */
PSAM_API void psam_composite_aligned_destroy(psam_composite_aligned_t* composite);

/**
 * Configure coverage weighting rule for an aligned composite
 */
PSAM_API void psam_composite_aligned_set_coverage_rule(
    psam_composite_aligned_t* composite,
    psam_coverage_rule_t rule
);

/**
 * Override unknown-token policy (defaults to alignment->unknown_policy)
 */
PSAM_API void psam_composite_aligned_set_unknown_policy(
    psam_composite_aligned_t* composite,
    psam_unknown_policy_t policy
);

/**
 * Update the base weight used during aligned prediction
 */
PSAM_API int psam_composite_aligned_set_base_weight(
    psam_composite_aligned_t* composite,
    float weight
);

/**
 * Update an aligned layer's weight (by layer identifier)
 */
PSAM_API int psam_composite_aligned_update_layer_weight(
    psam_composite_aligned_t* composite,
    const char* layer_id,
    float new_weight
);

/**
 * Update an aligned layer's bias (by layer identifier)
 */
PSAM_API int psam_composite_aligned_update_layer_bias(
    psam_composite_aligned_t* composite,
    const char* layer_id,
    float new_bias
);

PSAM_API psam_composite_aligned_t* psam_composite_load_aligned(
    const char* path,
    bool verify_integrity
);

/* Debug/introspection helpers */

/**
 * Get coverage statistics for a layer
 *
 * @param alignment     Vocabulary alignment
 * @param layer_index   Layer index (0 = base, 1+ = overlays)
 * @return Coverage ratio (0.0 to 1.0), or -1.0 on error
 */
PSAM_API float psam_vocab_alignment_get_coverage(
    const psam_vocab_alignment_t* alignment,
    uint32_t layer_index
);

/**
 * Map token ID: unified → local (for a specific layer)
 *
 * @param remap         Layer's remapping table
 * @param unified_id    Token ID in unified vocab space
 * @return Local token ID, or PSAM_VOCAB_INVALID_ID if not in layer
 */
PSAM_API uint32_t psam_vocab_remap_unified_to_local(
    const psam_vocab_remap_t* remap,
    uint32_t unified_id
);

/**
 * Map token ID: local → unified (for a specific layer)
 *
 * @param remap         Layer's remapping table
 * @param local_id      Token ID in layer's local vocab space
 * @return Unified token ID, or PSAM_VOCAB_INVALID_ID if invalid
 */
PSAM_API uint32_t psam_vocab_remap_local_to_unified(
    const psam_vocab_remap_t* remap,
    uint32_t local_id
);

#ifdef __cplusplus
}
#endif

#endif /* PSAM_VOCAB_ALIGNMENT_H */
