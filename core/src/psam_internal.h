/**
 * psam_internal.h - Internal structures and utilities
 *
 * This header is NOT part of the public API. It's only for internal use
 * by libpsam implementation files.
 */

#ifndef PSAM_INTERNAL_H
#define PSAM_INTERNAL_H

#include "psam.h"
#include <pthread.h>
#include <stdbool.h>

/* ============================ Internal Data Structures ============================ */

/**
 * Layer composition node (linked list)
 */
typedef struct layer_node {
    char id[64];
    psam_model_t* model;
    float weight;
    struct layer_node* next;
} layer_node_t;

/**
 * CSR (Compressed Sparse Row) storage
 */
typedef struct {
    uint32_t* row_offsets;    /* Row pointers (size: row_count + 1) */
    uint32_t* targets;        /* Column indices (size: edge_count) */
    int16_t* weights;         /* Edge weights (size: edge_count) */
    float* row_scales;        /* Row normalization (size: row_count) */
    uint32_t row_count;
    uint64_t edge_count;
} csr_storage_t;

/**
 * Row descriptor for CSR lookups
 */
typedef struct {
    uint32_t source;          /* Source token */
    uint32_t offset;          /* Offset in context window */
    uint32_t totalObservations;
} row_descriptor_t;

/**
 * Complete model structure (opaque to public API)
 */
struct psam_model {
    /* Configuration */
    psam_config_t config;

    /* Core data */
    csr_storage_t* csr;
    float* bias;              /* Bias terms (size: vocab_size) */
    uint32_t* unigram_counts; /* Token frequencies (size: vocab_size) */
    uint64_t total_tokens;

    /* Row metadata for CSR lookups */
    row_descriptor_t* row_descriptors;  /* Array of row descriptors */
    uint32_t row_descriptor_count;

    /* Training state */
    void* training_data;      /* Opaque training accumulator */
    bool is_finalized;

    /* Layer composition */
    layer_node_t* layers;

    /* Thread safety */
    pthread_rwlock_t lock;
};

/* ============================ Internal Function Declarations ============================ */

/* From train.c */
void psam_free_training_data(void* training_data);

/* From csr.c */
void psam_csr_accumulate(
    const uint32_t* row_offsets,
    const uint32_t* targets,
    const int16_t* weights,
    const float* row_scales,
    const float* row_contrib,
    uint32_t row_count,
    uint32_t vocab_size,
    float* out_scores
);

#endif /* PSAM_INTERNAL_H */
