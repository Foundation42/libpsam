/**
 * psam_internal.h - Internal structures and utilities
 *
 * This header is NOT part of the public API. It's only for internal use
 * by libpsam implementation files.
 */

#ifndef PSAM_INTERNAL_H
#define PSAM_INTERNAL_H

#include "psam.h"
#include <stdbool.h>

#define PSAM_CONSENSUS_GAIN 0.5f

/* Platform-specific read-write lock */
#ifdef _WIN32
    #include <windows.h>
    typedef SRWLOCK psam_lock_t;

    #define psam_lock_init(lock) InitializeSRWLock(lock)
    #define psam_lock_destroy(lock) /* SRWLock doesn't need cleanup */
    #define psam_lock_rdlock(lock) AcquireSRWLockShared(lock)
    #define psam_lock_wrlock(lock) AcquireSRWLockExclusive(lock)
    #define psam_lock_unlock_rd(lock) ReleaseSRWLockShared(lock)
    #define psam_lock_unlock_wr(lock) ReleaseSRWLockExclusive(lock)
#else
    #include <pthread.h>
    typedef pthread_rwlock_t psam_lock_t;

    #define psam_lock_init(lock) pthread_rwlock_init(lock, NULL)
    #define psam_lock_destroy(lock) pthread_rwlock_destroy(lock)
    #define psam_lock_rdlock(lock) pthread_rwlock_rdlock(lock)
    #define psam_lock_wrlock(lock) pthread_rwlock_wrlock(lock)
    #define psam_lock_unlock_rd(lock) pthread_rwlock_unlock(lock)
    #define psam_lock_unlock_wr(lock) pthread_rwlock_unlock(lock)
#endif

/* ============================ Internal Data Structures ============================ */

/**
 * Provenance metadata (shared with public API).
 */
typedef psam_provenance_t psam_provenance_internal_t;

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
    psam_provenance_internal_t provenance;

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

    /* Thread safety */
    psam_lock_t lock;
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

/* From infer.c - shared utilities */
float compute_idf(const psam_model_t* model, uint32_t token);
int find_row_index(const psam_model_t* model, uint32_t token, uint32_t offset);
int compare_predictions(const void* a, const void* b);

#endif /* PSAM_INTERNAL_H */
