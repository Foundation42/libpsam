/**
 * csr.c - Compressed Sparse Row storage and accumulation
 *
 * This is the core accumulation kernel that already exists in accumulator.c.
 * We'll migrate that proven implementation here.
 */

#include "psam.h"
#include <stdint.h>

/**
 * CSR accumulation kernel - the proven fast path from accumulator.c
 *
 * Given CSR-formatted association matrix and context weights,
 * accumulates scores for all vocabulary tokens.
 */
void psam_csr_accumulate(
    const uint32_t* row_offsets,
    const uint32_t* targets,
    const int16_t* weights,
    const float* row_scales,
    const float* row_contrib,
    uint32_t row_count,
    uint32_t vocab_size,
    float* out_scores
) {
    /* Initialize scores to zero */
    for (uint32_t i = 0; i < vocab_size; i++) {
        out_scores[i] = 0.0f;
    }

    /* Accumulate contributions from each active row */
    for (uint32_t row = 0; row < row_count; row++) {
        const uint32_t start = row_offsets[row];
        const uint32_t end = row_offsets[row + 1];
        const float factor = row_contrib[row] * row_scales[row];

        /* Inner loop: accumulate edge contributions */
        for (uint32_t edge = start; edge < end; edge++) {
            out_scores[targets[edge]] += factor * weights[edge];
        }
    }
}

/**
 * SIMD-accelerated version (future optimization)
 * TODO: Implement with AVX2/NEON when ready
 */
void psam_csr_accumulate_simd(
    const uint32_t* row_offsets,
    const uint32_t* targets,
    const int16_t* weights,
    const float* row_scales,
    const float* row_contrib,
    uint32_t row_count,
    uint32_t vocab_size,
    float* out_scores
) {
    /* For now, fallback to scalar version */
    psam_csr_accumulate(row_offsets, targets, weights, row_scales,
                        row_contrib, row_count, vocab_size, out_scores);
}
