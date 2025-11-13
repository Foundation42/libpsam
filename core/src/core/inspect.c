/**
 * inspect.c - Advanced introspection APIs for model analysis and visualization
 *
 * Provides APIs to extract internal model data for:
 * - Network visualization (edge extraction)
 * - Configuration introspection
 * - Memory-based model loading (WASM/browser use)
 */

#include "../psam_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================ Edge Extraction ============================ */

typedef struct {
    psam_edge_t edge;
    float abs_weight;  /* For sorting by absolute weight */
} edge_with_sort_t;

static int compare_edges_by_weight_desc(const void* a, const void* b) {
    const edge_with_sort_t* ea = (const edge_with_sort_t*)a;
    const edge_with_sort_t* eb = (const edge_with_sort_t*)b;

    if (ea->abs_weight > eb->abs_weight) return -1;
    if (ea->abs_weight < eb->abs_weight) return 1;
    return 0;
}

int psam_get_edges(
    const psam_model_t* model,
    uint32_t source_token,
    float min_weight,
    size_t max_edges,
    psam_edge_t* out_edges
) {
    if (!model || !out_edges) {
        return PSAM_ERR_NULL_PARAM;
    }

    if (!model->is_finalized || !model->csr) {
        return PSAM_ERR_NOT_TRAINED;
    }

    if (max_edges == 0) {
        return 0;
    }

    const csr_storage_t* csr = model->csr;
    const row_descriptor_t* descriptors = model->row_descriptors;
    const uint32_t descriptor_count = model->row_descriptor_count;

    if (descriptor_count == 0 || !descriptors) {
        return 0;  /* No edges to extract */
    }

    /* Allocate temporary storage for all candidate edges */
    const size_t max_candidates = (source_token == UINT32_MAX)
        ? csr->edge_count
        : 1024;  /* Initial guess for single source */

    edge_with_sort_t* candidates = (edge_with_sort_t*)malloc(
        max_candidates * sizeof(edge_with_sort_t)
    );
    if (!candidates) {
        return PSAM_ERR_OUT_OF_MEMORY;
    }

    size_t candidate_count = 0;

    /* Iterate through row descriptors */
    for (uint32_t desc_idx = 0; desc_idx < descriptor_count; desc_idx++) {
        const row_descriptor_t* desc = &descriptors[desc_idx];

        /* Filter by source token if specified */
        if (source_token != UINT32_MAX && desc->source != source_token) {
            continue;
        }

        /* Get CSR row range for this descriptor */
        uint32_t row_start = csr->row_offsets[desc_idx];
        uint32_t row_end = csr->row_offsets[desc_idx + 1];
        float row_scale = csr->row_scales[desc_idx];

        /* Extract edges from this row */
        for (uint32_t edge_idx = row_start; edge_idx < row_end; edge_idx++) {
            uint32_t target = csr->targets[edge_idx];
            int16_t weight_raw = csr->weights[edge_idx];
            float weight = ((float)weight_raw / 1000.0f) * row_scale;

            /* Filter by minimum weight */
            float abs_weight = weight < 0 ? -weight : weight;
            if (abs_weight < min_weight) {
                continue;
            }

            /* Reallocate if needed */
            if (candidate_count >= max_candidates) {
                size_t new_max = max_candidates * 2;
                edge_with_sort_t* new_candidates = (edge_with_sort_t*)realloc(
                    candidates, new_max * sizeof(edge_with_sort_t)
                );
                if (!new_candidates) {
                    free(candidates);
                    return PSAM_ERR_OUT_OF_MEMORY;
                }
                candidates = new_candidates;
            }

            /* Add candidate edge */
            candidates[candidate_count].edge.source_token = desc->source;
            candidates[candidate_count].edge.target_token = target;
            candidates[candidate_count].edge.offset = (int16_t)desc->offset;
            candidates[candidate_count].edge.weight = weight;
            candidates[candidate_count].edge.observations = desc->totalObservations;
            candidates[candidate_count].abs_weight = abs_weight;
            candidate_count++;
        }
    }

    /* Sort by absolute weight (descending) */
    qsort(candidates, candidate_count, sizeof(edge_with_sort_t),
          compare_edges_by_weight_desc);

    /* Copy top edges to output buffer */
    size_t output_count = (candidate_count < max_edges) ? candidate_count : max_edges;
    for (size_t i = 0; i < output_count; i++) {
        out_edges[i] = candidates[i].edge;
    }

    free(candidates);
    return (int)output_count;
}

/* ============================ Configuration Access ============================ */

psam_error_t psam_get_config(const psam_model_t* model, psam_config_t* out_config) {
    if (!model || !out_config) {
        return PSAM_ERR_NULL_PARAM;
    }

    /* Cast away const for lock - safe for read lock */
    psam_model_t* m = (psam_model_t*)model;
    psam_lock_rdlock(&m->lock);
    *out_config = model->config;
    psam_lock_unlock_rd(&m->lock);

    return PSAM_OK;
}

/* ============================ Memory-based Loading ============================ */

/* Helper structure for memory-based file operations */
typedef struct {
    const uint8_t* data;
    size_t size;
    size_t position;
} memory_file_t;

static size_t mem_read(void* ptr, size_t size, size_t count, memory_file_t* mf) {
    size_t bytes_to_read = size * count;
    size_t bytes_available = mf->size - mf->position;

    if (bytes_to_read > bytes_available) {
        bytes_to_read = bytes_available;
    }

    if (bytes_to_read > 0) {
        memcpy(ptr, mf->data + mf->position, bytes_to_read);
        mf->position += bytes_to_read;
    }

    return bytes_to_read / size;
}

#define PSAM_MAGIC 0x4D415350  /* "PSAM" in little-endian */
#define PSAM_VERSION 2

psam_model_t* psam_load_from_memory(const void* buffer, size_t size) {
    if (!buffer || size < 8) {
        return NULL;
    }

    memory_file_t mf = {
        .data = (const uint8_t*)buffer,
        .size = size,
        .position = 0
    };

    psam_model_t* model = NULL;
    csr_storage_t* csr = NULL;

    /* 1. Read and validate header */
    uint32_t magic, version;
    if (mem_read(&magic, sizeof(uint32_t), 1, &mf) != 1 ||
        mem_read(&version, sizeof(uint32_t), 1, &mf) != 1) {
        return NULL;
    }

    if (magic != PSAM_MAGIC) {
        return NULL;  /* Invalid file format */
    }

    if (version < 1 || version > PSAM_VERSION) {
        return NULL;  /* Unsupported version */
    }

    /* 2. Read provenance metadata (version >= 2) */
    psam_provenance_t provenance;
    memset(&provenance, 0, sizeof(provenance));

    if (version >= 2) {
        if (mem_read(&provenance.created_timestamp, sizeof(uint64_t), 1, &mf) != 1) {
            return NULL;
        }

        char created_by_raw[PSAM_CREATED_BY_MAX];
        if (mem_read(created_by_raw, sizeof(char), PSAM_CREATED_BY_MAX, &mf) != PSAM_CREATED_BY_MAX) {
            return NULL;
        }
        memcpy(provenance.created_by, created_by_raw, PSAM_CREATED_BY_MAX);
        provenance.created_by[PSAM_CREATED_BY_MAX - 1] = '\0';

        if (mem_read(provenance.source_hash, sizeof(uint8_t), PSAM_SOURCE_HASH_SIZE, &mf) != PSAM_SOURCE_HASH_SIZE) {
            return NULL;
        }
    } else {
        provenance.created_timestamp = 0;
        strncpy(provenance.created_by, "legacy-psam", PSAM_CREATED_BY_MAX - 1);
        memset(provenance.source_hash, 0, PSAM_SOURCE_HASH_SIZE);
    }

    /* 3. Read config */
    psam_config_t config;
    memset(&config, 0, sizeof(config));

    if (mem_read(&config.vocab_size, sizeof(uint32_t), 1, &mf) != 1 ||
        mem_read(&config.window, sizeof(uint32_t), 1, &mf) != 1 ||
        mem_read(&config.top_k, sizeof(uint32_t), 1, &mf) != 1 ||
        mem_read(&config.alpha, sizeof(float), 1, &mf) != 1 ||
        mem_read(&config.min_evidence, sizeof(float), 1, &mf) != 1 ||
        mem_read(&config.enable_idf, sizeof(uint8_t), 1, &mf) != 1 ||
        mem_read(&config.enable_ppmi, sizeof(uint8_t), 1, &mf) != 1 ||
        mem_read(&config.edge_dropout, sizeof(float), 1, &mf) != 1) {
        return NULL;
    }

    /* 4. Create model with loaded config */
    model = psam_create_with_config(&config);
    if (!model) {
        return NULL;
    }

    /* Apply provenance */
    psam_lock_wrlock(&model->lock);
    model->provenance = provenance;
    psam_lock_unlock_wr(&model->lock);

    /* 5. Read model metadata */
    if (mem_read(&model->total_tokens, sizeof(uint64_t), 1, &mf) != 1) {
        goto error;
    }

    /* 6. Read CSR dimensions */
    uint32_t row_count;
    uint64_t edge_count;

    if (mem_read(&row_count, sizeof(uint32_t), 1, &mf) != 1 ||
        mem_read(&edge_count, sizeof(uint64_t), 1, &mf) != 1) {
        goto error;
    }

    /* 7. Read CSR arrays (if present) */
    if (row_count > 0 && edge_count > 0) {
        csr = (csr_storage_t*)calloc(1, sizeof(csr_storage_t));
        if (!csr) {
            goto error;
        }

        csr->row_count = row_count;
        csr->edge_count = edge_count;

        /* Allocate CSR arrays */
        csr->row_offsets = (uint32_t*)malloc((row_count + 1) * sizeof(uint32_t));
        csr->targets = (uint32_t*)malloc(edge_count * sizeof(uint32_t));
        csr->weights = (int16_t*)malloc(edge_count * sizeof(int16_t));
        csr->row_scales = (float*)malloc(row_count * sizeof(float));

        if (!csr->row_offsets || !csr->targets || !csr->weights || !csr->row_scales) {
            goto error;
        }

        /* Read CSR data */
        if (mem_read(csr->row_offsets, sizeof(uint32_t), row_count + 1, &mf) != row_count + 1 ||
            mem_read(csr->targets, sizeof(uint32_t), edge_count, &mf) != edge_count ||
            mem_read(csr->weights, sizeof(int16_t), edge_count, &mf) != edge_count ||
            mem_read(csr->row_scales, sizeof(float), row_count, &mf) != row_count) {
            goto error;
        }

        model->csr = csr;
    }

    /* 8. Read bias array */
    if (mem_read(model->bias, sizeof(float), config.vocab_size, &mf) != config.vocab_size) {
        goto error;
    }

    /* 9. Read unigram counts */
    if (mem_read(model->unigram_counts, sizeof(uint32_t), config.vocab_size, &mf) != config.vocab_size) {
        goto error;
    }

    /* 10. Read row descriptors */
    uint32_t desc_count;
    if (mem_read(&desc_count, sizeof(uint32_t), 1, &mf) != 1) {
        fprintf(stderr, "[inspect.c] Failed to read descriptor count at position %zu/%zu\n", mf.position, mf.size);
        goto error;
    }

    fprintf(stderr, "[inspect.c] Read descriptor count: %u at position %zu/%zu\n", desc_count, mf.position, mf.size);

    if (desc_count > 0) {
        model->row_descriptors = (row_descriptor_t*)malloc(desc_count * sizeof(row_descriptor_t));
        if (!model->row_descriptors) {
            fprintf(stderr, "[inspect.c] Failed to allocate memory for %u descriptors\n", desc_count);
            goto error;
        }

        size_t read_count = mem_read(model->row_descriptors, sizeof(row_descriptor_t), desc_count, &mf);
        if (read_count != desc_count) {
            fprintf(stderr, "[inspect.c] Failed to read descriptors: got %zu, expected %u\n", read_count, desc_count);
            goto error;
        }

        model->row_descriptor_count = desc_count;
        fprintf(stderr, "[inspect.c] Successfully loaded %u row descriptors\n", desc_count);
    } else {
        fprintf(stderr, "[inspect.c] WARNING: No row descriptors in file (desc_count=0)\n");
    }

    /* Mark model as finalized */
    model->is_finalized = true;

    return model;

error:
    if (csr) {
        free(csr->row_offsets);
        free(csr->targets);
        free(csr->weights);
        free(csr->row_scales);
        free(csr);
    }
    if (model) {
        psam_destroy(model);
    }
    return NULL;
}
