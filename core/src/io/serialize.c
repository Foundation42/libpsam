/**
 * serialize.c - Binary serialization for model persistence
 *
 * File format (compatible with TypeScript io/serialize.ts):
 * - Magic number (4 bytes): "PSAM"
 * - Version (4 bytes): 2
 * - Provenance metadata (timestamp, created_by, source_hash)
 * - Config section
 * - Vocabulary section (optional)
 * - CSR section (row_offsets, targets, weights, row_scales)
 * - Bias section
 * - Unigram counts section
 */

#define _POSIX_C_SOURCE 200809L

#include "../psam_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PSAM_MAGIC 0x4D415350  /* "PSAM" in little-endian */
#define PSAM_VERSION 2

/* ============================ Serialization ============================ */

psam_error_t psam_save(const psam_model_t* model, const char* path) {
    if (!model || !path) {
        return PSAM_ERR_NULL_PARAM;
    }

    if (!model->is_finalized) {
        return PSAM_ERR_NOT_TRAINED;
    }

    FILE* f = fopen(path, "wb");
    if (!f) {
        return PSAM_ERR_IO;
    }

    psam_error_t result = PSAM_OK;

    /* 1. Write magic number and version */
    uint32_t magic = PSAM_MAGIC;
    uint32_t version = PSAM_VERSION;
    if (fwrite(&magic, sizeof(uint32_t), 1, f) != 1 ||
        fwrite(&version, sizeof(uint32_t), 1, f) != 1) {
        result = PSAM_ERR_IO;
        goto cleanup;
    }

    /* 2. Write provenance metadata */
    uint64_t created_timestamp = model->provenance.created_timestamp;
    if (fwrite(&created_timestamp, sizeof(uint64_t), 1, f) != 1) {
        result = PSAM_ERR_IO;
        goto cleanup;
    }

    char created_by_buffer[PSAM_CREATED_BY_MAX] = {0};
    if (model->provenance.created_by[0] != '\0') {
        snprintf(created_by_buffer, PSAM_CREATED_BY_MAX, "%s", model->provenance.created_by);
    }
    if (fwrite(created_by_buffer, sizeof(char), PSAM_CREATED_BY_MAX, f) != PSAM_CREATED_BY_MAX) {
        result = PSAM_ERR_IO;
        goto cleanup;
    }

    if (fwrite(model->provenance.source_hash, sizeof(uint8_t), PSAM_SOURCE_HASH_SIZE, f) != PSAM_SOURCE_HASH_SIZE) {
        result = PSAM_ERR_IO;
        goto cleanup;
    }

    /* 3. Write config */
    if (fwrite(&model->config.vocab_size, sizeof(uint32_t), 1, f) != 1 ||
        fwrite(&model->config.window, sizeof(uint32_t), 1, f) != 1 ||
        fwrite(&model->config.top_k, sizeof(uint32_t), 1, f) != 1 ||
        fwrite(&model->config.alpha, sizeof(float), 1, f) != 1 ||
        fwrite(&model->config.min_evidence, sizeof(float), 1, f) != 1 ||
        fwrite(&model->config.enable_idf, sizeof(uint8_t), 1, f) != 1 ||
        fwrite(&model->config.enable_ppmi, sizeof(uint8_t), 1, f) != 1 ||
        fwrite(&model->config.edge_dropout, sizeof(float), 1, f) != 1) {
        result = PSAM_ERR_IO;
        goto cleanup;
    }

    /* 4. Write model metadata */
    if (fwrite(&model->total_tokens, sizeof(uint64_t), 1, f) != 1) {
        result = PSAM_ERR_IO;
        goto cleanup;
    }

    /* 5. Write CSR dimensions */
    uint32_t row_count = model->csr ? model->csr->row_count : 0;
    uint64_t edge_count = model->csr ? model->csr->edge_count : 0;

    if (fwrite(&row_count, sizeof(uint32_t), 1, f) != 1 ||
        fwrite(&edge_count, sizeof(uint64_t), 1, f) != 1) {
        result = PSAM_ERR_IO;
        goto cleanup;
    }

    /* 6. Write CSR arrays (if present) */
    if (model->csr && row_count > 0 && edge_count > 0) {
        /* Row offsets: row_count + 1 elements */
        if (fwrite(model->csr->row_offsets, sizeof(uint32_t), row_count + 1, f) != row_count + 1) {
            result = PSAM_ERR_IO;
            goto cleanup;
        }

        /* Targets: edge_count elements */
        if (fwrite(model->csr->targets, sizeof(uint32_t), edge_count, f) != edge_count) {
            result = PSAM_ERR_IO;
            goto cleanup;
        }

        /* Weights: edge_count elements */
        if (fwrite(model->csr->weights, sizeof(int16_t), edge_count, f) != edge_count) {
            result = PSAM_ERR_IO;
            goto cleanup;
        }

        /* Row scales: row_count elements */
        if (fwrite(model->csr->row_scales, sizeof(float), row_count, f) != row_count) {
            result = PSAM_ERR_IO;
            goto cleanup;
        }
    }

    /* 7. Write bias array */
    if (fwrite(model->bias, sizeof(float), model->config.vocab_size, f) != model->config.vocab_size) {
        result = PSAM_ERR_IO;
        goto cleanup;
    }

    /* 8. Write unigram counts */
    if (fwrite(model->unigram_counts, sizeof(uint32_t), model->config.vocab_size, f) != model->config.vocab_size) {
        result = PSAM_ERR_IO;
        goto cleanup;
    }

    /* 9. Write row descriptor count and descriptors */
    if (fwrite(&model->row_descriptor_count, sizeof(uint32_t), 1, f) != 1) {
        result = PSAM_ERR_IO;
        goto cleanup;
    }

    if (model->row_descriptor_count > 0 && model->row_descriptors) {
        if (fwrite(model->row_descriptors, sizeof(row_descriptor_t), model->row_descriptor_count, f) != model->row_descriptor_count) {
            result = PSAM_ERR_IO;
            goto cleanup;
        }
    }

cleanup:
    fclose(f);
    return result;
}

psam_model_t* psam_load(const char* path) {
    if (!path) {
        return NULL;
    }

    FILE* f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }

    psam_model_t* model = NULL;
    csr_storage_t* csr = NULL;

    /* 1. Read and validate header */
    uint32_t magic, version;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 ||
        fread(&version, sizeof(uint32_t), 1, f) != 1) {
        goto error;
    }

    if (magic != PSAM_MAGIC) {
        goto error;  /* Invalid file format */
    }

    if (version < 1 || version > PSAM_VERSION) {
        goto error;  /* Unsupported version */
    }

    /* 2. Read provenance metadata (version >= 2) */
    psam_provenance_t provenance;
    memset(&provenance, 0, sizeof(provenance));

    if (version >= 2) {
        if (fread(&provenance.created_timestamp, sizeof(uint64_t), 1, f) != 1) {
            goto error;
        }

        char created_by_raw[PSAM_CREATED_BY_MAX];
        if (fread(created_by_raw, sizeof(char), PSAM_CREATED_BY_MAX, f) != PSAM_CREATED_BY_MAX) {
            goto error;
        }
        memcpy(provenance.created_by, created_by_raw, PSAM_CREATED_BY_MAX);
        provenance.created_by[PSAM_CREATED_BY_MAX - 1] = '\0';

        if (fread(provenance.source_hash, sizeof(uint8_t), PSAM_SOURCE_HASH_SIZE, f) != PSAM_SOURCE_HASH_SIZE) {
            goto error;
        }
    } else {
        provenance.created_timestamp = 0;
        strncpy(provenance.created_by, "legacy-psam", PSAM_CREATED_BY_MAX - 1);
        memset(provenance.source_hash, 0, PSAM_SOURCE_HASH_SIZE);
    }

    /* 3. Read config */
    psam_config_t config;
    memset(&config, 0, sizeof(config));

    if (fread(&config.vocab_size, sizeof(uint32_t), 1, f) != 1 ||
        fread(&config.window, sizeof(uint32_t), 1, f) != 1 ||
        fread(&config.top_k, sizeof(uint32_t), 1, f) != 1 ||
        fread(&config.alpha, sizeof(float), 1, f) != 1 ||
        fread(&config.min_evidence, sizeof(float), 1, f) != 1 ||
        fread(&config.enable_idf, sizeof(uint8_t), 1, f) != 1 ||
        fread(&config.enable_ppmi, sizeof(uint8_t), 1, f) != 1 ||
        fread(&config.edge_dropout, sizeof(float), 1, f) != 1) {
        goto error;
    }

    /* 4. Create model with loaded config */
    model = psam_create_with_config(&config);
    if (!model) {
        goto error;
    }

    /* Apply provenance */
    psam_lock_wrlock(&model->lock);
    model->provenance = provenance;
    psam_lock_unlock_wr(&model->lock);

    /* 5. Read model metadata */
    if (fread(&model->total_tokens, sizeof(uint64_t), 1, f) != 1) {
        goto error;
    }

    /* 6. Read CSR dimensions */
    uint32_t row_count;
    uint64_t edge_count;

    if (fread(&row_count, sizeof(uint32_t), 1, f) != 1 ||
        fread(&edge_count, sizeof(uint64_t), 1, f) != 1) {
        goto error;
    }

    /* 7. Read CSR arrays (if present) */
    if (row_count > 0 && edge_count > 0) {
        csr = calloc(1, sizeof(csr_storage_t));
        if (!csr) {
            goto error;
        }

        csr->row_count = row_count;
        csr->edge_count = edge_count;

        /* Allocate row_offsets: row_count + 1 */
        csr->row_offsets = malloc(sizeof(uint32_t) * (row_count + 1));
        if (!csr->row_offsets) {
            goto error;
        }

        if (fread(csr->row_offsets, sizeof(uint32_t), row_count + 1, f) != row_count + 1) {
            goto error;
        }

        /* Allocate targets */
        csr->targets = malloc(sizeof(uint32_t) * edge_count);
        if (!csr->targets) {
            goto error;
        }

        if (fread(csr->targets, sizeof(uint32_t), edge_count, f) != edge_count) {
            goto error;
        }

        /* Allocate weights */
        csr->weights = malloc(sizeof(int16_t) * edge_count);
        if (!csr->weights) {
            goto error;
        }

        if (fread(csr->weights, sizeof(int16_t), edge_count, f) != edge_count) {
            goto error;
        }

        /* Allocate row_scales */
        csr->row_scales = malloc(sizeof(float) * row_count);
        if (!csr->row_scales) {
            goto error;
        }

        if (fread(csr->row_scales, sizeof(float), row_count, f) != row_count) {
            goto error;
        }

        model->csr = csr;
        csr = NULL;  /* Ownership transferred to model */
    }

    /* 7. Read bias array */
    if (fread(model->bias, sizeof(float), config.vocab_size, f) != config.vocab_size) {
        goto error;
    }

    /* 8. Read unigram counts */
    if (fread(model->unigram_counts, sizeof(uint32_t), config.vocab_size, f) != config.vocab_size) {
        goto error;
    }

    /* 9. Read row descriptor count and descriptors */
    if (fread(&model->row_descriptor_count, sizeof(uint32_t), 1, f) != 1) {
        goto error;
    }

    if (model->row_descriptor_count > 0) {
        model->row_descriptors = malloc(sizeof(row_descriptor_t) * model->row_descriptor_count);
        if (!model->row_descriptors) {
            goto error;
        }

        if (fread(model->row_descriptors, sizeof(row_descriptor_t), model->row_descriptor_count, f) != model->row_descriptor_count) {
            goto error;
        }
    }

    /* 10. Mark model as finalized */
    model->is_finalized = true;

    fclose(f);
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
    fclose(f);
    return NULL;
}
