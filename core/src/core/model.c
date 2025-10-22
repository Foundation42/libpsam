/**
 * model.c - PSAM model lifecycle and core data structures
 */

#define _POSIX_C_SOURCE 200809L

#include "../psam_internal.h"
#include <stdlib.h>
#include <string.h>

/* ============================ Default Configuration ============================ */

static const psam_config_t DEFAULT_CONFIG = {
    .vocab_size = 0,
    .window = 8,
    .top_k = 32,
    .alpha = 0.1f,
    .min_evidence = 1.0f,
    .enable_idf = true,
    .enable_ppmi = true,
    .edge_dropout = 0.0f
};

/* ============================ Lifecycle Functions ============================ */

psam_model_t* psam_create(uint32_t vocab_size, uint32_t window, uint32_t top_k) {
    psam_config_t config = DEFAULT_CONFIG;
    config.vocab_size = vocab_size;
    config.window = window;
    config.top_k = top_k;
    return psam_create_with_config(&config);
}

psam_model_t* psam_create_with_config(const psam_config_t* config) {
    if (!config || config->vocab_size == 0) {
        return NULL;
    }

    psam_model_t* model = calloc(1, sizeof(psam_model_t));
    if (!model) {
        return NULL;
    }

    /* Copy configuration */
    model->config = *config;

    /* Allocate bias array */
    model->bias = calloc(config->vocab_size, sizeof(float));
    if (!model->bias) {
        free(model);
        return NULL;
    }

    /* Allocate unigram counts */
    model->unigram_counts = calloc(config->vocab_size, sizeof(uint32_t));
    if (!model->unigram_counts) {
        free(model->bias);
        free(model);
        return NULL;
    }

    /* Initialize thread safety */
    psam_lock_init(&model->lock);

    model->is_finalized = false;
    model->total_tokens = 0;
    model->csr = NULL;
    model->training_data = NULL;
    model->layers = NULL;

    return model;
}

void psam_destroy(psam_model_t* model) {
    if (!model) {
        return;
    }

    psam_lock_destroy(&model->lock);

    /* Free CSR storage */
    if (model->csr) {
        free(model->csr->row_offsets);
        free(model->csr->targets);
        free(model->csr->weights);
        free(model->csr->row_scales);
        free(model->csr);
    }

    /* Free row descriptors */
    free(model->row_descriptors);

    /* Free bias and counts */
    free(model->bias);
    free(model->unigram_counts);

    /* Free training data (implemented in train.c) */
    extern void psam_free_training_data(void* training_data);
    if (model->training_data) {
        psam_free_training_data(model->training_data);
    }

    /* Free layer list (but not the overlay models - caller owns them) */
    layer_node_t* layer = model->layers;
    while (layer) {
        layer_node_t* next = layer->next;
        free(layer);
        layer = next;
    }

    free(model);
}

/* ============================ Introspection ============================ */

psam_error_t psam_get_stats(const psam_model_t* model, psam_stats_t* out_stats) {
    if (!model || !out_stats) {
        return PSAM_ERR_NULL_PARAM;
    }

    psam_lock_rdlock((psam_lock_t*)&model->lock);

    memset(out_stats, 0, sizeof(psam_stats_t));
    out_stats->vocab_size = model->config.vocab_size;
    out_stats->total_tokens = model->total_tokens;

    if (model->csr) {
        out_stats->row_count = model->csr->row_count;
        out_stats->edge_count = model->csr->edge_count;

        /* Estimate memory usage */
        out_stats->memory_bytes = sizeof(psam_model_t);
        out_stats->memory_bytes += model->config.vocab_size * sizeof(float);  /* bias */
        out_stats->memory_bytes += model->config.vocab_size * sizeof(uint32_t); /* unigram_counts */
        out_stats->memory_bytes += (model->csr->row_count + 1) * sizeof(uint32_t); /* row_offsets */
        out_stats->memory_bytes += model->csr->edge_count * sizeof(uint32_t); /* targets */
        out_stats->memory_bytes += model->csr->edge_count * sizeof(int16_t); /* weights */
        out_stats->memory_bytes += model->csr->row_count * sizeof(float); /* row_scales */
    }

    psam_lock_unlock_rd((psam_lock_t*)&model->lock);

    return PSAM_OK;
}

const char* psam_error_string(psam_error_t error) {
    switch (error) {
        case PSAM_OK: return "Success";
        case PSAM_ERR_NULL_PARAM: return "Null parameter provided";
        case PSAM_ERR_INVALID_CONFIG: return "Invalid configuration";
        case PSAM_ERR_OUT_OF_MEMORY: return "Out of memory";
        case PSAM_ERR_IO: return "I/O error";
        case PSAM_ERR_INVALID_MODEL: return "Invalid or corrupted model";
        case PSAM_ERR_NOT_TRAINED: return "Model not finalized for inference";
        case PSAM_ERR_LAYER_NOT_FOUND: return "Layer ID not found";
        default: return "Unknown error";
    }
}

const char* psam_version(void) {
    return "1.0.0-alpha";
}
