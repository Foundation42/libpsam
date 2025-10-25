/**
 * train.c - Training logic (co-occurrence counting, PPMI, IDF)
 *
 * Implements the sliding window co-occurrence training from TypeScript.
 */

#define _POSIX_C_SOURCE 200809L

#include "../psam_internal.h"
#include "training_hash.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EPSILON 1e-10f
#define INITIAL_EDGE_CAPACITY 16

/* ============================ Edge Structure ============================ */

typedef struct {
    uint32_t target;
    float weight;
    uint32_t count;
} edge_t;

/* ============================ Row Accumulator ============================  */

struct row_accumulator_t {
    uint32_t source;
    uint32_t offset;
    uint32_t total_observations;
    edge_hash_t* edges;  /* Changed from array to hash table */
};

static row_accumulator_t* row_accumulator_create(uint32_t source, uint32_t offset) {
    row_accumulator_t* row = calloc(1, sizeof(row_accumulator_t));
    if (!row) return NULL;

    row->source = source;
    row->offset = offset;
    row->total_observations = 0;
    row->edges = edge_hash_create(INITIAL_EDGE_CAPACITY);

    if (!row->edges) {
        free(row);
        return NULL;
    }

    return row;
}

static void row_accumulator_destroy(row_accumulator_t* row) {
    if (row) {
        edge_hash_destroy(row->edges);
        free(row);
    }
}

static edge_entry_t* row_accumulator_find_or_create_edge(row_accumulator_t* row, uint32_t target) {
    return edge_hash_find_or_create(row->edges, target);
}

/* ============================ Training Data ============================ */

typedef struct {
    row_hash_t* rows;  /* Changed from array to hash table */
} training_data_t;

static training_data_t* training_data_create(void) {
    training_data_t* data = calloc(1, sizeof(training_data_t));
    if (!data) return NULL;

    data->rows = row_hash_create(1024);  /* Start with decent size */

    if (!data->rows) {
        free(data);
        return NULL;
    }

    return data;
}

void psam_free_training_data(void* training_data) {
    if (!training_data) return;

    training_data_t* data = (training_data_t*)training_data;
    row_hash_destroy(data->rows, row_accumulator_destroy);
    free(data);
}

static row_accumulator_t* training_data_find_or_create_row(
    training_data_t* data,
    uint32_t source,
    uint32_t offset
) {
    /* Try to find existing row */
    row_accumulator_t* row = row_hash_find(data->rows, source, offset);
    if (row) {
        return row;
    }

    /* Create new row */
    row = row_accumulator_create(source, offset);
    if (!row) return NULL;

    /* Insert into hash table */
    if (row_hash_insert(data->rows, source, offset, row) != 0) {
        row_accumulator_destroy(row);
        return NULL;
    }

    return row;
}

/* ============================ Helper Functions ============================ */

static float compute_ppmi(
    uint32_t pair_count,
    uint32_t total_pairs,
    const psam_model_t* model,
    uint32_t source,
    uint32_t target
) {
    (void)total_pairs;
    // Match JavaScript: no +1 smoothing, just use actual counts
    uint32_t source_count = model->unigram_counts[source];
    uint32_t target_count = model->unigram_counts[target];

    // Fallback to 1 if zero (like JavaScript's || 1)
    if (source_count == 0) source_count = 1;
    if (target_count == 0) target_count = 1;

    float total = fmaxf((float)model->total_tokens, 1.0f);
    float p_source = (float)source_count / total;
    float p_target = (float)target_count / total;
    float p_pair = (float)pair_count / total;

    if (p_pair <= 0.0f || p_source <= 0.0f || p_target <= 0.0f) {
        return 0.0f;
    }

    float ratio = p_pair / (p_source * p_target);
    return fmaxf(0.0f, logf(ratio));
}

static float compute_idf_for_training(const psam_model_t* model, uint32_t token) {
    uint32_t occurrences = model->unigram_counts[token];
    if (occurrences == 0) occurrences = 1;

    return logf((1.0f + (float)model->total_tokens) / (1.0f + (float)occurrences)) + 1.0f;
}

/* ============================ Training API ============================ */

psam_error_t psam_train_token(psam_model_t* model, uint32_t token) {
    if (!model) {
        return PSAM_ERR_NULL_PARAM;
    }

    if (token >= model->config.vocab_size) {
        return PSAM_ERR_INVALID_CONFIG;
    }

    /* Increment unigram count */
    model->total_tokens++;
    model->unigram_counts[token]++;

    /* TODO: Need to buffer tokens and process windows
     * For now, this is a placeholder that just tracks unigrams
     */

    return PSAM_OK;
}

psam_error_t psam_train_batch(psam_model_t* model, const uint32_t* tokens, size_t num_tokens) {
    if (!model || !tokens) {
        return PSAM_ERR_NULL_PARAM;
    }

    if (model->is_finalized) {
        return PSAM_ERR_INVALID_CONFIG;  /* Cannot train after finalization */
    }

    /* Ensure training data exists */
    if (!model->training_data) {
        model->training_data = training_data_create();
        if (!model->training_data) {
            return PSAM_ERR_OUT_OF_MEMORY;
        }
    }

    training_data_t* data = (training_data_t*)model->training_data;
    const uint32_t window = model->config.window;

    /* Process each token */
    for (size_t j = 0; j < num_tokens; j++) {
        uint32_t target = tokens[j];
        if (target >= model->config.vocab_size) {
            continue;  /* Skip out-of-vocab tokens */
        }

        /* Increment unigram */
        model->total_tokens++;
        model->unigram_counts[target]++;

        /* Process window before this token */
        size_t start = j > window ? j - window : 0;
        for (size_t i = start; i < j; i++) {
            uint32_t source = tokens[i];
            if (source >= model->config.vocab_size) {
                continue;
            }

            uint32_t offset = (uint32_t)(j - i);

            /* Find or create row for (source, offset) */
            row_accumulator_t* row = training_data_find_or_create_row(data, source, offset);
            if (!row) {
                return PSAM_ERR_OUT_OF_MEMORY;
            }

            row->total_observations++;

            /* Find or create edge for target */
            edge_entry_t* edge = row_accumulator_find_or_create_edge(row, target);
            if (!edge) {
                return PSAM_ERR_OUT_OF_MEMORY;
            }

            edge->count++;

            /* Compute weight using PPMI, IDF, distance decay */
            if (edge->count < (uint32_t)model->config.min_evidence) {
                edge->weight = 0.0f;
                continue;
            }

            float ppmi = model->config.enable_ppmi
                ? compute_ppmi(edge->count, model->total_tokens, model, source, target)
                : 1.0f;

            float idf = model->config.enable_idf
                ? compute_idf_for_training(model, source)
                : 1.0f;

            float distance_decay = expf(-model->config.alpha * (float)offset);
            edge->weight = ppmi * idf * distance_decay;
        }
    }

    return PSAM_OK;
}

/* ============================ Comparator for Sorting ============================ */

static int compare_rows(const void* a, const void* b) {
    const row_accumulator_t* row_a = *(const row_accumulator_t**)a;
    const row_accumulator_t* row_b = *(const row_accumulator_t**)b;

    if (row_a->source != row_b->source) {
        return (int)row_a->source - (int)row_b->source;
    }
    return (int)row_a->offset - (int)row_b->offset;
}

static int compare_edges(const void* a, const void* b) {
    const edge_t* edge_a = (const edge_t*)a;
    const edge_t* edge_b = (const edge_t*)b;

    /* Sort by weight descending, then by target ascending */
    if (edge_b->weight != edge_a->weight) {
        if (edge_b->weight > edge_a->weight) return 1;
        if (edge_b->weight < edge_a->weight) return -1;
    }
    return (int)edge_a->target - (int)edge_b->target;
}

psam_error_t psam_finalize_training(psam_model_t* model) {
    if (!model) {
        return PSAM_ERR_NULL_PARAM;
    }

    if (model->is_finalized) {
        return PSAM_OK;  /* Already finalized */
    }

    if (!model->training_data) {
        /* No training data - mark as finalized with empty CSR */
        model->is_finalized = true;
        return PSAM_OK;
    }

    training_data_t* data = (training_data_t*)model->training_data;

    /* 1. Collect all rows into an array for sorting */
    uint32_t total_row_count = data->rows->entry_count;
    row_accumulator_t** row_array = malloc(sizeof(row_accumulator_t*) * total_row_count);
    if (!row_array) {
        return PSAM_ERR_OUT_OF_MEMORY;
    }

    row_iterator_t it;
    row_iterator_init(&it, data->rows);
    uint32_t row_idx = 0;
    row_accumulator_t* row;
    while ((row = row_iterator_next(&it)) != NULL) {
        row_array[row_idx++] = row;
    }

    /* Sort rows by (source, offset) */
    qsort(row_array, total_row_count, sizeof(row_accumulator_t*), compare_rows);

    /* 2. Count total edges (respecting min_evidence) */
    uint64_t total_edges = 0;
    uint32_t valid_row_count = 0;

    for (uint32_t i = 0; i < total_row_count; i++) {
        row = row_array[i];
        uint32_t valid_edges = 0;

        edge_iterator_t edge_it;
        edge_iterator_init(&edge_it, row->edges);
        edge_entry_t* edge;
        while ((edge = edge_iterator_next(&edge_it)) != NULL) {
            if (edge->count >= (uint32_t)model->config.min_evidence) {
                valid_edges++;
            }
        }

        if (valid_edges > 0) {
            total_edges += valid_edges;
            valid_row_count++;
        }
    }

    if (valid_row_count == 0 || total_edges == 0) {
        /* No valid data - mark as finalized */
        free(row_array);
        model->is_finalized = true;
        psam_free_training_data(model->training_data);
        model->training_data = NULL;
        return PSAM_OK;
    }

    /* 3. Allocate CSR arrays */
    csr_storage_t* csr = calloc(1, sizeof(csr_storage_t));
    if (!csr) {
        return PSAM_ERR_OUT_OF_MEMORY;
    }

    csr->row_count = valid_row_count;
    csr->edge_count = total_edges;

    csr->row_offsets = malloc(sizeof(uint32_t) * (valid_row_count + 1));
    csr->targets = malloc(sizeof(uint32_t) * total_edges);
    csr->weights = malloc(sizeof(int16_t) * total_edges);
    csr->row_scales = malloc(sizeof(float) * valid_row_count);

    if (!csr->row_offsets || !csr->targets || !csr->weights || !csr->row_scales) {
        free(csr->row_offsets);
        free(csr->targets);
        free(csr->weights);
        free(csr->row_scales);
        free(csr);
        return PSAM_ERR_OUT_OF_MEMORY;
    }

    /* Allocate row descriptors */
    model->row_descriptors = malloc(sizeof(row_descriptor_t) * valid_row_count);
    if (!model->row_descriptors) {
        free(csr->row_offsets);
        free(csr->targets);
        free(csr->weights);
        free(csr->row_scales);
        free(csr);
        return PSAM_ERR_OUT_OF_MEMORY;
    }
    model->row_descriptor_count = valid_row_count;

    /* 4. Build CSR structure */
    uint32_t cursor = 0;
    uint32_t row_index = 0;

    for (uint32_t i = 0; i < total_row_count; i++) {
        row = row_array[i];

        /* Filter and collect edges from hash table */
        uint32_t edge_capacity = row->edges->entry_count;
        edge_t* valid_edges = malloc(sizeof(edge_t) * edge_capacity);
        uint32_t valid_count = 0;

        edge_iterator_t edge_it;
        edge_iterator_init(&edge_it, row->edges);
        edge_entry_t* edge_entry;
        while ((edge_entry = edge_iterator_next(&edge_it)) != NULL) {
            if (edge_entry->count >= (uint32_t)model->config.min_evidence) {
                valid_edges[valid_count].target = edge_entry->target;
                valid_edges[valid_count].weight = edge_entry->weight;
                valid_edges[valid_count].count = edge_entry->count;
                valid_count++;
            }
        }

        if (valid_count == 0) {
            free(valid_edges);
            continue;
        }

        qsort(valid_edges, valid_count, sizeof(edge_t), compare_edges);

        /* Find max weight for quantization */
        float max_weight = 0.0f;
        for (uint32_t e = 0; e < valid_count; e++) {
            if (valid_edges[e].weight > max_weight) {
                max_weight = valid_edges[e].weight;
            }
        }

        float scale = max_weight > 0.0f ? max_weight / 32767.0f : 1.0f;

        /* Store row metadata */
        csr->row_offsets[row_index] = cursor;
        csr->row_scales[row_index] = scale;

        model->row_descriptors[row_index].source = row->source;
        model->row_descriptors[row_index].offset = row->offset;
        model->row_descriptors[row_index].totalObservations = row->total_observations;

        /* Store edges */
        for (uint32_t e = 0; e < valid_count; e++) {
            csr->targets[cursor] = valid_edges[e].target;
            csr->weights[cursor] = (int16_t)(valid_edges[e].weight / scale);
            cursor++;
        }

        free(valid_edges);
        row_index++;
    }

    csr->row_offsets[valid_row_count] = cursor;

    /* 5. Recompute bias */
    float denominator = fmaxf((float)model->total_tokens, 1.0f);
    for (uint32_t i = 0; i < model->config.vocab_size; i++) {
        uint32_t count = model->unigram_counts[i];
        model->bias[i] = logf((float)(count + 1) / denominator);
    }

    /* 6. Install CSR and mark as finalized */
    model->csr = csr;
    model->is_finalized = true;

    /* Free training data */
    free(row_array);
    psam_free_training_data(model->training_data);
    model->training_data = NULL;

    return PSAM_OK;
}
