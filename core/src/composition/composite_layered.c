/**
 * composite_layered.c - Runtime layered composites for libpsam
 */

#define _POSIX_C_SOURCE 200809L

#include "../psam_internal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    char id[PSAM_LAYER_ID_MAX];
    psam_model_t* model;
    float weight;
} layered_entry_t;

struct psam_composite {
    psam_composite_topology_t topology;
    psam_model_t* base;
    float base_weight;
    layered_entry_t* layers;
    size_t layer_count;
    size_t layer_capacity;
};

typedef struct {
    uint32_t token;
    float score;
} composite_score_t;

static bool validate_layer_compat(psam_model_t* base, psam_model_t* layer) {
    if (!base || !layer) {
        return false;
    }
    if (!layer->is_finalized) {
        return false;
    }
    return base->config.vocab_size == layer->config.vocab_size;
}

static layered_entry_t* find_layer(layered_entry_t* layers, size_t count, const char* id) {
    if (!layers || !id) {
        return NULL;
    }
    for (size_t i = 0; i < count; ++i) {
        if (strncmp(layers[i].id, id, PSAM_LAYER_ID_MAX) == 0) {
            return &layers[i];
        }
    }
    return NULL;
}

psam_composite_t* psam_create_layered(psam_model_t* base_model) {
    if (!base_model || !base_model->is_finalized) {
        return NULL;
    }

    psam_composite_t* composite = calloc(1, sizeof(psam_composite_t));
    if (!composite) {
        return NULL;
    }

    composite->topology = PSAM_COMPOSITE_LAYERED;
    composite->base = base_model;
    composite->base_weight = 1.0f;
    composite->layers = NULL;
    composite->layer_capacity = 0;
    composite->layer_count = 0;

    return composite;
}

void psam_composite_destroy(psam_composite_t* composite) {
    if (!composite) {
        return;
    }
    free(composite->layers);
    free(composite);
}

psam_error_t psam_composite_set_base_weight(psam_composite_t* composite, float weight) {
    if (!composite) {
        return PSAM_ERR_NULL_PARAM;
    }
    composite->base_weight = weight;
    return PSAM_OK;
}

psam_error_t psam_composite_add_layer(
    psam_composite_t* composite,
    const char* layer_id,
    psam_model_t* layer_model,
    float weight
) {
    if (!composite || !layer_id || !layer_model) {
        return PSAM_ERR_NULL_PARAM;
    }

    if (composite->topology != PSAM_COMPOSITE_LAYERED) {
        return PSAM_ERR_INVALID_CONFIG;
    }

    if (!validate_layer_compat(composite->base, layer_model)) {
        return PSAM_ERR_INVALID_CONFIG;
    }

    if (find_layer(composite->layers, composite->layer_count, layer_id)) {
        return PSAM_ERR_INVALID_CONFIG;
    }

    if (composite->layer_count == composite->layer_capacity) {
        size_t new_capacity = composite->layer_capacity ? composite->layer_capacity * 2 : 4;
        layered_entry_t* resized = realloc(composite->layers, new_capacity * sizeof(layered_entry_t));
        if (!resized) {
            return PSAM_ERR_OUT_OF_MEMORY;
        }
        composite->layers = resized;
        composite->layer_capacity = new_capacity;
    }

    layered_entry_t* entry = &composite->layers[composite->layer_count++];
    memset(entry, 0, sizeof(*entry));
    strncpy(entry->id, layer_id, PSAM_LAYER_ID_MAX - 1);
    entry->model = layer_model;
    entry->weight = weight;

    return PSAM_OK;
}

psam_error_t psam_composite_remove_layer(psam_composite_t* composite, const char* layer_id) {
    if (!composite || !layer_id) {
        return PSAM_ERR_NULL_PARAM;
    }

    for (size_t i = 0; i < composite->layer_count; ++i) {
        if (strncmp(composite->layers[i].id, layer_id, PSAM_LAYER_ID_MAX) == 0) {
            if (i + 1 < composite->layer_count) {
                memmove(
                    &composite->layers[i],
                    &composite->layers[i + 1],
                    (composite->layer_count - i - 1) * sizeof(layered_entry_t)
                );
            }
            composite->layer_count--;
            return PSAM_OK;
        }
    }

    return PSAM_ERR_LAYER_NOT_FOUND;
}

psam_error_t psam_composite_update_layer_weight(
    psam_composite_t* composite,
    const char* layer_id,
    float new_weight
) {
    if (!composite || !layer_id) {
        return PSAM_ERR_NULL_PARAM;
    }

    layered_entry_t* entry = find_layer(composite->layers, composite->layer_count, layer_id);
    if (!entry) {
        return PSAM_ERR_LAYER_NOT_FOUND;
    }

    entry->weight = new_weight;
    return PSAM_OK;
}

int psam_composite_list_layers(
    const psam_composite_t* composite,
    psam_composite_layer_info_t* out_layers,
    size_t max_layers
) {
    if (!composite || (max_layers > 0 && !out_layers)) {
        return PSAM_ERR_NULL_PARAM;
    }

    size_t count = composite->layer_count < max_layers ? composite->layer_count : max_layers;
    for (size_t i = 0; i < count; ++i) {
        snprintf(
            out_layers[i].id,
            PSAM_LAYER_ID_MAX,
            "%s",
            composite->layers[i].id
        );
        out_layers[i].weight = composite->layers[i].weight;
    }

    return (int)count;
}

static int compare_scores_desc(const void* a, const void* b) {
    const composite_score_t* sa = (const composite_score_t*)a;
    const composite_score_t* sb = (const composite_score_t*)b;
    if (sa->score > sb->score) return -1;
    if (sa->score < sb->score) return 1;
    return 0;
}

static void accumulate_scores(
    composite_score_t* accum,
    size_t* accum_size,
    size_t capacity,
    const psam_prediction_t* preds,
    size_t pred_count,
    float weight
) {
    for (size_t i = 0; i < pred_count; ++i) {
        uint32_t token = preds[i].token;
        float contribution = preds[i].score * weight;
        if (contribution == 0.0f) {
            continue;
        }
        bool found = false;

        for (size_t j = 0; j < *accum_size; ++j) {
            if (accum[j].token == token) {
                accum[j].score += contribution;
                found = true;
                break;
            }
        }

        if (!found && *accum_size < capacity) {
            accum[*accum_size].token = token;
            accum[*accum_size].score = contribution;
            (*accum_size)++;
        }
    }
}

int psam_composite_predict(
    psam_composite_t* composite,
    const uint32_t* context,
    size_t context_len,
    psam_prediction_t* out_preds,
    size_t max_preds
) {
    if (!composite || !context || !out_preds) {
        return PSAM_ERR_NULL_PARAM;
    }

    if (!composite->base) {
        return PSAM_ERR_INVALID_CONFIG;
    }

    if (max_preds == 0) {
        return 0;
    }

    size_t capacity = (composite->layer_count + 1) * max_preds;
    if (capacity == 0) {
        return PSAM_ERR_INVALID_CONFIG;
    }

    composite_score_t* accum = calloc(capacity, sizeof(composite_score_t));
    if (!accum) {
        return PSAM_ERR_OUT_OF_MEMORY;
    }

    psam_prediction_t* scratch = malloc(max_preds * sizeof(psam_prediction_t));
    if (!scratch) {
        free(accum);
        return PSAM_ERR_OUT_OF_MEMORY;
    }

    size_t accum_size = 0;
    int result = 0;

    int count = psam_predict(composite->base, context, context_len, scratch, max_preds);
    if (count < 0) {
        result = count;
        goto cleanup;
    }

    accumulate_scores(accum, &accum_size, capacity, scratch, (size_t)count, composite->base_weight);

    for (size_t i = 0; i < composite->layer_count; ++i) {
        layered_entry_t* entry = &composite->layers[i];
        count = psam_predict(entry->model, context, context_len, scratch, max_preds);
        if (count < 0) {
            result = count;
            goto cleanup;
        }
        accumulate_scores(accum, &accum_size, capacity, scratch, (size_t)count, entry->weight);
    }

    if (accum_size == 0) {
        result = 0;
        goto cleanup;
    }

    qsort(accum, accum_size, sizeof(composite_score_t), compare_scores_desc);

    size_t to_copy = accum_size < max_preds ? accum_size : max_preds;
    for (size_t i = 0; i < to_copy; ++i) {
        out_preds[i].token = accum[i].token;
        out_preds[i].score = accum[i].score;
        out_preds[i].calibrated_prob = 0.0f;
    }

    result = (int)to_copy;

cleanup:
    free(scratch);
    free(accum);
    return result;
}
