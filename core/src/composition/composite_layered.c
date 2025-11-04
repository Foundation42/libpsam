/**
 * composite_layered.c - Runtime layered composites for libpsam
 */

#define _POSIX_C_SOURCE 200809L

#include "../psam_internal.h"
#include "../../include/psam_composite.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

typedef struct {
    char id[PSAM_LAYER_ID_MAX];
    psam_model_t* model;
    float weight;
    float bias;
    bool owns_model;
} layered_entry_t;

struct psam_composite {
    psam_composite_topology_t topology;
    psam_model_t* base;
    float base_weight;
    layered_entry_t* layers;
    size_t layer_count;
    size_t layer_capacity;
    bool owns_base;
    psam_sampler_t sampler_defaults;  /* Default sampler config from .psamc */
};

static psam_error_t composite_add_layer_internal(
    psam_composite_t* composite,
    const char* layer_id,
    psam_model_t* layer_model,
    float weight,
    bool take_ownership
);

typedef struct {
    uint32_t token;
    float score;
    float raw_strength;
    uint32_t support_count;
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
    composite->owns_base = false;

    /* Initialize sampler defaults */
    composite->sampler_defaults = (psam_sampler_t){
        .transform = PSAM_LOGIT_ZSCORE,
        .temperature = 1.0f,
        .top_k = 50,
        .top_p = 0.95f,
        .seed = 42
    };

    return composite;
}

void psam_composite_destroy(psam_composite_t* composite) {
    if (!composite) {
        return;
    }
    if (composite->owns_base && composite->base) {
        psam_destroy(composite->base);
        composite->base = NULL;
    }
    if (composite->layers) {
        for (size_t i = 0; i < composite->layer_count; ++i) {
            if (composite->layers[i].owns_model && composite->layers[i].model) {
                psam_destroy(composite->layers[i].model);
            }
        }
        free(composite->layers);
    }
    free(composite);
}

psam_error_t psam_composite_set_base_weight(psam_composite_t* composite, float weight) {
    if (!composite) {
        return PSAM_ERR_NULL_PARAM;
    }
    composite->base_weight = weight;
    return PSAM_OK;
}

static psam_error_t composite_add_layer_internal(
    psam_composite_t* composite,
    const char* layer_id,
    psam_model_t* layer_model,
    float weight,
    bool take_ownership
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
    entry->owns_model = take_ownership;

    return PSAM_OK;
}

psam_error_t psam_composite_add_layer(
    psam_composite_t* composite,
    const char* layer_id,
    psam_model_t* layer_model,
    float weight
) {
    return composite_add_layer_internal(composite, layer_id, layer_model, weight, false);
}

psam_error_t psam_composite_remove_layer(psam_composite_t* composite, const char* layer_id) {
    if (!composite || !layer_id) {
        return PSAM_ERR_NULL_PARAM;
    }

    for (size_t i = 0; i < composite->layer_count; ++i) {
        if (strncmp(composite->layers[i].id, layer_id, PSAM_LAYER_ID_MAX) == 0) {
            if (composite->layers[i].owns_model && composite->layers[i].model) {
                psam_destroy(composite->layers[i].model);
            }
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
        out_layers[i].bias = composite->layers[i].bias;
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

static bool is_absolute_path(const char* path) {
    if (!path || path[0] == '\0') {
        return false;
    }
#ifdef _WIN32
    if (path[0] == '/' || path[0] == '\\') {
        return true;
    }
    if (isalpha((unsigned char)path[0]) && path[1] == ':' && (path[2] == '/' || path[2] == '\\')) {
        return true;
    }
#else
    if (path[0] == '/') {
        return true;
    }
#endif
    return false;
}

static char* resolve_reference_path(const char* composite_path, const char* ref_path) {
    if (!ref_path) {
        return NULL;
    }
    if (is_absolute_path(ref_path)) {
        return strdup(ref_path);
    }

    if (!composite_path) {
        return strdup(ref_path);
    }

    const char* last_slash = strrchr(composite_path, '/');
#ifdef _WIN32
    const char* last_backslash = strrchr(composite_path, '\\');
    if (!last_slash || (last_backslash && last_backslash > last_slash)) {
        last_slash = last_backslash;
    }
#endif
    size_t dir_len = last_slash ? (size_t)(last_slash - composite_path + 1) : 0;
    size_t ref_len = strlen(ref_path);
    char* resolved = malloc(dir_len + ref_len + 1);
    if (!resolved) {
        return NULL;
    }
    if (dir_len > 0) {
        memcpy(resolved, composite_path, dir_len);
    }
    memcpy(resolved + dir_len, ref_path, ref_len + 1);
    return resolved;
}

static psam_model_t* load_model_from_ref(const char* composite_path, const psamc_model_ref_t* ref) {
    if (!ref) {
        return NULL;
    }
    char* resolved = resolve_reference_path(composite_path, ref->url);
    const char* path_to_load = resolved ? resolved : ref->url;
    fprintf(stderr, "DEBUG: load_model_from_ref - composite_path='%s', ref->url='%s', resolved='%s'\n",
            composite_path, ref->url, resolved ? resolved : "NULL");
    psam_model_t* model = psam_load(path_to_load);
    if (!model) {
        fprintf(stderr, "DEBUG: load_model_from_ref - psam_load('%s') failed\n", path_to_load);
    }
    free(resolved);
    return model;
}

psam_composite_t* psam_composite_load_file(const char* path, bool verify_integrity) {
    if (!path) {
        fprintf(stderr, "DEBUG: psam_composite_load_file - path is NULL\n");
        return NULL;
    }

    fprintf(stderr, "DEBUG: psam_composite_load_file - loading '%s' with verify=%d\n", path, verify_integrity);
    psamc_composite_t* spec = psamc_load(path, verify_integrity);
    if (!spec) {
        fprintf(stderr, "DEBUG: psam_composite_load_file - psamc_load failed\n");
        return NULL;
    }

    fprintf(stderr, "DEBUG: psam_composite_load_file - psamc_load succeeded, refs=%u\n", spec->manifest.num_references);
    if (spec->manifest.num_references == 0) {
        fprintf(stderr, "DEBUG: psam_composite_load_file - no references\n");
        psamc_free(spec);
        return NULL;
    }

    if (spec->topology.base_ref_index >= spec->manifest.num_references) {
        spec->topology.base_ref_index = 0;
    }

    fprintf(stderr, "DEBUG: psam_composite_load_file - loading base model from ref_index=%u, url='%s'\n",
            spec->topology.base_ref_index, spec->manifest.refs[spec->topology.base_ref_index].url);
    psam_model_t* base_model = load_model_from_ref(path, &spec->manifest.refs[spec->topology.base_ref_index]);
    if (!base_model) {
        fprintf(stderr, "DEBUG: psam_composite_load_file - load_model_from_ref failed for base\n");
        psamc_free(spec);
        return NULL;
    }
    fprintf(stderr, "DEBUG: psam_composite_load_file - base model loaded successfully\n");

    psam_composite_t* composite = psam_create_layered(base_model);
    if (!composite) {
        fprintf(stderr, "DEBUG: psam_composite_load_file - psam_create_layered failed\n");
        psam_destroy(base_model);
        psamc_free(spec);
        return NULL;
    }
    composite->owns_base = true;
    psam_composite_set_base_weight(composite, spec->topology.base_weight);

    fprintf(stderr, "DEBUG: psam_composite_load_file - loading %u overlay layers\n", spec->topology.layer_count);
    for (uint32_t i = 0; i < spec->topology.layer_count; ++i) {
        const psamc_layer_entry_t* entry = &spec->topology.layers[i];
        if (entry->ref_index >= spec->manifest.num_references) {
            continue;
        }
        fprintf(stderr, "DEBUG: loading layer %u, ref_index=%u, url='%s'\n",
                i, entry->ref_index, spec->manifest.refs[entry->ref_index].url);
        psam_model_t* overlay = load_model_from_ref(path, &spec->manifest.refs[entry->ref_index]);
        if (!overlay) {
            fprintf(stderr, "DEBUG: failed to load overlay layer %u\n", i);
            psam_composite_destroy(composite);
            psamc_free(spec);
            return NULL;
        }
        char fallback_id[PSAM_LAYER_ID_MAX];
        const char* layer_id = entry->layer_id[0] != '\0'
            ? entry->layer_id
            : (snprintf(fallback_id, sizeof(fallback_id), "layer-%u", i), fallback_id);
        fprintf(stderr, "DEBUG: adding layer '%s' with weight=%.3f\n", layer_id, entry->weight);
        psam_error_t err = composite_add_layer_internal(
            composite,
            layer_id,
            overlay,
            entry->weight,
            true
        );
        if (err != PSAM_OK) {
            fprintf(stderr, "DEBUG: composite_add_layer_internal failed with err=%d\n", err);
            psam_destroy(overlay);
            psam_composite_destroy(composite);
            psamc_free(spec);
            return NULL;
        }
        fprintf(stderr, "DEBUG: layer %u added successfully\n", i);
        psam_composite_update_layer_bias(composite, layer_id, entry->bias);
    }

    psam_sampler_t sampler_defaults = {
        .transform = spec->sampler_defaults.logit_transform,
        .temperature = spec->sampler_defaults.temperature,
        .top_k = spec->sampler_defaults.top_k,
        .top_p = spec->sampler_defaults.top_p,
        .seed = spec->sampler_defaults.seed
    };
    psam_composite_set_sampler_defaults(composite, &sampler_defaults);

    fprintf(stderr, "DEBUG: psam_composite_load_file - all layers loaded, returning composite\n");
    psamc_free(spec);
    return composite;
}

static void accumulate_scores(
    composite_score_t* accum,
    size_t* accum_size,
    size_t capacity,
    const psam_prediction_t* preds,
    size_t pred_count,
    float weight,
    float bias
) {
    for (size_t i = 0; i < pred_count; ++i) {
        uint32_t token = preds[i].token;
        float contribution = preds[i].score * weight + bias;
        float raw_component = preds[i].raw_strength * weight;
        uint32_t support_component = preds[i].support_count;
        if (contribution == 0.0f && bias == 0.0f) {
            continue;
        }
        bool found = false;

        for (size_t j = 0; j < *accum_size; ++j) {
            if (accum[j].token == token) {
                accum[j].score += contribution;
                accum[j].raw_strength += raw_component;
                if (support_component > 0) {
                    uint64_t updated = (uint64_t)accum[j].support_count + support_component;
                    accum[j].support_count = (updated > UINT32_MAX) ? UINT32_MAX : (uint32_t)updated;
                }
                found = true;
                break;
            }
        }

        if (!found && *accum_size < capacity) {
            accum[*accum_size].token = token;
            accum[*accum_size].score = contribution;
            accum[*accum_size].raw_strength = raw_component;
            accum[*accum_size].support_count = support_component;
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

    accumulate_scores(accum, &accum_size, capacity, scratch, (size_t)count, composite->base_weight, 0.0f);

    for (size_t i = 0; i < composite->layer_count; ++i) {
        layered_entry_t* entry = &composite->layers[i];
        count = psam_predict(entry->model, context, context_len, scratch, max_preds);
        if (count < 0) {
            result = count;
            goto cleanup;
        }
        accumulate_scores(accum, &accum_size, capacity, scratch, (size_t)count, entry->weight, entry->bias);
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
        out_preds[i].raw_strength = accum[i].raw_strength;
        uint32_t supports = accum[i].support_count;
        out_preds[i].support_count = (supports > UINT16_MAX) ? UINT16_MAX : (uint16_t)supports;
        out_preds[i]._reserved = 0;
        out_preds[i].calibrated_prob = 0.0f;
    }

    result = (int)to_copy;

cleanup:
    free(scratch);
    free(accum);
    return result;
}

psam_error_t psam_composite_update_layer_bias(
    psam_composite_t* composite,
    const char* layer_id,
    float new_bias
) {
    if (!composite || !layer_id) {
        return PSAM_ERR_NULL_PARAM;
    }

    layered_entry_t* layer = find_layer(composite->layers, composite->layer_count, layer_id);
    if (!layer) {
        return PSAM_ERR_LAYER_NOT_FOUND;
    }

    layer->bias = new_bias;
    return PSAM_OK;
}

/* Forward declarations from infer.c - these are internal sampler helpers */
extern void apply_logit_transform(float* logits, const float* scores, size_t n, psam_logit_transform_t transform);
extern void apply_temperature_and_softmax(float* probs, float* logits, size_t n, float temperature);

int psam_composite_predict_with_sampler(
    psam_composite_t* composite,
    const uint32_t* context,
    size_t context_len,
    const psam_sampler_t* sampler,
    psam_prediction_t* out_preds,
    size_t max_preds
) {
    if (!composite || !context || !out_preds) {
        return PSAM_ERR_NULL_PARAM;
    }

    /* Use composite's default sampler if none provided */
    if (!sampler) {
        sampler = &composite->sampler_defaults;
    }

    /* First get raw predictions using existing composite predict */
    int count = psam_composite_predict(composite, context, context_len, out_preds, max_preds);
    if (count <= 0) {
        return count;
    }

    size_t n = (size_t)count;

    /* Extract scores and apply sampler transforms */
    float* scores = malloc(n * sizeof(float));
    float* logits = malloc(n * sizeof(float));
    float* probs = malloc(n * sizeof(float));
    if (!scores || !logits || !probs) {
        free(scores);
        free(logits);
        free(probs);
        return PSAM_ERR_OUT_OF_MEMORY;
    }

    for (size_t i = 0; i < n; ++i) {
        scores[i] = out_preds[i].score;
    }

    /* Apply logit transform */
    apply_logit_transform(logits, scores, n, sampler->transform);

    /* Apply temperature and softmax */
    apply_temperature_and_softmax(probs, logits, n, sampler->temperature);

    /* Populate calibrated_prob field */
    for (size_t i = 0; i < n; ++i) {
        out_preds[i].calibrated_prob = probs[i];
    }

    free(scores);
    free(logits);
    free(probs);

    return count;
}

void psam_composite_set_sampler_defaults(
    psam_composite_t* composite,
    const psam_sampler_t* sampler
) {
    if (!composite || !sampler) {
        return;
    }
    composite->sampler_defaults = *sampler;
}

void psam_composite_get_sampler_defaults(
    const psam_composite_t* composite,
    psam_sampler_t* out_sampler
) {
    if (!composite || !out_sampler) {
        return;
    }
    *out_sampler = composite->sampler_defaults;
}
