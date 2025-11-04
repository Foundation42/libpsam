/**
 * infer.c - Inference engine using CSR accumulation
 */

#define _POSIX_C_SOURCE 200809L

#include "../psam_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

/**
 * Compute IDF (Inverse Document Frequency) for a token.
 * Formula: log((1 + total_tokens) / (1 + token_count)) + 1
 */
static float compute_idf(const psam_model_t* model, uint32_t token) {
    if (token >= model->config.vocab_size) {
        return 1.0f;
    }

    uint32_t occurrences = model->unigram_counts[token];
    if (occurrences == 0) {
        occurrences = 1;  /* Avoid log(inf) */
    }

    return logf((1.0f + (float)model->total_tokens) / (1.0f + (float)occurrences)) + 1.0f;
}

/**
 * Comparison function for sorting predictions by score (descending)
 */
static int compare_predictions(const void* a, const void* b) {
    const psam_prediction_t* pa = (const psam_prediction_t*)a;
    const psam_prediction_t* pb = (const psam_prediction_t*)b;
    if (pa->score > pb->score) return -1;
    if (pa->score < pb->score) return 1;
    return 0;
}

/**
 * Find row index for (token, offset) pair using binary search.
 * Rows are sorted by (source, offset) in ascending order.
 */
static int find_row_index(const psam_model_t* model, uint32_t token, uint32_t offset) {
    if (!model->row_descriptors || model->row_descriptor_count == 0) {
        return -1;
    }

    /* Binary search on sorted row descriptors */
    int left = 0;
    int right = (int)model->row_descriptor_count - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        const row_descriptor_t* row = &model->row_descriptors[mid];

        if (row->source == token && row->offset == offset) {
            return mid;  /* Found exact match */
        }

        /* Compare (source, offset) tuples */
        if (row->source < token || (row->source == token && row->offset < offset)) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;  /* Not found */
}

/* ============================ Inference API ============================ */

int psam_predict(
    psam_model_t* model,
    const uint32_t* context,
    size_t context_len,
    psam_prediction_t* out_preds,
    size_t max_preds
) {
    if (!model || !context || !out_preds) {
        return PSAM_ERR_NULL_PARAM;
    }

    if (!model->is_finalized) {
        return PSAM_ERR_NOT_TRAINED;
    }

    if (model->config.vocab_size == 0 || context_len == 0) {
        return 0;  /* No predictions possible */
    }

    psam_lock_rdlock(&model->lock);

    const uint32_t vocab_size = model->config.vocab_size;
    int result = 0;

    /* 1. Allocate buffers for scores and support metadata */
    float* scores = calloc(vocab_size, sizeof(float));
    float* raw_strength = calloc(vocab_size, sizeof(float));
    uint16_t* support_counts = calloc(vocab_size, sizeof(uint16_t));
    if (!scores || !raw_strength || !support_counts) {
        free(scores);
        free(raw_strength);
        free(support_counts);
        result = PSAM_ERR_OUT_OF_MEMORY;
        goto cleanup;
    }

    /* Initialize scores with bias values */
    for (uint32_t i = 0; i < vocab_size; i++) {
        scores[i] = model->bias[i];
    }

    /* 2. Process each context token */
    if (model->csr && model->csr->row_count > 0) {
        for (size_t i = 0; i < context_len; i++) {
            uint32_t token = context[i];
            if (token >= vocab_size) {
                continue;  /* Skip out-of-vocabulary tokens */
            }

            /* Calculate offset (distance from end of context) */
            uint32_t offset = (uint32_t)(context_len - i);

            /* Find row for (token, offset) pair */
            int row_idx = find_row_index(model, token, offset);
            if (row_idx < 0) {
                continue;  /* Row not found */
            }

            /* Get row bounds from CSR */
            uint32_t row_start = model->csr->row_offsets[row_idx];
            uint32_t row_end = model->csr->row_offsets[row_idx + 1];
            float row_scale = model->csr->row_scales[row_idx];

            /* Accumulate scores for all targets in this row */
            for (uint32_t edge = row_start; edge < row_end; edge++) {
                uint32_t target = model->csr->targets[edge];
                if (target >= vocab_size) {
                    continue;  /* Safety check */
                }

                /* Weight is quantized int16, contribution includes row_scale for dequantization */
                float weight = (float)model->csr->weights[edge];
                float delta = row_scale * weight;

                scores[target] += delta;
                raw_strength[target] += delta;
                if (delta != 0.0f && support_counts[target] < UINT16_MAX) {
                    support_counts[target]++;
                }
            }
        }
    }

    /* 3. Collect all scores into predictions array */
    size_t num_candidates = max_preds < vocab_size ? max_preds : vocab_size;
    psam_prediction_t* all_preds = malloc(vocab_size * sizeof(psam_prediction_t));
    if (!all_preds) {
        result = PSAM_ERR_OUT_OF_MEMORY;
        goto cleanup_scores;
    }

    for (uint32_t i = 0; i < vocab_size; i++) {
        all_preds[i].token = i;
        all_preds[i].score = scores[i];
        all_preds[i].raw_strength = raw_strength[i];
        all_preds[i].support_count = support_counts[i];
        all_preds[i]._reserved = 0;
        all_preds[i].calibrated_prob = 0.0f;  /* TODO: Implement calibration */
    }

    /* 4. Sort by score (descending) */
    qsort(all_preds, vocab_size, sizeof(psam_prediction_t), compare_predictions);

    /* 5. Copy top-K to output */
    size_t output_count = num_candidates < vocab_size ? num_candidates : vocab_size;
    memcpy(out_preds, all_preds, output_count * sizeof(psam_prediction_t));

    result = (int)output_count;

    free(all_preds);

cleanup_scores:
    free(scores);
    free(raw_strength);
    free(support_counts);

cleanup:
    psam_lock_unlock_rd(&model->lock);

    return result;
}

/* ============================ Sampler Utilities ============================ */

/**
 * Compute mean and standard deviation of scores
 */
static void compute_stats(const float* scores, size_t n, float* out_mean, float* out_std) {
    if (n == 0) {
        *out_mean = 0.0f;
        *out_std = 1.0f;
        return;
    }

    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += scores[i];
    }
    float mean = (float)(sum / (double)n);

    double var_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = scores[i] - mean;
        var_sum += diff * diff;
    }
    float std = sqrtf((float)(var_sum / (double)n));

    *out_mean = mean;
    *out_std = std > 1e-6f ? std : 1e-6f;  /* Avoid division by zero */
}

/**
 * Apply logit transform based on mode
 * (Internal helper, exposed for composite layer usage)
 */
void apply_logit_transform(
    float* logits,
    const float* scores,
    size_t n,
    psam_logit_transform_t transform
) {
    if (transform == PSAM_LOGIT_RAW || transform == PSAM_LOGIT_LEGACY) {
        /* Just copy scores as-is (max subtraction happens in softmax) */
        memcpy(logits, scores, n * sizeof(float));
    } else if (transform == PSAM_LOGIT_ZSCORE) {
        /* Z-score normalization */
        float mean, std;
        compute_stats(scores, n, &mean, &std);
        for (size_t i = 0; i < n; ++i) {
            logits[i] = (scores[i] - mean) / std;
        }
    } else if (transform == PSAM_LOGIT_CALIBRATED) {
        /* TODO: Implement calibration with composite calibration data */
        /* For now, fall back to z-score */
        float mean, std;
        compute_stats(scores, n, &mean, &std);
        for (size_t i = 0; i < n; ++i) {
            logits[i] = (scores[i] - mean) / std;
        }
    }
}

/**
 * Apply temperature and compute softmax with numerical stability
 * (Internal helper, exposed for composite layer usage)
 */
void apply_temperature_and_softmax(
    float* probs,
    float* logits,
    size_t n,
    float temperature
) {
    /* Apply temperature scaling */
    if (temperature != 1.0f) {
        float inv_temp = 1.0f / temperature;
        for (size_t i = 0; i < n; ++i) {
            logits[i] *= inv_temp;
        }
    }

    /* Find max for numerical stability */
    float max_logit = logits[0];
    for (size_t i = 1; i < n; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    /* Compute softmax */
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double val = exp((double)(logits[i] - max_logit));
        probs[i] = (float)val;
        sum += val;
    }

    /* Normalize */
    if (sum > 0.0) {
        for (size_t i = 0; i < n; ++i) {
            probs[i] /= (float)sum;
        }
    }
}

/**
 * Get default sampler configuration
 */
static psam_sampler_t get_default_sampler(void) {
    psam_sampler_t sampler = {
        .transform = PSAM_LOGIT_ZSCORE,
        .temperature = 1.0f,
        .top_k = 0,
        .top_p = 0.95f,
        .seed = 42
    };
    return sampler;
}

int psam_predict_with_sampler(
    psam_model_t* model,
    const uint32_t* context,
    size_t context_len,
    const psam_sampler_t* sampler,
    psam_prediction_t* out_preds,
    size_t max_preds
) {
    if (!model || !context || !out_preds) {
        return PSAM_ERR_NULL_PARAM;
    }

    /* Use default sampler if none provided */
    psam_sampler_t default_sampler;
    if (!sampler) {
        default_sampler = get_default_sampler();
        sampler = &default_sampler;
    }

    /* First, get raw predictions */
    int count = psam_predict(model, context, context_len, out_preds, max_preds);
    if (count <= 0) {
        return count;
    }

    size_t n = (size_t)count;

    /* Extract scores into a temporary array */
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

psam_error_t psam_predict_batch(
    psam_model_t* model,
    const uint32_t** contexts,
    const size_t* context_lens,
    size_t batch_size,
    psam_prediction_t** out_preds,
    size_t max_preds_per_context,
    int* out_counts
) {
    if (!model || !contexts || !context_lens || !out_preds || !out_counts) {
        return PSAM_ERR_NULL_PARAM;
    }

    /* Sequential processing for now */
    /* TODO: Parallelize with OpenMP or thread pool */
    for (size_t i = 0; i < batch_size; i++) {
        out_counts[i] = psam_predict(
            model,
            contexts[i],
            context_lens[i],
            out_preds[i],
            max_preds_per_context
        );

        if (out_counts[i] < 0) {
            return (psam_error_t)out_counts[i];
        }
    }

    return PSAM_OK;
}

/**
 * Comparison function for sorting explain terms by contribution (descending)
 */
static int compare_explain_terms(const void* a, const void* b) {
    const psam_explain_term_t* ta = (const psam_explain_term_t*)a;
    const psam_explain_term_t* tb = (const psam_explain_term_t*)b;
    if (ta->contribution > tb->contribution) return -1;
    if (ta->contribution < tb->contribution) return 1;
    return 0;
}

static void maybe_record_explain_term(
    psam_explain_term_t* buffer,
    size_t* stored,
    size_t capacity,
    const psam_explain_term_t* term
) {
    if (capacity == 0) {
        return;
    }

    if (*stored < capacity) {
        buffer[*stored] = *term;
        (*stored)++;
        return;
    }

    /* Replace the smallest contribution if this term is stronger */
    size_t min_idx = 0;
    for (size_t i = 1; i < capacity; i++) {
        if (buffer[i].contribution < buffer[min_idx].contribution) {
            min_idx = i;
        }
    }

    if (buffer[min_idx].contribution < term->contribution) {
        buffer[min_idx] = *term;
    }
}

psam_error_t psam_explain(
    psam_model_t* model,
    const uint32_t* context,
    size_t context_len,
    uint32_t candidate_token,
    psam_explain_term_t* out_terms,
    int max_terms,
    psam_explain_result_t* result
) {
    if (!model || !context || !result) {
        return PSAM_ERR_NULL_PARAM;
    }

    if (max_terms < 0) {
        return PSAM_ERR_INVALID_CONFIG;
    }

    if (max_terms > 0 && !out_terms) {
        return PSAM_ERR_NULL_PARAM;
    }

    if (!model->is_finalized) {
        return PSAM_ERR_NOT_TRAINED;
    }

    const uint32_t vocab_size = model->config.vocab_size;
    const size_t capacity = max_terms > 0 ? (size_t)max_terms : 0;

    if (candidate_token >= vocab_size) {
        result->candidate = candidate_token;
        result->bias_score = 0.0f;
        result->total_score = 0.0f;
        result->term_count = 0;
        return PSAM_OK;
    }

    if (vocab_size == 0 || context_len == 0) {
        result->candidate = candidate_token;
        result->bias_score = model->bias ? model->bias[candidate_token] : 0.0f;
        result->total_score = result->bias_score;
        result->term_count = 0;
        return PSAM_OK;
    }

    /* Initialize result in case we exit early */
    result->candidate = candidate_token;
    result->bias_score = 0.0f;
    result->total_score = 0.0f;
    result->term_count = 0;

    psam_lock_rdlock(&model->lock);

    size_t stored_terms = 0;
    size_t total_terms = 0;
    float contribution_sum = 0.0f;

    if (model->bias) {
        result->bias_score = model->bias[candidate_token];
        result->total_score = result->bias_score;
    }

    /* Process each context token to find contributions to candidate */
    if (model->csr && model->csr->row_count > 0) {
        for (size_t i = 0; i < context_len; i++) {
            uint32_t token = context[i];
            if (token >= vocab_size) {
                continue;  /* Skip out-of-vocabulary tokens */
            }

            /* Calculate offset (distance from end of context) */
            uint32_t offset = (uint32_t)(context_len - i);

            /* Find row for (token, offset) pair */
            int row_idx = find_row_index(model, token, offset);
            if (row_idx < 0) {
                continue;  /* Row not found */
            }

            /* Compute factors */
            float idf = model->config.enable_idf ? compute_idf(model, token) : 1.0f;
            float distance_decay = expf(-model->config.alpha * (float)offset);

            /* Get row bounds from CSR */
            uint32_t row_start = model->csr->row_offsets[row_idx];
            uint32_t row_end = model->csr->row_offsets[row_idx + 1];
            float row_scale = model->csr->row_scales[row_idx];

            /* Search for candidate_token in this row */
            for (uint32_t edge = row_start; edge < row_end; edge++) {
                uint32_t target = model->csr->targets[edge];

                if (target == candidate_token) {
                    /* Found a contribution! */
                    float stored_weight = (float)model->csr->weights[edge] * row_scale;
                    float denom = idf * distance_decay;
                    float weight_ppmi = stored_weight;
                    if (denom > 0.0f) {
                        weight_ppmi = stored_weight / denom;
                    }

                    float contribution = stored_weight;

                    psam_explain_term_t term;
                    term.source_token = token;

                    int32_t signed_offset = (int32_t)offset;
                    if (signed_offset > INT16_MAX) {
                        signed_offset = INT16_MAX;
                    }
                    term.rel_offset = (int16_t)(-signed_offset);
                    term.weight_ppmi = weight_ppmi;
                    term.idf = idf;
                    term.decay = distance_decay;
                    term.contribution = contribution;

                    maybe_record_explain_term(out_terms, &stored_terms, capacity, &term);

                    total_terms++;
                    contribution_sum += contribution;
                    break;  /* Found the target, move to next context token */
                }
            }
        }
    }

    if (stored_terms > 1) {
        qsort(out_terms, stored_terms, sizeof(psam_explain_term_t), compare_explain_terms);
    }

    result->term_count = (int32_t)total_terms;
    result->total_score = result->bias_score + contribution_sum;

    psam_lock_unlock_rd(&model->lock);

    return PSAM_OK;
}
