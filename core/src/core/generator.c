/**
 * generator.c - Stateful generator with residual and salience tracking
 *
 * Maintains residual buffer and salience anchors across sequential predictions.
 */

#include "../psam_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * Single deferred association
 */
typedef struct {
    uint32_t candidate;
    float contribution;
    int remaining_offset;
} deferred_association_t;

/**
 * Residual buffer
 */
typedef struct {
    deferred_association_t* associations;
    size_t count;
    size_t capacity;
} residual_buffer_t;

/**
 * EWMA state for a single token
 */
typedef struct {
    float ewma_freq;        /* Exponentially weighted moving average of frequency */
    float ewma_contrib;     /* EWMA of contribution to predictions */
    float prev_ewma_contrib; /* Previous EWMA contrib for computing delta */
    uint32_t last_seen_pos; /* Last position where this token was seen */
} ewma_state_t;

/**
 * Single salient anchor
 */
typedef struct {
    uint32_t token;         /* Token ID */
    uint32_t position;      /* Position where token was last seen */
    float salience_score;   /* S_t(a) = freq + eta*delta_contrib + kappa*IDF */
} salient_anchor_t;

/**
 * Salience tracker - maintains anchors and EWMA state
 */
typedef struct {
    /* Configuration */
    psam_salience_config_t config;

    /* EWMA state per token (sparse, only for seen tokens) */
    ewma_state_t* ewma_states;      /* Array sized to vocab_size */

    /* Anchor heap (max-heap by salience score) */
    salient_anchor_t* anchors;      /* Array of size max_anchors */
    size_t anchor_count;

    /* Current generation position (for computing distances) */
    uint32_t current_position;
    size_t last_context_len;        /* Track last context length */

    /* Alpha factors for EWMA updates */
    float alpha_freq;               /* = 1 - exp(-ln(2) / halflife_freq) */
    float alpha_contrib;            /* = 1 - exp(-ln(2) / halflife_contrib) */
} salience_tracker_t;

/**
 * High-perplexity token tracker (simpler than full salience)
 */
typedef struct {
    uint32_t* tokens;      /* Array of high-perplexity token IDs */
    size_t count;          /* Current count */
    size_t capacity;       /* Max capacity */
    float boost_amount;    /* How much to boost these tokens */
} perplexity_tracker_t;

/**
 * Stateful generator structure
 */
struct psam_generator {
    psam_model_t* model;              /* Reference to model (not owned) */
    psam_residual_config_t res_config; /* Residual configuration */
    psam_salience_config_t sal_config; /* Salience configuration */
    psam_sampler_t sampler;           /* Sampler configuration */
    residual_buffer_t* residuals;     /* Persistent residual buffer */
    salience_tracker_t* salience;     /* Salience tracker (NULL if disabled) */
    perplexity_tracker_t* perplexity; /* High-perplexity token tracker */
    bool use_sampler;                 /* Whether sampler was provided */
};

/* Forward declarations for internal functions */
static residual_buffer_t* create_residual_buffer(size_t initial_capacity);
static void destroy_residual_buffer(residual_buffer_t* buffer);
static void add_residual(residual_buffer_t* buffer, uint32_t candidate, float contribution, int remaining_offset);
static void apply_and_age_residuals(residual_buffer_t* buffer, float* scores, uint32_t vocab_size);
static void compute_future_residuals(
    psam_generator_t* gen,
    const uint32_t* context,
    size_t context_len
);

/* Salience tracking functions */
static salience_tracker_t* create_salience_tracker(
    const psam_salience_config_t* config,
    uint32_t vocab_size
);
static void destroy_salience_tracker(salience_tracker_t* tracker);
static void salience_update_context(
    salience_tracker_t* tracker,
    psam_model_t* model,
    const uint32_t* context,
    size_t context_len
);
static void salience_update_contributions(
    salience_tracker_t* tracker,
    psam_model_t* model,
    const float* raw_strengths,
    const uint32_t* token_ids,
    size_t count
);
static void salience_apply_anchors(
    salience_tracker_t* tracker,
    psam_model_t* model,
    const uint32_t* context,
    size_t context_len,
    float* scores
);

/* Perplexity-based token tracking */
static perplexity_tracker_t* create_perplexity_tracker(size_t capacity, float boost_amount);
static void destroy_perplexity_tracker(perplexity_tracker_t* tracker);
static void perplexity_track_context(
    perplexity_tracker_t* tracker,
    psam_model_t* model,
    const uint32_t* context,
    size_t context_len,
    float perplexity_threshold
);
static void perplexity_boost_scores(
    perplexity_tracker_t* tracker,
    float* scores,
    uint32_t vocab_size
);

/* Extern declaration for IDF computation (from infer.c) */
extern float compute_idf(const psam_model_t* model, uint32_t token);
extern int find_row_index(const psam_model_t* model, uint32_t token, uint32_t offset);

/* ============================ Public API ============================ */

psam_generator_t* psam_create_generator(
    psam_model_t* model,
    const psam_residual_config_t* residual_config,
    const psam_salience_config_t* salience_config,
    const psam_sampler_t* sampler
) {
    if (!model || !model->is_finalized) {
        return NULL;
    }

    psam_generator_t* gen = malloc(sizeof(psam_generator_t));
    if (!gen) {
        return NULL;
    }

    gen->model = model;

    /* Copy residual config with defaults */
    if (residual_config && residual_config->enable) {
        gen->res_config = *residual_config;
        if (gen->res_config.max_lookahead <= 0) {
            gen->res_config.max_lookahead = 3;
        }
        if (gen->res_config.residual_decay <= 0.0f || gen->res_config.residual_decay > 1.0f) {
            gen->res_config.residual_decay = 0.8f;
        }
        if (gen->res_config.residual_blend < 0.0f || gen->res_config.residual_blend > 1.0f) {
            gen->res_config.residual_blend = 0.4f;
        }
    } else {
        /* Disabled */
        memset(&gen->res_config, 0, sizeof(psam_residual_config_t));
        gen->res_config.enable = false;
    }

    /* Copy salience config with defaults */
    if (salience_config && salience_config->enable) {
        gen->sal_config = *salience_config;
        if (gen->sal_config.max_anchors <= 0) {
            gen->sal_config.max_anchors = 16;
        }
        if (gen->sal_config.ewma_freq_halflife <= 0.0f) {
            gen->sal_config.ewma_freq_halflife = 128.0f;
        }
        if (gen->sal_config.ewma_contrib_halflife <= 0.0f) {
            gen->sal_config.ewma_contrib_halflife = 64.0f;
        }
        if (gen->sal_config.eta <= 0.0f) {
            gen->sal_config.eta = 1.0f;
        }
        if (gen->sal_config.kappa <= 0.0f) {
            gen->sal_config.kappa = 0.25f;
        }
        if (gen->sal_config.beta <= 0.0f) {
            gen->sal_config.beta = 0.3f;
        }
        if (gen->sal_config.pop_decay_distance <= 0.0f) {
            gen->sal_config.pop_decay_distance = 256.0f;
        }
        if (gen->sal_config.min_salience <= 0.0f) {
            gen->sal_config.min_salience = 0.1f;
        }
    } else {
        /* Disabled */
        memset(&gen->sal_config, 0, sizeof(psam_salience_config_t));
        gen->sal_config.enable = false;
    }

    /* Copy sampler config */
    gen->use_sampler = (sampler != NULL);
    if (sampler) {
        gen->sampler = *sampler;
    } else {
        /* Default sampler */
        memset(&gen->sampler, 0, sizeof(psam_sampler_t));
        gen->sampler.transform = PSAM_LOGIT_ZSCORE;
        gen->sampler.temperature = 1.0f;
        gen->sampler.top_p = 0.95f;
        gen->sampler.seed = 42;
    }

    /* Create residual buffer */
    gen->residuals = create_residual_buffer(model->config.vocab_size);
    if (!gen->residuals) {
        free(gen);
        return NULL;
    }

    /* Create salience tracker if enabled */
    if (gen->sal_config.enable) {
        gen->salience = create_salience_tracker(&gen->sal_config, model->config.vocab_size);
        if (!gen->salience) {
            destroy_residual_buffer(gen->residuals);
            free(gen);
            return NULL;
        }
    } else {
        gen->salience = NULL;
    }

    /* Create perplexity tracker (always enabled for now, simple boost) */
    gen->perplexity = create_perplexity_tracker(32, 50.0f);  /* Track up to 32 tokens, boost by 50 */
    if (!gen->perplexity) {
        destroy_salience_tracker(gen->salience);
        destroy_residual_buffer(gen->residuals);
        free(gen);
        return NULL;
    }

    return gen;
}

void psam_destroy_generator(psam_generator_t* generator) {
    if (!generator) {
        return;
    }

    destroy_residual_buffer(generator->residuals);
    destroy_salience_tracker(generator->salience);
    destroy_perplexity_tracker(generator->perplexity);
    free(generator);
}

psam_error_t psam_generator_reset(psam_generator_t* generator) {
    if (!generator) {
        return PSAM_ERR_NULL_PARAM;
    }

    /* Clear residual buffer */
    if (generator->residuals) {
        generator->residuals->count = 0;
    }

    /* Reset salience tracker */
    if (generator->salience) {
        generator->salience->anchor_count = 0;
        generator->salience->current_position = 0;
        generator->salience->last_context_len = 0;
        /* Zero out EWMA states */
        memset(generator->salience->ewma_states, 0,
               generator->model->config.vocab_size * sizeof(ewma_state_t));
    }

    /* Reset perplexity tracker */
    if (generator->perplexity) {
        generator->perplexity->count = 0;
    }

    return PSAM_OK;
}

int psam_generator_predict(
    psam_generator_t* generator,
    const uint32_t* context,
    size_t context_len,
    psam_prediction_t* out_preds,
    size_t max_preds
) {
    if (!generator || !context || !out_preds) {
        return PSAM_ERR_NULL_PARAM;
    }

    psam_model_t* model = generator->model;
    const uint32_t vocab_size = model->config.vocab_size;

    if (vocab_size == 0 || context_len == 0) {
        return 0;
    }

    psam_lock_rdlock(&model->lock);

    int result = 0;

    /* Allocate score buffers */
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

    /* Initialize with bias */
    for (uint32_t i = 0; i < vocab_size; i++) {
        scores[i] = model->bias[i];
    }

    /* Compute current-position scores (standard PSAM logic) */
    if (model->csr && model->csr->row_count > 0) {
        for (size_t ctx_idx = 0; ctx_idx < context_len; ctx_idx++) {
            uint32_t token = context[ctx_idx];
            if (token >= vocab_size) {
                continue;
            }

            uint32_t offset = (uint32_t)(context_len - ctx_idx);
            int row_idx = find_row_index(model, token, offset);

            if (row_idx < 0) {
                continue;
            }

            uint32_t row_start = model->csr->row_offsets[row_idx];
            uint32_t row_end = model->csr->row_offsets[row_idx + 1];
            float row_scale = model->csr->row_scales[row_idx];

            /* Apply IDF and distance decay */
            float idf = compute_idf(model, token);
            float decay = expf(-model->config.alpha * (float)(offset - 1));

            for (uint32_t edge = row_start; edge < row_end; edge++) {
                uint32_t target = model->csr->targets[edge];
                if (target >= vocab_size) continue;

                float weight = (float)model->csr->weights[edge];
                float delta = row_scale * weight * idf * decay;

                scores[target] += delta;
                raw_strength[target] += delta;
                if (delta != 0.0f && support_counts[target] < UINT16_MAX) {
                    support_counts[target]++;
                }
            }
        }
    }

    /* Track high-perplexity tokens from context */
    if (generator->perplexity) {
        perplexity_track_context(generator->perplexity, model, context, context_len, 2.0f);
    }

    /* Boost high-perplexity tokens */
    if (generator->perplexity) {
        perplexity_boost_scores(generator->perplexity, scores, vocab_size);
    }

    /* Update salience tracker with context (if enabled) */
    if (generator->salience) {
        salience_update_context(generator->salience, model, context, context_len);
    }

    /* Apply residuals from previous steps (if enabled) */
    if (generator->res_config.enable) {
        apply_and_age_residuals(generator->residuals, scores, vocab_size);
    }

    /* Apply salience anchors (if enabled) */
    if (generator->salience) {
        salience_apply_anchors(generator->salience, model, context, context_len, scores);
    }

    /* Compute new residuals for future positions (if enabled) */
    if (generator->res_config.enable) {
        compute_future_residuals(generator, context, context_len);
    }

    /* Build predictions array */
    psam_prediction_t* all_preds = malloc(vocab_size * sizeof(psam_prediction_t));
    if (!all_preds) {
        result = PSAM_ERR_OUT_OF_MEMORY;
        goto cleanup_buffers;
    }

    for (uint32_t i = 0; i < vocab_size; i++) {
        all_preds[i].token = i;
        float raw = raw_strength[i];
        float bias_component = scores[i] - raw;
        float contextual = raw;

        /* Consensus gain */
        if (support_counts[i] > 1) {
            contextual *= 1.0f + PSAM_CONSENSUS_GAIN * (float)(support_counts[i] - 1);
        }

        float final_score = bias_component + contextual;
        all_preds[i].score = final_score;
        all_preds[i].raw_strength = raw;
        all_preds[i].support_count = support_counts[i];
        all_preds[i]._reserved = 0;
        all_preds[i].calibrated_prob = 0.0f;
    }

    /* Sort by score */
    extern int compare_predictions(const void* a, const void* b);
    qsort(all_preds, vocab_size, sizeof(psam_prediction_t), compare_predictions);

    /* Copy top-K to output */
    size_t output_count = max_preds < vocab_size ? max_preds : vocab_size;
    memcpy(out_preds, all_preds, output_count * sizeof(psam_prediction_t));

    /* Update salience contributions with prediction results (if enabled) */
    if (generator->salience) {
        /* Extract token IDs and raw strengths from top predictions */
        uint32_t* pred_token_ids = malloc(output_count * sizeof(uint32_t));
        float* pred_raw_strengths = malloc(output_count * sizeof(float));

        if (pred_token_ids && pred_raw_strengths) {
            for (size_t i = 0; i < output_count; i++) {
                pred_token_ids[i] = all_preds[i].token;
                pred_raw_strengths[i] = all_preds[i].raw_strength;
            }

            salience_update_contributions(generator->salience, model,
                                        pred_raw_strengths, pred_token_ids, output_count);
        }

        free(pred_token_ids);
        free(pred_raw_strengths);
    }

    result = (int)output_count;

    free(all_preds);

cleanup_buffers:
    free(scores);
    free(raw_strength);
    free(support_counts);

cleanup:
    psam_lock_unlock_rd(&model->lock);

    return result;
}

/* ============================ Internal Helper Functions ============================ */

static residual_buffer_t* create_residual_buffer(size_t initial_capacity) {
    residual_buffer_t* buffer = malloc(sizeof(residual_buffer_t));
    if (!buffer) {
        return NULL;
    }

    buffer->associations = malloc(initial_capacity * sizeof(deferred_association_t));
    if (!buffer->associations) {
        free(buffer);
        return NULL;
    }

    buffer->count = 0;
    buffer->capacity = initial_capacity;
    return buffer;
}

static void destroy_residual_buffer(residual_buffer_t* buffer) {
    if (buffer) {
        free(buffer->associations);
        free(buffer);
    }
}

static void add_residual(
    residual_buffer_t* buffer,
    uint32_t candidate,
    float contribution,
    int remaining_offset
) {
    /* Grow buffer if needed */
    if (buffer->count >= buffer->capacity) {
        size_t new_capacity = buffer->capacity * 2;
        deferred_association_t* new_assocs = realloc(
            buffer->associations,
            new_capacity * sizeof(deferred_association_t)
        );
        if (!new_assocs) {
            return;  /* Silently skip */
        }
        buffer->associations = new_assocs;
        buffer->capacity = new_capacity;
    }

    /* Add association */
    buffer->associations[buffer->count].candidate = candidate;
    buffer->associations[buffer->count].contribution = contribution;
    buffer->associations[buffer->count].remaining_offset = remaining_offset;
    buffer->count++;
}

static void apply_and_age_residuals(residual_buffer_t* buffer, float* scores, uint32_t vocab_size) {
    size_t write_idx = 0;

    for (size_t i = 0; i < buffer->count; i++) {
        deferred_association_t* assoc = &buffer->associations[i];

        if (assoc->remaining_offset == 0) {
            /* Fire now! */
            if (assoc->candidate < vocab_size) {
                scores[assoc->candidate] += assoc->contribution;
            }
            /* Don't keep this association */
        } else {
            /* Age it and keep it */
            assoc->remaining_offset--;
            if (write_idx != i) {
                buffer->associations[write_idx] = *assoc;
            }
            write_idx++;
        }
    }

    buffer->count = write_idx;
}

static void compute_future_residuals(
    psam_generator_t* gen,
    const uint32_t* context,
    size_t context_len
) {
    psam_model_t* model = gen->model;
    const uint32_t vocab_size = model->config.vocab_size;
    const uint32_t window = model->config.window;
    const int max_lookahead = gen->res_config.max_lookahead;
    const float residual_decay = gen->res_config.residual_decay;
    const float residual_blend = gen->res_config.residual_blend;

    /* For each context token, compute associations at future offsets */
    for (size_t ctx_idx = 0; ctx_idx < context_len; ctx_idx++) {
        uint32_t token = context[ctx_idx];
        if (token >= vocab_size) {
            continue;
        }

        float idf = compute_idf(model, token);
        uint32_t current_offset = (uint32_t)(context_len - ctx_idx);

        /* Look ahead */
        for (int lookahead = 1; lookahead <= max_lookahead; lookahead++) {
            uint32_t future_offset = current_offset + lookahead;

            if (future_offset > window) {
                break;
            }

            /* Find associations at this future offset */
            int row_idx = find_row_index(model, token, future_offset);
            if (row_idx < 0) {
                continue;
            }

            uint32_t row_start = model->csr->row_offsets[row_idx];
            uint32_t row_end = model->csr->row_offsets[row_idx + 1];
            float row_scale = model->csr->row_scales[row_idx];

            /* Distance decay for future position */
            float decay = expf(-model->config.alpha * (float)(future_offset - 1));

            /* Residual decay for deferred activation */
            float defer_decay = powf(residual_decay, (float)lookahead);

            /* Add associations to residual buffer */
            for (uint32_t edge = row_start; edge < row_end; edge++) {
                uint32_t target = model->csr->targets[edge];
                if (target >= vocab_size) continue;

                float weight = (float)model->csr->weights[edge];
                float contribution = row_scale * weight * idf * decay * defer_decay * residual_blend;

                /* Add to residual buffer */
                add_residual(gen->residuals, target, contribution, lookahead - 1);
            }
        }
    }
}

/* ============================ Salience Tracking Implementation ============================ */

/**
 * Create and initialize salience tracker
 */
static salience_tracker_t* create_salience_tracker(
    const psam_salience_config_t* config,
    uint32_t vocab_size
) {
    salience_tracker_t* tracker = malloc(sizeof(salience_tracker_t));
    if (!tracker) {
        return NULL;
    }

    tracker->config = *config;
    tracker->current_position = 0;
    tracker->last_context_len = 0;
    tracker->anchor_count = 0;

    /* Allocate EWMA states for all tokens */
    tracker->ewma_states = calloc(vocab_size, sizeof(ewma_state_t));
    if (!tracker->ewma_states) {
        free(tracker);
        return NULL;
    }

    /* Allocate anchor array */
    tracker->anchors = malloc(config->max_anchors * sizeof(salient_anchor_t));
    if (!tracker->anchors) {
        free(tracker->ewma_states);
        free(tracker);
        return NULL;
    }

    /* Compute alpha factors for EWMA: alpha = 1 - exp(-ln(2) / halflife) */
    tracker->alpha_freq = 1.0f - expf(-0.693147f / config->ewma_freq_halflife);
    tracker->alpha_contrib = 1.0f - expf(-0.693147f / config->ewma_contrib_halflife);

    return tracker;
}

/**
 * Destroy salience tracker
 */
static void destroy_salience_tracker(salience_tracker_t* tracker) {
    if (!tracker) {
        return;
    }

    free(tracker->ewma_states);
    free(tracker->anchors);
    free(tracker);
}

/**
 * Update EWMA frequency for tokens in context
 */
static void salience_update_context(
    salience_tracker_t* tracker,
    psam_model_t* model,
    const uint32_t* context,
    size_t context_len
) {
    const uint32_t vocab_size = model->config.vocab_size;

    /* Only process NEW tokens (from last_context_len to context_len) */
    size_t start_idx = tracker->last_context_len;
    if (start_idx > context_len) {
        start_idx = 0;  /* Reset if context shrunk */
    }

    /* Update frequency EWMA for each NEW token */
    for (size_t i = start_idx; i < context_len; i++) {
        uint32_t token = context[i];
        if (token >= vocab_size) {
            continue;
        }

        ewma_state_t* state = &tracker->ewma_states[token];

        /* EWMA update: ewma_new = alpha * 1.0 + (1 - alpha) * ewma_old */
        state->ewma_freq = tracker->alpha_freq + (1.0f - tracker->alpha_freq) * state->ewma_freq;
        state->last_seen_pos = tracker->current_position;

        tracker->current_position++;  /* Increment for each token processed */
    }

    /* Update last context length */
    tracker->last_context_len = context_len;
}

/**
 * Min-heap operations for maintaining top-K anchors
 */
static void heap_sift_down(salient_anchor_t* anchors, size_t count, size_t idx) {
    while (1) {
        size_t smallest = idx;
        size_t left = 2 * idx + 1;
        size_t right = 2 * idx + 2;

        if (left < count && anchors[left].salience_score < anchors[smallest].salience_score) {
            smallest = left;
        }
        if (right < count && anchors[right].salience_score < anchors[smallest].salience_score) {
            smallest = right;
        }

        if (smallest == idx) {
            break;
        }

        /* Swap */
        salient_anchor_t temp = anchors[idx];
        anchors[idx] = anchors[smallest];
        anchors[smallest] = temp;

        idx = smallest;
    }
}

static void heap_insert(salient_anchor_t* anchors, size_t* count, size_t max_count,
                       uint32_t token, uint32_t position, float salience_score) {
    if (*count < max_count) {
        /* Add to heap */
        anchors[*count].token = token;
        anchors[*count].position = position;
        anchors[*count].salience_score = salience_score;
        (*count)++;

        /* Heapify */
        for (int i = (*count / 2) - 1; i >= 0; i--) {
            heap_sift_down(anchors, *count, i);
        }
    } else if (salience_score > anchors[0].salience_score) {
        /* Replace minimum */
        anchors[0].token = token;
        anchors[0].position = position;
        anchors[0].salience_score = salience_score;
        heap_sift_down(anchors, *count, 0);
    }
}

/**
 * Update contribution EWMA and recompute salience scores
 */
static void salience_update_contributions(
    salience_tracker_t* tracker,
    psam_model_t* model,
    const float* raw_strengths,
    const uint32_t* token_ids,
    size_t count
) {
    const uint32_t vocab_size = model->config.vocab_size;

    /* Update EWMA contribution for top predictions */
    for (size_t i = 0; i < count; i++) {
        uint32_t token = token_ids[i];
        if (token >= vocab_size) {
            continue;
        }

        ewma_state_t* state = &tracker->ewma_states[token];
        float contrib = raw_strengths[i];

        /* Save previous for delta computation */
        state->prev_ewma_contrib = state->ewma_contrib;

        /* Update EWMA */
        state->ewma_contrib = tracker->alpha_contrib * contrib +
                             (1.0f - tracker->alpha_contrib) * state->ewma_contrib;
    }

    /* Recompute salience scores and rebuild anchor heap */
    tracker->anchor_count = 0;

    for (uint32_t token = 0; token < vocab_size; token++) {
        ewma_state_t* state = &tracker->ewma_states[token];

        /* Skip tokens with no activity */
        if (state->ewma_freq < 1e-6f && state->ewma_contrib < 1e-6f) {
            continue;
        }

        /* Compute delta (pop-out signal) */
        float delta_contrib = state->ewma_contrib - state->prev_ewma_contrib;

        /* Compute salience score: S_t(a) = ewma_freq + eta*delta_contrib + kappa*IDF */
        float idf = compute_idf(model, token);
        float salience = state->ewma_freq +
                        tracker->config.eta * delta_contrib +
                        tracker->config.kappa * idf;

        /* Only consider tokens above minimum threshold */
        if (salience < tracker->config.min_salience) {
            continue;
        }

        /* Insert into anchor heap */
        heap_insert(tracker->anchors, &tracker->anchor_count, tracker->config.max_anchors,
                   token, state->last_seen_pos, salience);
    }
}

/**
 * Apply anchor votes to prediction scores
 */
static void salience_apply_anchors(
    salience_tracker_t* tracker,
    psam_model_t* model,
    const uint32_t* context,
    size_t context_len,
    float* scores
) {
    const uint32_t vocab_size = model->config.vocab_size;

    (void)context;      /* Unused - kept for future extensions */
    (void)context_len;  /* Unused - kept for future extensions */

    if (tracker->anchor_count == 0 || !model->csr) {
        return;
    }

    /* For each anchor, compute its contribution to scores */
    for (size_t a = 0; a < tracker->anchor_count; a++) {
        salient_anchor_t* anchor = &tracker->anchors[a];
        uint32_t anchor_token = anchor->token;

        if (anchor_token >= vocab_size) {
            continue;
        }

        /* Compute distance from current position */
        uint32_t distance = tracker->current_position - anchor->position;

        /* Long-range penalty: pop(D) with slow falloff */
        float pop_penalty = expf(-(float)distance / tracker->config.pop_decay_distance);

        /* Salience weight (normalized to 0-1 range) */
        float salience_weight = fminf(1.0f, anchor->salience_score);

        /* Combined weight */
        float anchor_weight = tracker->config.beta * salience_weight * pop_penalty;

        if (anchor_weight < 1e-6f) {
            continue;
        }

        /* Get IDF for anchor token */
        float idf = compute_idf(model, anchor_token);

        /* Find all associations for this anchor (collapsed across offsets) */
        /* We approximate by summing contributions across reasonable offsets */
        for (uint32_t offset = 1; offset <= model->config.window; offset++) {
            int row_idx = find_row_index(model, anchor_token, offset);
            if (row_idx < 0) {
                continue;
            }

            uint32_t row_start = model->csr->row_offsets[row_idx];
            uint32_t row_end = model->csr->row_offsets[row_idx + 1];
            float row_scale = model->csr->row_scales[row_idx];

            /* Distance decay for this offset */
            float decay = expf(-model->config.alpha * (float)(offset - 1));

            /* Add contribution to each target */
            for (uint32_t edge = row_start; edge < row_end; edge++) {
                uint32_t target = model->csr->targets[edge];
                if (target >= vocab_size) continue;

                float weight = (float)model->csr->weights[edge];
                float contribution = anchor_weight * row_scale * weight * idf * decay;

                scores[target] += contribution;
            }
        }
    }
}

/* ============================ Perplexity Token Tracking ============================ */

/**
 * Create perplexity tracker
 */
static perplexity_tracker_t* create_perplexity_tracker(size_t capacity, float boost_amount) {
    perplexity_tracker_t* tracker = malloc(sizeof(perplexity_tracker_t));
    if (!tracker) {
        return NULL;
    }

    tracker->tokens = malloc(capacity * sizeof(uint32_t));
    if (!tracker->tokens) {
        free(tracker);
        return NULL;
    }

    tracker->count = 0;
    tracker->capacity = capacity;
    tracker->boost_amount = boost_amount;

    return tracker;
}

/**
 * Destroy perplexity tracker
 */
static void destroy_perplexity_tracker(perplexity_tracker_t* tracker) {
    if (!tracker) {
        return;
    }

    free(tracker->tokens);
    free(tracker);
}

/**
 * Compute perplexity for a token and track if high
 *
 * Simple perplexity approximation: Use IDF as proxy
 * High IDF = rare/informative = high perplexity
 */
static void perplexity_track_context(
    perplexity_tracker_t* tracker,
    psam_model_t* model,
    const uint32_t* context,
    size_t context_len,
    float perplexity_threshold
) {
    const uint32_t vocab_size = model->config.vocab_size;

    /* Check each token in context */
    for (size_t i = 0; i < context_len; i++) {
        uint32_t token = context[i];
        if (token >= vocab_size) {
            continue;
        }

        /* Use IDF as perplexity proxy: high IDF = informative = track it */
        float idf = compute_idf(model, token);

        if (idf > perplexity_threshold) {
            /* Check if already tracked */
            bool already_tracked = false;
            for (size_t j = 0; j < tracker->count; j++) {
                if (tracker->tokens[j] == token) {
                    already_tracked = true;
                    break;
                }
            }

            /* Add if not already tracked and have space */
            if (!already_tracked && tracker->count < tracker->capacity) {
                tracker->tokens[tracker->count++] = token;
            }
        }
    }
}

/**
 * Boost scores for tracked high-perplexity tokens
 */
static void perplexity_boost_scores(
    perplexity_tracker_t* tracker,
    float* scores,
    uint32_t vocab_size
) {
    for (size_t i = 0; i < tracker->count; i++) {
        uint32_t token = tracker->tokens[i];
        if (token < vocab_size) {
            scores[token] += tracker->boost_amount;
        }
    }
}
