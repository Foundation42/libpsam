/**
 * vocab_alignment.c - Vocabulary alignment implementation
 */

#include "psam_vocab_alignment.h"
#include "psam.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>

/* Helper: Compare function for qsort/bsearch on sparse entries */
static int compare_sparse_entry(const void* a, const void* b) {
    const psam_vocab_sparse_entry_t* ea = (const psam_vocab_sparse_entry_t*)a;
    const psam_vocab_sparse_entry_t* eb = (const psam_vocab_sparse_entry_t*)b;
    if (ea->unified_id < eb->unified_id) return -1;
    if (ea->unified_id > eb->unified_id) return 1;
    return 0;
}

/* Helper: Compare function for qsort on string pointers */
static int compare_str_ptr(const void* a, const void* b) {
    return strcmp(*(const char**)a, *(const char**)b);
}

/* Helper: Binary search for string in sorted array */
static int find_token_index(const char* token, char** sorted_tokens, size_t count) {
    char** found = bsearch(&token, sorted_tokens, count, sizeof(char*), compare_str_ptr);
    if (!found) return -1;
    return (int)(found - sorted_tokens);
}

/* Helper: Load vocabulary from TSV file */
typedef struct {
    char** tokens;
    uint32_t count;
} vocab_tsv_t;

static int load_vocab_tsv(const char* path, vocab_tsv_t* out_vocab) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "ERROR: Failed to open vocab '%s': %s\n", path, strerror(errno));
        return -1;
    }

    char** temp_tokens = NULL;
    uint32_t capacity = 1000;
    uint32_t count = 0;

    temp_tokens = malloc(capacity * sizeof(char*));
    if (!temp_tokens) {
        fclose(f);
        return -1;
    }

    char line[4096];
    uint32_t expected_id = 0;

    while (fgets(line, sizeof(line), f)) {
        char* tab = strchr(line, '\t');
        if (!tab) {
            fprintf(stderr, "ERROR: Malformed vocab line in '%s': %s\n", path, line);
            for (uint32_t i = 0; i < count; ++i) free(temp_tokens[i]);
            free(temp_tokens);
            fclose(f);
            return -1;
        }

        /* Verify sequential IDs */
        *tab = '\0';
        uint32_t id = 0;
        if (sscanf(line, "%u", &id) != 1 || id != expected_id) {
            fprintf(stderr, "ERROR: Vocab IDs must be sequential in '%s', got %u expected %u\n",
                    path, id, expected_id);
            for (uint32_t i = 0; i < count; ++i) free(temp_tokens[i]);
            free(temp_tokens);
            fclose(f);
            return -1;
        }
        *tab = '\t';
        expected_id++;

        /* Extract token */
        char* token_start = tab + 1;
        size_t token_len = strlen(token_start);
        if (token_len > 0 && token_start[token_len - 1] == '\n') {
            token_start[token_len - 1] = '\0';
        }

        /* Expand if needed */
        if (count >= capacity) {
            capacity *= 2;
            char** new_tokens = realloc(temp_tokens, capacity * sizeof(char*));
            if (!new_tokens) {
                for (uint32_t i = 0; i < count; ++i) free(temp_tokens[i]);
                free(temp_tokens);
                fclose(f);
                return -1;
            }
            temp_tokens = new_tokens;
        }

        temp_tokens[count++] = strdup(token_start);
    }
    fclose(f);

    if (count == 0) {
        fprintf(stderr, "ERROR: Empty vocabulary file '%s'\n", path);
        free(temp_tokens);
        return -1;
    }

    out_vocab->tokens = temp_tokens;
    out_vocab->count = count;
    return 0;
}

/**
 * Build unified vocabulary from vocabulary files
 *
 * Strategy:
 * 1. Load all vocabularies from TSV files
 * 2. Collect all unique tokens and sort
 * 3. Build bidirectional mappings for each vocabulary
 */
psam_vocab_alignment_t* psam_build_vocab_alignment_from_files(
    const char** vocab_paths,
    size_t n_vocabs,
    const char** layer_ids,
    uint32_t* out_unified_vocab_size
) {
    if (!vocab_paths || n_vocabs == 0 || !out_unified_vocab_size) {
        fprintf(stderr, "psam_build_vocab_alignment_from_files: invalid arguments\n");
        return NULL;
    }

    /* Step 1: Load all vocabularies from files */
    vocab_tsv_t* vocabs = malloc(n_vocabs * sizeof(vocab_tsv_t));
    if (!vocabs) {
        return NULL;
    }

    for (size_t i = 0; i < n_vocabs; ++i) {
        if (load_vocab_tsv(vocab_paths[i], &vocabs[i]) != 0) {
            /* Free already-loaded vocabs */
            for (size_t j = 0; j < i; ++j) {
                for (uint32_t k = 0; k < vocabs[j].count; ++k) {
                    free(vocabs[j].tokens[k]);
                }
                free(vocabs[j].tokens);
            }
            free(vocabs);
            return NULL;
        }
    }

    /* Step 2: Collect all unique tokens */
    typedef struct {
        char** tokens;
        size_t count;
        size_t capacity;
    } token_list_t;

    token_list_t all_tokens = {0};
    all_tokens.capacity = 10000;
    all_tokens.tokens = malloc(all_tokens.capacity * sizeof(char*));
    if (!all_tokens.tokens) {
        for (size_t i = 0; i < n_vocabs; ++i) {
            for (uint32_t k = 0; k < vocabs[i].count; ++k) {
                free(vocabs[i].tokens[k]);
            }
            free(vocabs[i].tokens);
        }
        free(vocabs);
        return NULL;
    }

    /* Collect all tokens from all vocabularies */
    for (size_t vocab_idx = 0; vocab_idx < n_vocabs; ++vocab_idx) {
        for (uint32_t tok_idx = 0; tok_idx < vocabs[vocab_idx].count; ++tok_idx) {
            if (all_tokens.count >= all_tokens.capacity) {
                all_tokens.capacity *= 2;
                char** new_tokens = realloc(all_tokens.tokens, all_tokens.capacity * sizeof(char*));
                if (!new_tokens) {
                    for (size_t j = 0; j < all_tokens.count; ++j) free(all_tokens.tokens[j]);
                    free(all_tokens.tokens);
                    for (size_t i = 0; i < n_vocabs; ++i) {
                        for (uint32_t k = 0; k < vocabs[i].count; ++k) {
                            free(vocabs[i].tokens[k]);
                        }
                        free(vocabs[i].tokens);
                    }
                    free(vocabs);
                    return NULL;
                }
                all_tokens.tokens = new_tokens;
            }
            all_tokens.tokens[all_tokens.count++] = strdup(vocabs[vocab_idx].tokens[tok_idx]);
        }
    }

    /* Step 3: Sort and deduplicate */
    qsort(all_tokens.tokens, all_tokens.count, sizeof(char*), compare_str_ptr);

    size_t unique_count = 0;
    for (size_t i = 0; i < all_tokens.count; ++i) {
        if (i == 0 || strcmp(all_tokens.tokens[i], all_tokens.tokens[unique_count - 1]) != 0) {
            all_tokens.tokens[unique_count++] = all_tokens.tokens[i];
        } else {
            free(all_tokens.tokens[i]);  /* Free duplicate */
        }
    }

    /* Step 4: Create alignment structure with unified vocabulary */
    psam_vocab_alignment_t* alignment = calloc(1, sizeof(psam_vocab_alignment_t));
    if (!alignment) {
        for (size_t i = 0; i < unique_count; ++i) free(all_tokens.tokens[i]);
        free(all_tokens.tokens);
        for (size_t i = 0; i < n_vocabs; ++i) {
            for (uint32_t k = 0; k < vocabs[i].count; ++k) {
                free(vocabs[i].tokens[k]);
            }
            free(vocabs[i].tokens);
        }
        free(vocabs);
        return NULL;
    }

    alignment->num_layers = (uint32_t)n_vocabs;
    alignment->unknown_policy = PSAM_UNKNOWN_SKIP;
    alignment->owns_unified_tokens = true;
    alignment->unified_vocab_size = (uint32_t)unique_count;
    alignment->unified_tokens = malloc(unique_count * sizeof(char*));
    if (!alignment->unified_tokens) {
        for (size_t i = 0; i < unique_count; ++i) free(all_tokens.tokens[i]);
        free(all_tokens.tokens);
        for (size_t i = 0; i < n_vocabs; ++i) {
            for (uint32_t k = 0; k < vocabs[i].count; ++k) {
                free(vocabs[i].tokens[k]);
            }
            free(vocabs[i].tokens);
        }
        free(vocabs);
        free(alignment);
        return NULL;
    }

    for (size_t i = 0; i < unique_count; ++i) {
        alignment->unified_tokens[i] = all_tokens.tokens[i];  /* Transfer ownership */
    }
    free(all_tokens.tokens);  /* Free container */

    /* Step 5: Allocate per-layer remapping tables */
    alignment->layer_remaps = calloc(n_vocabs, sizeof(psam_vocab_remap_t));
    if (!alignment->layer_remaps) {
        psam_vocab_alignment_destroy(alignment);
        for (size_t i = 0; i < n_vocabs; ++i) {
            for (uint32_t k = 0; k < vocabs[i].count; ++k) {
                free(vocabs[i].tokens[k]);
            }
            free(vocabs[i].tokens);
        }
        free(vocabs);
        return NULL;
    }

    /* Step 6: Build per-layer remapping tables */
    for (size_t vocab_idx = 0; vocab_idx < n_vocabs; ++vocab_idx) {
        psam_vocab_remap_t* remap = &alignment->layer_remaps[vocab_idx];
        vocab_tsv_t* vocab = &vocabs[vocab_idx];

        remap->local_vocab_size = vocab->count;

        /* Allocate dense local→unified map */
        remap->local_to_unified = malloc(vocab->count * sizeof(uint32_t));
        if (!remap->local_to_unified) {
            psam_vocab_alignment_destroy(alignment);
            for (size_t i = 0; i < n_vocabs; ++i) {
                for (uint32_t k = 0; k < vocabs[i].count; ++k) {
                    free(vocabs[i].tokens[k]);
                }
                free(vocabs[i].tokens);
            }
            free(vocabs);
            return NULL;
        }

        /* Allocate sparse unified→local map */
        remap->unified_to_local_sparse = malloc(vocab->count * sizeof(psam_vocab_sparse_entry_t));
        if (!remap->unified_to_local_sparse) {
            psam_vocab_alignment_destroy(alignment);
            for (size_t i = 0; i < n_vocabs; ++i) {
                for (uint32_t k = 0; k < vocabs[i].count; ++k) {
                    free(vocabs[i].tokens[k]);
                }
                free(vocabs[i].tokens);
            }
            free(vocabs);
            return NULL;
        }
        remap->unified_to_local_count = 0;

        /* Build mappings for this layer */
        for (uint32_t local_id = 0; local_id < vocab->count; ++local_id) {
            const char* token = vocab->tokens[local_id];

            /* Find token in unified vocabulary */
            int unified_id = find_token_index(token, alignment->unified_tokens, unique_count);
            if (unified_id < 0) {
                fprintf(stderr, "ERROR: Token '%s' from layer %zu not found in unified vocab\n",
                        token, vocab_idx);
                remap->local_to_unified[local_id] = PSAM_VOCAB_INVALID_ID;
                continue;
            }

            /* Dense mapping: local → unified */
            remap->local_to_unified[local_id] = (uint32_t)unified_id;

            /* Sparse mapping: unified → local */
            remap->unified_to_local_sparse[remap->unified_to_local_count].unified_id = (uint32_t)unified_id;
            remap->unified_to_local_sparse[remap->unified_to_local_count].local_id = local_id;
            remap->unified_to_local_count++;
        }

        /* Sort sparse map for binary search */
        qsort(remap->unified_to_local_sparse, remap->unified_to_local_count,
              sizeof(psam_vocab_sparse_entry_t), compare_sparse_entry);

        /* Calculate coverage */
        remap->coverage = (float)remap->unified_to_local_count / (float)alignment->unified_vocab_size;
    }

    /* Free original vocabulary data (tokens were copied to alignment) */
    for (size_t i = 0; i < n_vocabs; ++i) {
        for (uint32_t k = 0; k < vocabs[i].count; ++k) {
            free(vocabs[i].tokens[k]);
        }
        free(vocabs[i].tokens);
    }
    free(vocabs);

    *out_unified_vocab_size = alignment->unified_vocab_size;

    fprintf(stderr, "INFO: Built vocab alignment with %u tokens from %zu layers\n",
            alignment->unified_vocab_size, n_vocabs);
    for (size_t i = 0; i < n_vocabs; ++i) {
        const char* layer_name = (layer_ids && layer_ids[i]) ? layer_ids[i] : "(unnamed)";
        fprintf(stderr, "  Layer %zu (%s): %u tokens, coverage %.1f%%\n",
                i, layer_name, alignment->layer_remaps[i].local_vocab_size,
                alignment->layer_remaps[i].coverage * 100.0f);
    }

    return alignment;
}

void psam_vocab_alignment_destroy(psam_vocab_alignment_t* alignment) {
    if (!alignment) return;

    if (alignment->owns_unified_tokens && alignment->unified_tokens) {
        for (uint32_t i = 0; i < alignment->unified_vocab_size; ++i) {
            free(alignment->unified_tokens[i]);
        }
        free(alignment->unified_tokens);
    }

    if (alignment->layer_remaps) {
        for (uint32_t i = 0; i < alignment->num_layers; ++i) {
            free(alignment->layer_remaps[i].local_to_unified);
            free(alignment->layer_remaps[i].unified_to_local_sparse);
        }
        free(alignment->layer_remaps);
    }

    free(alignment);
}

uint32_t psam_vocab_remap_local_to_unified(
    const psam_vocab_remap_t* remap,
    uint32_t local_id
) {
    if (!remap || local_id >= remap->local_vocab_size) {
        return PSAM_VOCAB_INVALID_ID;
    }
    return remap->local_to_unified[local_id];
}

uint32_t psam_vocab_remap_unified_to_local(
    const psam_vocab_remap_t* remap,
    uint32_t unified_id
) {
    if (!remap) {
        return PSAM_VOCAB_INVALID_ID;
    }

    /* Binary search in sparse map */
    psam_vocab_sparse_entry_t key = { .unified_id = unified_id, .local_id = 0 };
    psam_vocab_sparse_entry_t* found = bsearch(&key, remap->unified_to_local_sparse,
                                                 remap->unified_to_local_count,
                                                 sizeof(psam_vocab_sparse_entry_t),
                                                 compare_sparse_entry);
    if (!found) {
        return PSAM_VOCAB_INVALID_ID;
    }

    return found->local_id;
}

float psam_vocab_alignment_get_coverage(
    const psam_vocab_alignment_t* alignment,
    uint32_t layer_index
) {
    if (!alignment || layer_index >= alignment->num_layers) {
        return -1.0f;
    }
    return alignment->layer_remaps[layer_index].coverage;
}

/* ========================================================================
 * Aligned Composite Implementation (Week 3)
 * ======================================================================== */

typedef struct {
    uint32_t token;
    float score;
} aligned_score_t;

static float apply_coverage_rule(psam_coverage_rule_t rule, float coverage) {
    if (coverage <= 0.0f) {
        return 0.0f;
    }
    switch (rule) {
        case PSAM_COVER_LINEAR:
            return coverage;
        case PSAM_COVER_SQRT:
            return sqrtf(coverage);
        case PSAM_COVER_NONE:
        default:
            return 1.0f;
    }
}

static void accumulate_aligned_scores(
    aligned_score_t* accum,
    size_t* accum_size,
    size_t capacity,
    const psam_prediction_t* preds,
    size_t pred_count,
    const psam_vocab_remap_t* remap,
    float weight,
    float bias
) {
    if (!accum || !accum_size || !preds || !remap) {
        return;
    }

    for (size_t i = 0; i < pred_count; ++i) {
        uint32_t unified_id = psam_vocab_remap_local_to_unified(remap, preds[i].token);
        if (unified_id == PSAM_VOCAB_INVALID_ID) {
            continue;
        }

        float contribution = preds[i].score * weight + bias;
        if (contribution == 0.0f && bias == 0.0f) {
            continue;
        }

        bool found = false;
        for (size_t j = 0; j < *accum_size; ++j) {
            if (accum[j].token == unified_id) {
                accum[j].score += contribution;
                found = true;
                break;
            }
        }

        if (!found && *accum_size < capacity) {
            accum[*accum_size].token = unified_id;
            accum[*accum_size].score = contribution;
            (*accum_size)++;
        }
    }
}

static size_t remap_context_for_layer(
    const psam_vocab_remap_t* remap,
    const uint32_t* context,
    size_t context_len,
    psam_unknown_policy_t policy,
    uint32_t* out_local,
    size_t* out_known_tokens
) {
    if (!remap || !context || context_len == 0 || !out_local) {
        if (out_known_tokens) {
            *out_known_tokens = 0;
        }
        return 0;
    }

    size_t local_len = 0;
    size_t known_tokens = 0;

    for (size_t i = 0; i < context_len; ++i) {
        uint32_t unified_id = context[i];
        uint32_t local_id = psam_vocab_remap_unified_to_local(remap, unified_id);

        if (local_id == PSAM_VOCAB_INVALID_ID) {
            if (policy == PSAM_UNKNOWN_MAP_UNK) {
                if (remap->local_vocab_size == 0) {
                    continue;
                }
                local_id = remap->local_vocab_size - 1;
            } else if (policy == PSAM_UNKNOWN_COVERAGE || policy == PSAM_UNKNOWN_SKIP) {
                continue;
            }
        } else {
            known_tokens++;
        }

        out_local[local_len++] = local_id;
    }

    if (out_known_tokens) {
        *out_known_tokens = known_tokens;
    }
    return local_len;
}

static int compare_aligned_scores_desc(const void* a, const void* b) {
    const aligned_score_t* sa = (const aligned_score_t*)a;
    const aligned_score_t* sb = (const aligned_score_t*)b;
    if (sa->score > sb->score) return -1;
    if (sa->score < sb->score) return 1;
    return 0;
}

psam_composite_aligned_t* psam_create_composite_aligned(
    psam_model_t* base,
    psam_vocab_alignment_t* alignment,
    bool owns_alignment,
    bool owns_base
) {
    if (!base || !alignment) {
        fprintf(stderr, "ERROR: psam_create_composite_aligned requires base model and alignment\n");
        return NULL;
    }

    if (alignment->num_layers < 1) {
        fprintf(stderr, "ERROR: Alignment must have at least one layer (base)\n");
        return NULL;
    }

    /* Create standard composite with base model */
    psam_composite_t* composite = psam_create_layered(base);
    if (!composite) {
        fprintf(stderr, "ERROR: Failed to create layered composite\n");
        return NULL;
    }

    /* Wrap in aligned composite structure */
    psam_composite_aligned_t* aligned = calloc(1, sizeof(psam_composite_aligned_t));
    if (!aligned) {
        psam_composite_destroy(composite);
        return NULL;
    }

    aligned->composite = composite;
    aligned->alignment = alignment;
    aligned->unknown_policy = alignment ? alignment->unknown_policy : PSAM_UNKNOWN_SKIP;
    aligned->coverage_rule = PSAM_COVER_NONE;
    aligned->base_model = base;
    aligned->base_weight = 1.0f;
    aligned->owns_base_model = owns_base;
    aligned->layers = NULL;
    aligned->layer_count = 0;
    aligned->layer_capacity = 0;
    aligned->owns_composite = true;
    aligned->owns_alignment = owns_alignment;

    fprintf(stderr, "INFO: Created aligned composite with %u-token unified vocabulary\n",
            alignment->unified_vocab_size);

    return aligned;
}

int psam_composite_aligned_add_layer(
    psam_composite_aligned_t* composite,
    const char* layer_id,
    psam_model_t* model,
    float weight,
    bool owns_model
) {
    if (!composite || !composite->composite || !layer_id || !model) {
        fprintf(stderr, "ERROR: Invalid arguments to psam_composite_aligned_add_layer\n");
        return -1;
    }

    if (!composite->alignment || composite->alignment->num_layers <= composite->layer_count + 1) {
        fprintf(stderr, "ERROR: Alignment does not have remap entry for layer '%s'\n", layer_id);
        return -1;
    }

    if (composite->layer_count == composite->layer_capacity) {
        size_t new_capacity = composite->layer_capacity ? composite->layer_capacity * 2 : 4;
        psam_aligned_layer_t* resized = realloc(composite->layers, new_capacity * sizeof(psam_aligned_layer_t));
        if (!resized) {
            fprintf(stderr, "ERROR: Failed to grow aligned layer list\n");
            return -1;
        }
        composite->layers = resized;
        composite->layer_capacity = new_capacity;
    }

    psam_aligned_layer_t* entry = &composite->layers[composite->layer_count];
    memset(entry, 0, sizeof(*entry));
    entry->id = strdup(layer_id);
    if (!entry->id) {
        fprintf(stderr, "ERROR: Failed to allocate layer id string\n");
        return -1;
    }
    entry->model = model;
    entry->weight = weight;
    entry->bias = 0.0f;
    entry->owns_model = owns_model;

    uint32_t alignment_index = (uint32_t)(composite->layer_count + 1);
    if (alignment_index >= composite->alignment->num_layers) {
        fprintf(stderr, "ERROR: Alignment missing remap for layer '%s'\n", layer_id);
        free(entry->id);
        entry->id = NULL;
        return -1;
    }
    entry->alignment_index = alignment_index;
    composite->layer_count++;

    fprintf(stderr, "INFO: Added layer '%s' with weight %.3f to aligned composite\n",
            layer_id, weight);

    return 0;
}

int psam_composite_aligned_predict(
    psam_composite_aligned_t* composite,
    const uint32_t* context,
    size_t context_len,
    void* out_preds,
    size_t max_preds
) {
    if (!composite || !composite->composite || !composite->alignment) {
        fprintf(stderr, "ERROR: Invalid aligned composite\n");
        return -1;
    }

    if (!context || context_len == 0 || !out_preds || max_preds == 0) {
        fprintf(stderr, "ERROR: Invalid prediction parameters\n");
        return -1;
    }

    psam_prediction_t* preds = (psam_prediction_t*)out_preds;
    size_t total_layers = composite->layer_count + 1; /* base + overlays */
    if (total_layers == 0) {
        fprintf(stderr, "ERROR: No layers available for aligned composite prediction\n");
        return -1;
    }

    size_t capacity = total_layers * max_preds;
    aligned_score_t* accum = calloc(capacity, sizeof(aligned_score_t));
    if (!accum) {
        return -1;
    }

    psam_prediction_t* scratch = malloc(max_preds * sizeof(psam_prediction_t));
    uint32_t* local_context = context_len > 0 ? malloc(context_len * sizeof(uint32_t)) : NULL;
    if (!scratch || (context_len > 0 && !local_context)) {
        free(accum);
        free(scratch);
        free(local_context);
        return -1;
    }

    size_t accum_size = 0;
    int result = 0;
    const psam_vocab_alignment_t* alignment = composite->alignment;
    psam_unknown_policy_t policy = composite->unknown_policy;
    psam_coverage_rule_t coverage_rule = composite->coverage_rule;

    /* Helper lambda via do-while pattern for base + overlays */
    for (size_t layer_idx = 0; layer_idx < total_layers; ++layer_idx) {
        psam_model_t* layer_model = NULL;
        const psam_vocab_remap_t* remap = NULL;
        float weight = 0.0f;
        float bias = 0.0f;

        if (layer_idx == 0) {
            layer_model = composite->base_model;
            remap = &alignment->layer_remaps[0];
            weight = composite->base_weight;
            bias = 0.0f;
        } else {
            psam_aligned_layer_t* layer = &composite->layers[layer_idx - 1];
            layer_model = layer->model;
            remap = &alignment->layer_remaps[layer->alignment_index];
            weight = layer->weight;
            bias = layer->bias;
        }

        if (!layer_model || !remap) {
            continue;
        }

        if (weight == 0.0f) {
            continue;
        }

        size_t known_tokens = 0;
        size_t local_len = remap_context_for_layer(
            remap,
            context,
            context_len,
            policy,
            local_context,
            &known_tokens
        );

        if (local_len == 0 && policy != PSAM_UNKNOWN_MAP_UNK) {
            continue;
        }

        int count = psam_predict(layer_model, local_context, local_len, scratch, max_preds);
        if (count < 0) {
            result = count;
            break;
        }

        float global_factor = apply_coverage_rule(coverage_rule, remap->coverage);
        if (coverage_rule == PSAM_COVER_NONE) {
            global_factor = 1.0f;
        }

        float context_factor = 1.0f;
        if (policy == PSAM_UNKNOWN_COVERAGE) {
            context_factor = (context_len > 0)
                ? (float)known_tokens / (float)context_len
                : 0.0f;
        }

        float effective_weight = weight * global_factor * context_factor;
        if (effective_weight == 0.0f && bias == 0.0f) {
            continue;
        }

        accumulate_aligned_scores(
            accum,
            &accum_size,
            capacity,
            scratch,
            (size_t)count,
            remap,
            effective_weight,
            bias
        );
    }

    if (result < 0) {
        free(local_context);
        free(scratch);
        free(accum);
        return result;
    }

    if (accum_size == 0) {
        free(local_context);
        free(scratch);
        free(accum);
        return 0;
    }

    qsort(accum, accum_size, sizeof(aligned_score_t), compare_aligned_scores_desc);

    size_t to_copy = accum_size < max_preds ? accum_size : max_preds;
    for (size_t i = 0; i < to_copy; ++i) {
        preds[i].token = accum[i].token;
        preds[i].score = accum[i].score;
        preds[i].calibrated_prob = 0.0f;
    }

    free(local_context);
    free(scratch);
    free(accum);

    return (int)to_copy;
}

/* Forward declarations from infer.c for sampler transforms */
extern void apply_logit_transform(float* logits, const float* scores, size_t n, psam_logit_transform_t transform);
extern void apply_temperature_and_softmax(float* probs, float* logits, size_t n, float temperature);

int psam_composite_aligned_predict_with_sampler(
    psam_composite_aligned_t* composite,
    const uint32_t* context,
    size_t context_len,
    const psam_sampler_t* sampler,
    void* out_preds,
    size_t max_preds
) {
    if (!composite || !out_preds) {
        return PSAM_ERR_NULL_PARAM;
    }

    psam_prediction_t* preds = (psam_prediction_t*)out_preds;

    int count = psam_composite_aligned_predict(composite, context, context_len, preds, max_preds);
    if (count <= 0) {
        return count;
    }

    psam_sampler_t fallback_sampler;
    const psam_sampler_t* sampler_use = sampler;
    if (!sampler_use) {
        psam_composite_get_sampler_defaults(composite->composite, &fallback_sampler);
        sampler_use = &fallback_sampler;
    }

    size_t n = (size_t)count;
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
        scores[i] = preds[i].score;
    }

    apply_logit_transform(logits, scores, n, sampler_use->transform);
    apply_temperature_and_softmax(probs, logits, n, sampler_use->temperature);

    for (size_t i = 0; i < n; ++i) {
        preds[i].calibrated_prob = probs[i];
    }

    free(scores);
    free(logits);
    free(probs);

    return count;
}

void psam_composite_aligned_destroy(psam_composite_aligned_t* composite) {
    if (!composite) return;

    if (composite->owns_composite && composite->composite) {
        psam_composite_destroy(composite->composite);
    }
    if (composite->owns_alignment && composite->alignment) {
        psam_vocab_alignment_destroy(composite->alignment);
    }
    if (composite->owns_base_model && composite->base_model) {
        psam_destroy(composite->base_model);
    }
    if (composite->layers) {
        for (size_t i = 0; i < composite->layer_count; ++i) {
            if (composite->layers[i].owns_model && composite->layers[i].model) {
                psam_destroy(composite->layers[i].model);
            }
            free(composite->layers[i].id);
        }
        free(composite->layers);
    }
    free(composite);
}

void psam_composite_aligned_set_coverage_rule(
    psam_composite_aligned_t* composite,
    psam_coverage_rule_t rule
) {
    if (!composite) {
        return;
    }
    composite->coverage_rule = rule;
}

void psam_composite_aligned_set_unknown_policy(
    psam_composite_aligned_t* composite,
    psam_unknown_policy_t policy
) {
    if (!composite) {
        return;
    }
    composite->unknown_policy = policy;
    if (composite->alignment) {
        composite->alignment->unknown_policy = policy;
    }
}

int psam_composite_aligned_set_base_weight(
    psam_composite_aligned_t* composite,
    float weight
) {
    if (!composite || !composite->composite) {
        return -1;
    }
    psam_error_t err = psam_composite_set_base_weight(composite->composite, weight);
    if (err != PSAM_OK) {
        return -1;
    }
    composite->base_weight = weight;
    return 0;
}

static psam_aligned_layer_t* find_aligned_layer(
    psam_composite_aligned_t* composite,
    const char* layer_id
) {
    if (!composite || !layer_id || !composite->layers) {
        return NULL;
    }
    for (size_t i = 0; i < composite->layer_count; ++i) {
        if (composite->layers[i].id && strcmp(composite->layers[i].id, layer_id) == 0) {
            return &composite->layers[i];
        }
    }
    return NULL;
}

int psam_composite_aligned_update_layer_weight(
    psam_composite_aligned_t* composite,
    const char* layer_id,
    float new_weight
) {
    if (!composite || !composite->composite || !layer_id) {
        return -1;
    }
    psam_aligned_layer_t* layer = find_aligned_layer(composite, layer_id);
    if (!layer) {
        return -1;
    }
    layer->weight = new_weight;
    return 0;
}

int psam_composite_aligned_update_layer_bias(
    psam_composite_aligned_t* composite,
    const char* layer_id,
    float new_bias
) {
    if (!composite || !composite->composite || !layer_id) {
        return -1;
    }
    psam_aligned_layer_t* layer = find_aligned_layer(composite, layer_id);
    if (!layer) {
        return -1;
    }
    layer->bias = new_bias;
    return 0;
}
