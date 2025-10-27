/**
 * vocab_alignment.c - Vocabulary alignment implementation
 */

#include "psam_vocab_alignment.h"
#include "psam.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

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

/* Placeholder implementations for aligned composite functions */
/* These will be implemented in Week 3 */

psam_composite_aligned_t* psam_create_composite_aligned(
    psam_model_t* base,
    psam_vocab_alignment_t* alignment,
    bool owns_alignment
) {
    (void)base;
    (void)alignment;
    (void)owns_alignment;
    fprintf(stderr, "psam_create_composite_aligned: not yet implemented\n");
    return NULL;
}

int psam_composite_aligned_add_layer(
    psam_composite_aligned_t* composite,
    const char* layer_id,
    psam_model_t* model,
    float weight,
    bool owns_model
) {
    (void)composite;
    (void)layer_id;
    (void)model;
    (void)weight;
    (void)owns_model;
    fprintf(stderr, "psam_composite_aligned_add_layer: not yet implemented\n");
    return -1;
}

int psam_composite_aligned_predict(
    psam_composite_aligned_t* composite,
    const uint32_t* context,
    size_t context_len,
    void* out_preds,
    size_t max_preds
) {
    (void)composite;
    (void)context;
    (void)context_len;
    (void)out_preds;
    (void)max_preds;
    fprintf(stderr, "psam_composite_aligned_predict: not yet implemented\n");
    return -1;
}

void psam_composite_aligned_destroy(psam_composite_aligned_t* composite) {
    if (!composite) return;

    if (composite->owns_composite && composite->composite) {
        psam_composite_destroy(composite->composite);
    }
    if (composite->owns_alignment && composite->alignment) {
        psam_vocab_alignment_destroy(composite->alignment);
    }
    free(composite);
}
