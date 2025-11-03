/**
 * test_aligned_composite.c - Test aligned composite with mixed vocabularies
 *
 * This tests the aligned composite API using Hamlet and Macbeth models
 * trained with the unified Shakespeare vocabulary (Phase 1 --vocab-in).
 *
 * Usage: ./test_aligned_composite
 */

#include "core/include/psam.h"
#include "core/include/psam_composite.h"
#include "core/include/psam_vocab_alignment.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <direct.h>
#endif

/* Helper: find a token ID in the unified vocabulary */
static int find_unified_token_id(const psam_vocab_alignment_t* alignment, const char* token) {
    if (!alignment || !token) {
        return -1;
    }
    for (uint32_t i = 0; i < alignment->unified_vocab_size; ++i) {
        if (alignment->unified_tokens[i] && strcmp(alignment->unified_tokens[i], token) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static int ensure_dir(const char* path) {
    if (!path || path[0] == '\0') {
        return -1;
    }
#ifdef _WIN32
    if (_mkdir(path) == 0 || errno == EEXIST) {
        return 0;
    }
#else
    if (mkdir(path, 0755) == 0 || errno == EEXIST) {
        return 0;
    }
#endif
    return -1;
}

int main(void) {
    const char* hamlet_model_path = "corpora/text/Folger/models/hamlet.psam";
    const char* macbeth_model_path = "corpora/text/Folger/models/macbeth.psam";
    const char* hamlet_vocab_path = "corpora/text/Folger/models/hamlet.tsv";
    const char* macbeth_vocab_path = "corpora/text/Folger/models/macbeth.tsv";

    printf("=== Aligned Composite Test ===\n\n");

    /* Step 1: Build vocabulary alignment */
    printf("Step 1: Building vocabulary alignment...\n");
    const char* vocab_paths[] = {hamlet_vocab_path, macbeth_vocab_path};
    uint32_t unified_size;

    psam_vocab_alignment_t* alignment = psam_build_vocab_alignment_from_files(
        vocab_paths, 2, NULL, &unified_size);

    if (!alignment) {
        fprintf(stderr, "ERROR: Failed to build vocab alignment\n");
        return 1;
    }

    printf("  Unified vocabulary: %u tokens\n", unified_size);
    printf("  Hamlet coverage: %.1f%%\n", psam_vocab_alignment_get_coverage(alignment, 0) * 100.0f);
    printf("  Macbeth coverage: %.1f%%\n\n", psam_vocab_alignment_get_coverage(alignment, 1) * 100.0f);

    /* Step 2: Load models */
    printf("Step 2: Loading models...\n");
    psam_model_t* hamlet = psam_load(hamlet_model_path);
    if (!hamlet) {
        fprintf(stderr, "ERROR: Failed to load Hamlet model\n");
        psam_vocab_alignment_destroy(alignment);
        return 1;
    }
    printf("  Loaded Hamlet model\n");

    psam_model_t* macbeth = psam_load(macbeth_model_path);
    if (!macbeth) {
        fprintf(stderr, "ERROR: Failed to load Macbeth model\n");
        psam_destroy(hamlet);
        psam_vocab_alignment_destroy(alignment);
        return 1;
    }
    printf("  Loaded Macbeth model\n\n");

    /* Step 3: Create aligned composite */
    printf("Step 3: Creating aligned composite...\n");
    psam_composite_aligned_t* composite = psam_create_composite_aligned(
        hamlet,  /* base */
        alignment,
        true,  /* owns alignment */
        false /* owns base */
    );

    if (!composite) {
        fprintf(stderr, "ERROR: Failed to create aligned composite\n");
        psam_destroy(hamlet);
        psam_destroy(macbeth);
        psam_vocab_alignment_destroy(alignment);
        return 1;
    }
    printf("  Created aligned composite with Hamlet as base\n");

    /* Configure aligned composite policies */
    psam_composite_aligned_set_unknown_policy(composite, PSAM_UNKNOWN_SKIP);
    psam_composite_aligned_set_coverage_rule(composite, PSAM_COVER_LINEAR);
    if (psam_composite_aligned_set_base_weight(composite, 1.0f) != 0) {
        fprintf(stderr, "ERROR: Failed to set base weight\n");
        psam_composite_aligned_destroy(composite);
        psam_destroy(hamlet);
        psam_destroy(macbeth);
        return 1;
    }

    /* Step 4: Add Macbeth overlay */
    printf("\nStep 4: Adding Macbeth overlay layer...\n");
    int rc = psam_composite_aligned_add_layer(
        composite,
        "macbeth",
        macbeth,
        0.5f,  /* weight: blend 50% Macbeth with Hamlet */
        false  /* don't transfer ownership */
    );

    if (rc != 0) {
        fprintf(stderr, "ERROR: Failed to add Macbeth layer\n");
        psam_composite_aligned_destroy(composite);
        psam_destroy(hamlet);
        psam_destroy(macbeth);
        return 1;
    }
    printf("  Added Macbeth layer with weight 0.5\n\n");

    if (psam_composite_aligned_update_layer_weight(composite, "macbeth", 0.5f) != 0) {
        fprintf(stderr, "ERROR: Failed to update layer weight\n");
        psam_composite_aligned_destroy(composite);
        psam_destroy(hamlet);
        psam_destroy(macbeth);
        return 1;
    }

    if (psam_composite_aligned_update_layer_bias(composite, "macbeth", 0.1f) != 0) {
        fprintf(stderr, "ERROR: Failed to update layer bias\n");
        psam_composite_aligned_destroy(composite);
        psam_destroy(hamlet);
        psam_destroy(macbeth);
        return 1;
    }

    /* Step 5: Test prediction */
    printf("Step 5: Testing prediction...\n");

    /* Build context: "To be or not to" */
    const char* context_tokens[] = {"To", "be", "or", "not", "to"};
    uint32_t context[5];
    int valid_tokens = 0;

    for (int i = 0; i < 5; ++i) {
        int id = find_unified_token_id(alignment, context_tokens[i]);
        if (id < 0) {
            fprintf(stderr, "WARNING: Token '%s' not found in vocabulary\n", context_tokens[i]);
            continue;
        }
        context[valid_tokens++] = (uint32_t)id;
    }

    if (valid_tokens == 0) {
        fprintf(stderr, "ERROR: No valid context tokens\n");
        psam_composite_aligned_destroy(composite);
        psam_destroy(hamlet);
        psam_destroy(macbeth);
        return 1;
    }

    printf("  Context: ");
    for (int i = 0; i < valid_tokens; ++i) {
        printf("%s ", context_tokens[i]);
    }
    printf("\n");

    /* Predict */
    psam_prediction_t preds[10];
    int num_preds = psam_composite_aligned_predict(
        composite,
        context,
        valid_tokens,
        preds,
        10
    );

    if (num_preds < 0) {
        fprintf(stderr, "ERROR: Prediction failed\n");
        psam_composite_aligned_destroy(composite);
        psam_destroy(hamlet);
        psam_destroy(macbeth);
        return 1;
    }

    printf("\n  Top %d predictions:\n", num_preds);
    for (int i = 0; i < num_preds; ++i) {
        printf("    %d. Token ID %u, score %.3f, calibrated_prob %.4f\n",
               i + 1,
               preds[i].token,
               preds[i].score,
               preds[i].calibrated_prob);
    }

    /* Round-trip save/load test using the public API */
    if (ensure_dir("test_output") != 0 || ensure_dir("test_output/maps") != 0) {
        fprintf(stderr, "ERROR: Failed to create test_output directories\n");
        psam_composite_aligned_destroy(composite);
        psam_destroy(hamlet);
        psam_destroy(macbeth);
        return 1;
    }

    const char* save_layer_ids[2] = {"hamlet", "macbeth"};
    const char* save_model_paths[2] = {hamlet_model_path, macbeth_model_path};
    float save_weights[2] = {1.0f, 0.5f};
    float save_biases[2] = {0.0f, 0.1f};
    const uint32_t save_local_vocab_sizes[2] = {
        alignment->layer_remaps[0].local_vocab_size,
        alignment->layer_remaps[1].local_vocab_size
    };
    const uint32_t* save_l2u_maps[2] = {
        alignment->layer_remaps[0].local_to_unified,
        alignment->layer_remaps[1].local_to_unified
    };

    uint32_t* save_u2l_pairs_buf[2] = {NULL, NULL};
    const uint32_t* save_u2l_pairs[2] = {NULL, NULL};
    const uint32_t save_u2l_counts[2] = {
        alignment->layer_remaps[0].unified_to_local_count,
        alignment->layer_remaps[1].unified_to_local_count
    };

    for (size_t i = 0; i < 2; ++i) {
        uint32_t count = save_u2l_counts[i];
        if (count > 0) {
            save_u2l_pairs_buf[i] = malloc((size_t)count * 2 * sizeof(uint32_t));
            if (!save_u2l_pairs_buf[i]) {
                fprintf(stderr, "ERROR: Out of memory building u2l pairs\n");
                for (size_t k = 0; k < i; ++k) {
                    free(save_u2l_pairs_buf[k]);
                }
                psam_composite_aligned_destroy(composite);
                psam_destroy(hamlet);
                psam_destroy(macbeth);
                return 1;
            }
            for (uint32_t j = 0; j < count; ++j) {
                save_u2l_pairs_buf[i][j * 2 + 0] = alignment->layer_remaps[i].unified_to_local_sparse[j].unified_id;
                save_u2l_pairs_buf[i][j * 2 + 1] = alignment->layer_remaps[i].unified_to_local_sparse[j].local_id;
            }
            save_u2l_pairs[i] = save_u2l_pairs_buf[i];
        }
    }

    char save_l2u_paths[2][256];
    char save_u2l_paths[2][256];
    snprintf(save_l2u_paths[0], sizeof(save_l2u_paths[0]), "test_output/maps/hamlet_api.l2u.u32");
    snprintf(save_l2u_paths[1], sizeof(save_l2u_paths[1]), "test_output/maps/macbeth_api.l2u.u32");
    snprintf(save_u2l_paths[0], sizeof(save_u2l_paths[0]), "test_output/maps/hamlet_api.u2l.pairs");
    snprintf(save_u2l_paths[1], sizeof(save_u2l_paths[1]), "test_output/maps/macbeth_api.u2l.pairs");

    const char* l2u_path_ptrs[2] = {save_l2u_paths[0], save_l2u_paths[1]};
    const char* u2l_path_ptrs[2] = {save_u2l_paths[0], save_u2l_paths[1]};
    const char* unified_vocab_path = "corpora/text/Folger/unified_vocab.tsv";

    int save_rc = psam_composite_save_v1(
        "test_output/aligned_unit.psamc",
        "test-aligned",
        unified_vocab_path,
        PSAM_UNKNOWN_SKIP,
        PSAM_COVER_LINEAR,
        NULL,
        2,
        save_layer_ids,
        save_model_paths,
        save_weights,
        save_biases,
        save_local_vocab_sizes,
        save_l2u_maps,
        save_u2l_pairs,
        save_u2l_counts,
        l2u_path_ptrs,
        u2l_path_ptrs,
        unified_size
    );

    for (size_t i = 0; i < 2; ++i) {
        free(save_u2l_pairs_buf[i]);
    }

    if (save_rc != 0) {
        fprintf(stderr, "ERROR: Failed to save aligned composite via API\n");
        psam_composite_aligned_destroy(composite);
        psam_destroy(hamlet);
        psam_destroy(macbeth);
        return 1;
    }

    psam_composite_aligned_t* reloaded = psam_composite_load_aligned("test_output/aligned_unit.psamc", false);
    if (!reloaded) {
        fprintf(stderr, "ERROR: Failed to reload aligned composite\n");
        psam_composite_aligned_destroy(composite);
        psam_destroy(hamlet);
        psam_destroy(macbeth);
        return 1;
    }

    psam_prediction_t preds_reload[10];
    int num_preds_reload = psam_composite_aligned_predict(reloaded, context, valid_tokens, preds_reload, 10);
    if (num_preds_reload <= 0) {
        fprintf(stderr, "ERROR: Reloaded composite produced no predictions\n");
        psam_composite_aligned_destroy(reloaded);
        psam_composite_aligned_destroy(composite);
        psam_destroy(hamlet);
        psam_destroy(macbeth);
        return 1;
    }
    printf("\n  Reloaded composite predictions: %d items\n", num_preds_reload);
    psam_composite_aligned_destroy(reloaded);

    /* Cleanup */
    printf("\n✓ Test completed successfully!\n");
    psam_composite_aligned_destroy(composite);
    psam_destroy(hamlet);
    psam_destroy(macbeth);

    return 0;
}
