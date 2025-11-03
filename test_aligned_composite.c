/**
 * test_aligned_composite.c - Test aligned composite with mixed vocabularies
 *
 * This tests the aligned composite API using Hamlet and Macbeth models
 * trained with the unified Shakespeare vocabulary (Phase 1 --vocab-in).
 *
 * Usage: ./test_aligned_composite
 */

#include "core/include/psam.h"
#include "core/include/psam_vocab_alignment.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
        true  /* owns alignment */
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

    /* Cleanup */
    printf("\n✓ Test completed successfully!\n");
    psam_composite_aligned_destroy(composite);
    psam_destroy(hamlet);
    psam_destroy(macbeth);

    return 0;
}
