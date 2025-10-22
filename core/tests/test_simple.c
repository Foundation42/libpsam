/**
 * Simple debug test to understand the associations being learned
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../include/psam.h"

int main() {
    printf("=== Simple PSAM Debug Test ===\n\n");

    // Very simple training: "a b c"
    // We want to learn: a -> b (offset 1), a -> c (offset 2), b -> c (offset 1)
    uint32_t tokens[] = {0, 1, 2};  // a=0, b=1, c=2
    size_t num_tokens = 3;

    printf("Training on: 0 1 2 (representing: a b c)\n\n");

    // Create model
    psam_config_t config = {
        .vocab_size = 3,
        .window = 8,
        .top_k = 10,
        .alpha = 0.1,
        .min_evidence = 1,
        .enable_idf = true,   // Enable to match demo
        .enable_ppmi = true,  // Enable to match demo
        .edge_dropout = 0.0
    };

    psam_model_t* model = psam_create_with_config(&config);
    assert(model != NULL);

    // Train
    psam_train_batch(model, tokens, num_tokens);
    psam_finalize_training(model);

    // Get stats
    psam_stats_t stats;
    psam_get_stats(model, &stats);
    printf("Stats: edges=%lu\n\n", stats.edge_count);

    // Test 1: context = [0] (just "a"), should predict 1 ("b")
    printf("Test 1: Context [0] -> ?\n");
    uint32_t context1[] = {0};
    psam_prediction_t preds1[10];
    int n1 = psam_predict(model, context1, 1, preds1, 10);
    printf("  Predictions (%d):\n", n1);
    for (int i = 0; i < n1 && i < 5; i++) {
        printf("    %d: token=%u score=%.3f\n", i+1, preds1[i].token, preds1[i].score);
    }
    if (n1 > 0 && preds1[0].token == 1) {
        printf("  ✓ PASS\n");
    } else {
        printf("  ✗ FAIL: Expected token 1\n");
    }
    printf("\n");

    // Test 2: context = [0, 1] (just "a b"), should predict 2 ("c")
    printf("Test 2: Context [0, 1] -> ?\n");
    uint32_t context2[] = {0, 1};
    psam_prediction_t preds2[10];
    int n2 = psam_predict(model, context2, 2, preds2, 10);
    printf("  Predictions (%d):\n", n2);
    for (int i = 0; i < n2 && i < 5; i++) {
        printf("    %d: token=%u score=%.3f\n", i+1, preds2[i].token, preds2[i].score);
    }
    if (n2 > 0 && preds2[0].token == 2) {
        printf("  ✓ PASS\n");
    } else {
        printf("  ✗ FAIL: Expected token 2\n");
    }

    psam_destroy(model);
    return 0;
}
