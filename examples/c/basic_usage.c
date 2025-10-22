/**
 * Basic usage example for libpsam
 *
 * Demonstrates:
 * - Creating a model
 * - Training on token sequences
 * - Making predictions
 * - Saving and loading
 */

#include <stdio.h>
#include <stdlib.h>
#include <psam.h>

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║          libpsam - Basic Usage Example                    ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    // Configuration
    const uint32_t VOCAB_SIZE = 100;
    const uint32_t WINDOW = 8;
    const uint32_t TOP_K = 10;

    // Create model
    printf("📦 Creating PSAM model...\n");
    printf("   - Vocabulary size: %u\n", VOCAB_SIZE);
    printf("   - Window: %u\n", WINDOW);
    printf("   - Top-K: %u\n\n", TOP_K);

    psam_model_t* model = psam_create(VOCAB_SIZE, WINDOW, TOP_K);
    if (!model) {
        fprintf(stderr, "❌ Failed to create model\n");
        return 1;
    }

    // Training data: "the quick brown fox jumps over the lazy dog"
    uint32_t tokens[] = {1, 2, 3, 4, 5, 6, 1, 7, 8};
    size_t num_tokens = sizeof(tokens) / sizeof(tokens[0]);

    printf("📚 Training on sequence...\n");
    printf("   Tokens: [");
    for (size_t i = 0; i < num_tokens; i++) {
        printf("%u%s", tokens[i], i < num_tokens - 1 ? ", " : "");
    }
    printf("]\n\n");

    // Train
    psam_error_t err = psam_train_batch(model, tokens, num_tokens);
    if (err != PSAM_OK) {
        fprintf(stderr, "❌ Training failed: %s\n", psam_error_string(err));
        psam_destroy(model);
        return 1;
    }

    // Finalize training
    err = psam_finalize_training(model);
    if (err != PSAM_OK) {
        fprintf(stderr, "❌ Finalization failed: %s\n", psam_error_string(err));
        psam_destroy(model);
        return 1;
    }

    printf("✓ Training complete!\n\n");

    // Get statistics
    psam_stats_t stats;
    err = psam_get_stats(model, &stats);
    if (err == PSAM_OK) {
        printf("📊 Model Statistics:\n");
        printf("   - Vocabulary: %u tokens\n", stats.vocab_size);
        printf("   - Rows: %u\n", stats.row_count);
        printf("   - Edges: %llu\n", (unsigned long long)stats.edge_count);
        printf("   - Memory: %llu bytes (%.1f KB)\n\n",
               (unsigned long long)stats.memory_bytes,
               stats.memory_bytes / 1024.0);
    }

    // Make predictions
    printf("🔮 Making predictions...\n");
    uint32_t context[] = {1, 2, 3};  // "the quick brown"
    size_t context_len = sizeof(context) / sizeof(context[0]);

    printf("   Context: [");
    for (size_t i = 0; i < context_len; i++) {
        printf("%u%s", context[i], i < context_len - 1 ? ", " : "");
    }
    printf("]\n\n");

    psam_prediction_t predictions[TOP_K];
    int num_preds = psam_predict(model, context, context_len, predictions, TOP_K);

    if (num_preds < 0) {
        fprintf(stderr, "❌ Prediction failed: %s\n",
                psam_error_string((psam_error_t)num_preds));
    } else {
        printf("   Predictions:\n");
        for (int i = 0; i < num_preds; i++) {
            printf("   %2d. Token %u (score: %.3f)\n",
                   i + 1, predictions[i].token_id, predictions[i].score);
        }
        printf("\n");
    }

    // Save model
    const char* filepath = "example_model.psam";
    printf("💾 Saving model to '%s'...\n", filepath);

    err = psam_save(model, filepath);
    if (err != PSAM_OK) {
        fprintf(stderr, "❌ Save failed: %s\n", psam_error_string(err));
    } else {
        printf("✓ Model saved!\n\n");
    }

    // Clean up
    psam_destroy(model);

    // Load model back
    printf("📂 Loading model from '%s'...\n", filepath);
    psam_model_t* loaded = psam_load(filepath);

    if (!loaded) {
        fprintf(stderr, "❌ Load failed\n");
        return 1;
    }

    printf("✓ Model loaded!\n\n");

    // Verify loaded model works
    printf("🔮 Testing loaded model...\n");
    num_preds = psam_predict(loaded, context, context_len, predictions, TOP_K);

    if (num_preds > 0) {
        printf("✓ Loaded model works! First prediction: Token %u\n\n",
               predictions[0].token_id);
    }

    // Cleanup
    psam_destroy(loaded);

    printf("🎉 Example complete!\n");
    printf("\n");
    printf("libpsam version: %s\n", psam_version());

    return 0;
}
