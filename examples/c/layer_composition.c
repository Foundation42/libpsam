/**
 * Layer composition example for libpsam
 *
 * Demonstrates domain adaptation through hot-swappable layers
 */

#include <stdio.h>
#include <stdlib.h>
#include <psam.h>

int main() {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║          libpsam - Layer Composition Example              ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    enum { VOCAB_SIZE = 100, WINDOW = 8, TOP_K = 5 };

    // Create base model (general domain)
    printf("📦 Creating base model (general domain)...\n");
    psam_model_t* base = psam_create(VOCAB_SIZE, WINDOW, TOP_K);

    // General text: "the cat sat on the mat"
    uint32_t general_tokens[] = {1, 2, 3, 4, 1, 5};
    psam_train_batch(base, general_tokens, 6);
    psam_finalize_training(base);
    printf("✓ Base model trained\n\n");

    // Create medical domain model
    printf("📦 Creating medical domain model...\n");
    psam_model_t* medical = psam_create(VOCAB_SIZE, WINDOW, TOP_K);

    // Medical text: "the patient has acute pain"
    uint32_t medical_tokens[] = {1, 10, 11, 12, 13};
    psam_train_batch(medical, medical_tokens, 5);
    psam_finalize_training(medical);
    printf("✓ Medical model trained\n\n");

    // Create legal domain model
    printf("📦 Creating legal domain model...\n");
    psam_model_t* legal = psam_create(VOCAB_SIZE, WINDOW, TOP_K);

    // Legal text: "the plaintiff claims damages"
    uint32_t legal_tokens[] = {1, 20, 21, 22};
    psam_train_batch(legal, legal_tokens, 4);
    psam_finalize_training(legal);
    printf("✓ Legal model trained\n\n");

    // Build layered composite
    printf("🧱 Building layered composite...\n");
    psam_composite_t* layered = psam_create_layered(base);
    if (!layered) {
        fprintf(stderr, "❌ Failed to create layered composite\n");
        goto cleanup;
    }

    // Test context
    uint32_t context[] = {1};  // "the"
    size_t context_len = 1;

    // Predictions with base model only
    printf("🔮 Predictions (base model only):\n");
    psam_prediction_t predictions[TOP_K];
    int num_preds = psam_predict(base, context, context_len, predictions, TOP_K);

    for (int i = 0; i < num_preds; i++) {
        printf("   %d. Token %u (%.3f)\n", i + 1,
               predictions[i].token, predictions[i].score);
    }
    printf("\n");

    // Add medical layer
    printf("➕ Adding medical layer (weight: 1.5)...\n");
    psam_error_t err = psam_composite_add_layer(layered, "medical", medical, 1.5f);
    if (err != PSAM_OK) {
        fprintf(stderr, "❌ Failed to add layer: %s\n", psam_error_string(err));
        goto cleanup;
    }

    printf("🔮 Predictions (base + medical):\n");
    num_preds = psam_composite_predict(layered, context, context_len, predictions, TOP_K);

    for (int i = 0; i < num_preds; i++) {
        printf("   %d. Token %u (%.3f)\n", i + 1,
               predictions[i].token, predictions[i].score);
    }
    printf("\n");

    // Update medical layer weight
    printf("⚙️  Updating medical layer weight to 2.0...\n");
    err = psam_composite_update_layer_weight(layered, "medical", 2.0f);
    if (err != PSAM_OK) {
        fprintf(stderr, "❌ Failed to update weight: %s\n", psam_error_string(err));
        goto cleanup;
    }

    printf("🔮 Predictions (base + medical 2.0×):\n");
    num_preds = psam_composite_predict(layered, context, context_len, predictions, TOP_K);

    for (int i = 0; i < num_preds; i++) {
        printf("   %d. Token %u (%.3f)\n", i + 1,
               predictions[i].token, predictions[i].score);
    }
    printf("\n");

    // Remove medical, add legal
    printf("🔄 Switching to legal domain...\n");
    err = psam_composite_remove_layer(layered, "medical");
    if (err != PSAM_OK) {
        fprintf(stderr, "❌ Failed to remove layer: %s\n", psam_error_string(err));
        goto cleanup;
    }

    err = psam_composite_add_layer(layered, "legal", legal, 1.5f);
    if (err != PSAM_OK) {
        fprintf(stderr, "❌ Failed to add layer: %s\n", psam_error_string(err));
        goto cleanup;
    }

    printf("🔮 Predictions (base + legal):\n");
    num_preds = psam_composite_predict(layered, context, context_len, predictions, TOP_K);

    for (int i = 0; i < num_preds; i++) {
        printf("   %d. Token %u (%.3f)\n", i + 1,
               predictions[i].token, predictions[i].score);
    }
    printf("\n");

    // List active layers
    psam_composite_layer_info_t infos[10];
    int num_layers = psam_composite_list_layers(layered, infos, 10);

    if (num_layers < 0) {
        fprintf(stderr, "❌ Failed to list layers: %s\n", psam_error_string(num_layers));
    } else {
        printf("📋 Active layers (%d):\n", num_layers);
        for (int i = 0; i < num_layers; i++) {
            printf("   - %s (weight %.2f)\n", infos[i].id, infos[i].weight);
        }
        printf("\n");
    }

    printf("🎉 Layer composition demo complete!\n");

cleanup:
    psam_composite_destroy(layered);
    psam_destroy(base);
    psam_destroy(medical);
    psam_destroy(legal);

    return 0;
}
