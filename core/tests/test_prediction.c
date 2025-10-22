/**
 * Test for prediction accuracy
 *
 * This test reproduces the demo scenario to debug incorrect predictions.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../include/psam.h"

// Simple tokenizer for testing
typedef struct {
    char** words;
    int count;
    int capacity;
} vocab_t;

vocab_t* vocab_create() {
    vocab_t* vocab = malloc(sizeof(vocab_t));
    vocab->words = malloc(sizeof(char*) * 100);
    vocab->count = 0;
    vocab->capacity = 100;
    return vocab;
}

int vocab_get_or_add(vocab_t* vocab, const char* word) {
    // Check if word exists
    for (int i = 0; i < vocab->count; i++) {
        if (strcmp(vocab->words[i], word) == 0) {
            return i;
        }
    }

    // Add new word
    vocab->words[vocab->count] = strdup(word);
    return vocab->count++;
}

const char* vocab_get_word(vocab_t* vocab, int id) {
    if (id < 0 || id >= vocab->count) return "<UNK>";
    return vocab->words[id];
}

void vocab_destroy(vocab_t* vocab) {
    for (int i = 0; i < vocab->count; i++) {
        free(vocab->words[i]);
    }
    free(vocab->words);
    free(vocab);
}

// Tokenize text into array of token IDs
uint32_t* tokenize(const char* text, vocab_t* vocab, size_t* out_len) {
    uint32_t* tokens = malloc(sizeof(uint32_t) * 1000);
    size_t count = 0;

    char* text_copy = strdup(text);
    char* token = strtok(text_copy, " \t\n");

    while (token != NULL) {
        // Handle punctuation
        size_t len = strlen(token);
        if (len > 1 && (token[len-1] == '.' || token[len-1] == ',' ||
                        token[len-1] == '!' || token[len-1] == '?')) {
            char punct[2] = {token[len-1], '\0'};
            token[len-1] = '\0';
            tokens[count++] = vocab_get_or_add(vocab, token);
            tokens[count++] = vocab_get_or_add(vocab, punct);
        } else {
            tokens[count++] = vocab_get_or_add(vocab, token);
        }

        token = strtok(NULL, " \t\n");
    }

    free(text_copy);
    *out_len = count;
    return tokens;
}

int main() {
    printf("=== PSAM Prediction Test ===\n\n");

    // Create vocabulary
    vocab_t* vocab = vocab_create();

    // Training text - same as demo
    const char* training_text =
        "the cat sat on the mat. "
        "the dog sat on the rug. "
        "the bird sat on the branch. "
        "the frog sat on the log.";

    printf("Training text:\n%s\n\n", training_text);

    // Tokenize training text
    size_t num_tokens;
    uint32_t* tokens = tokenize(training_text, vocab, &num_tokens);

    printf("Tokenized (%zu tokens):\n", num_tokens);
    for (size_t i = 0; i < num_tokens; i++) {
        printf("%s ", vocab_get_word(vocab, tokens[i]));
    }
    printf("\n\n");

    // Create model with same parameters as demo
    psam_config_t config = {
        .vocab_size = vocab->count,
        .window = 8,
        .top_k = 32,
        .alpha = 0.1,
        .min_evidence = 1,
        .enable_idf = true,
        .enable_ppmi = true,
        .edge_dropout = 0.0
    };

    printf("Creating model: vocab=%d, window=%d, top_k=%d\n",
           config.vocab_size, config.window, config.top_k);
    printf("                alpha=%.2f, min_evidence=%.0f, IDF=%d, PPMI=%d\n\n",
           config.alpha, config.min_evidence, config.enable_idf, config.enable_ppmi);

    psam_model_t* model = psam_create_with_config(&config);
    assert(model != NULL);

    // Train
    printf("Training...\n");
    psam_error_t err = psam_train_batch(model, tokens, num_tokens);
    assert(err == PSAM_OK);

    err = psam_finalize_training(model);
    assert(err == PSAM_OK);

    // Get stats
    psam_stats_t stats;
    psam_get_stats(model, &stats);
    printf("Stats: vocab=%d, edges=%lu, tokens=%lu\n\n",
           stats.vocab_size, stats.edge_count, stats.total_tokens);

    // Test prediction: "the dog sat on the" -> should predict "rug"
    const char* test_contexts[] = {
        "the dog sat on the",
        "the cat sat on the",
        "the bird sat on the",
        "the frog sat on the"
    };

    const char* expected[] = {
        "rug",
        "mat",
        "branch",
        "log"
    };

    int pass_count = 0;
    int fail_count = 0;

    for (int test = 0; test < 4; test++) {
        printf("Test %d: \"%s\" -> ?\n", test + 1, test_contexts[test]);

        // Tokenize context
        size_t context_len;
        uint32_t* context = tokenize(test_contexts[test], vocab, &context_len);

        printf("  Context tokens: ");
        for (size_t i = 0; i < context_len; i++) {
            printf("%s ", vocab_get_word(vocab, context[i]));
        }
        printf("(len=%zu)\n", context_len);

        // Predict (also get all scores for debugging)
        psam_prediction_t predictions[10];
        int num_preds = psam_predict(model, context, context_len, predictions, 10);

        // Also get scores for the expected word
        int expected_id = vocab_get_or_add(vocab, expected[test]);
        int expected_rank = -1;
        float expected_score = 0.0f;
        for (int i = 0; i < num_preds; i++) {
            if (predictions[i].token == (uint32_t)expected_id) {
                expected_rank = i + 1;
                expected_score = predictions[i].score;
                break;
            }
        }

        printf("  Expected \"%s\" (id=%d) rank=%d score=%.3f\n",
               expected[test], expected_id, expected_rank, expected_score);
        printf("  Predictions:\n");
        if (num_preds <= 0) {
            printf("    (no predictions)\n");
            fail_count++;
        } else {
            for (int i = 0; i < num_preds && i < 5; i++) {
                const char* word = vocab_get_word(vocab, predictions[i].token);
                printf("    %d. %s (score=%.3f, prob=%.3f)%s\n",
                       i + 1, word,
                       predictions[i].score,
                       predictions[i].calibrated_prob,
                       (i == 0) ? " <- TOP" : "");
            }

            // Check if top prediction matches expected
            const char* top_word = vocab_get_word(vocab, predictions[0].token);
            if (strcmp(top_word, expected[test]) == 0) {
                printf("  ✓ PASS: Predicted \"%s\" as expected\n", top_word);
                pass_count++;
            } else {
                printf("  ✗ FAIL: Expected \"%s\" but got \"%s\"\n", expected[test], top_word);
                fail_count++;
            }
        }
        printf("\n");

        free(context);
    }

    // Summary
    printf("========================================\n");
    printf("Results: %d PASS, %d FAIL\n", pass_count, fail_count);

    // Cleanup
    psam_destroy(model);
    free(tokens);
    vocab_destroy(vocab);

    return (fail_count > 0) ? 1 : 0;
}
