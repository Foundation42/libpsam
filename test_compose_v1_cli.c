/**
 * test_compose_v1_cli.c - Smoke test for v1 aligned composite CLI
 *
 * Quick integration test to validate the end-to-end v1 compose workflow:
 * 1. Prepare test vocabs
 * 2. Run compose command via CLI
 * 3. Verify output files exist
 * 4. Check JSON schema
 *
 * This is a minimal smoke test - comprehensive testing with Shakespeare models
 * should be done manually.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static int run_command(const char* cmd) {
    printf("Running: %s\n", cmd);
    return system(cmd);
}

int main(void) {
    printf("=== V1 Aligned Composite CLI Smoke Test ===\n\n");

    /* Use existing Shakespeare test data if available */
    const char* hamlet_model = "corpora/text/Folger/models/hamlet.psam";
    const char* macbeth_model = "corpora/text/Folger/models/macbeth.psam";
    const char* hamlet_vocab = "corpora/text/Folger/models/hamlet.tsv";
    const char* macbeth_vocab = "corpora/text/Folger/models/macbeth.tsv";
    const char* unified_vocab = "corpora/text/Folger/unified_vocab.tsv";

    /* Check if test data exists */
    if (!file_exists(hamlet_model) || !file_exists(macbeth_model)) {
        printf("⚠️  Shakespeare test models not found.\n");
        printf("    Skipping CLI integration test.\n");
        printf("    (Models should be at: %s, %s)\n", hamlet_model, macbeth_model);
        printf("\n✓ Test skipped (not a failure - test data not available)\n");
        return 0;
    }

    if (!file_exists(unified_vocab)) {
        printf("⚠️  Unified vocab not found at: %s\n", unified_vocab);
        printf("    Skipping CLI integration test.\n");
        printf("\n✓ Test skipped (not a failure - test data not available)\n");
        return 0;
    }

    printf("Found test data:\n");
    printf("  Hamlet model: %s\n", hamlet_model);
    printf("  Macbeth model: %s\n", macbeth_model);
    printf("  Unified vocab: %s\n\n", unified_vocab);

    /* Create output directory */
    system("mkdir -p test_output/maps");

    /* Run compose command */
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
        "./build/psam compose "
        "--out test_output/tragedies_v1.psamc "
        "--unified-vocab %s "
        "--unknown-policy unk "
        "--coverage-weight none "
        "--layer %s --vocab %s --weight 1.0 --bias 0.0 "
        "--layer %s --vocab %s --weight 0.8 --bias 0.1",
        unified_vocab,
        hamlet_model, hamlet_vocab,
        macbeth_model, macbeth_vocab);

    int rc = run_command(cmd);
    if (rc != 0) {
        fprintf(stderr, "❌ Compose command failed with exit code %d\n", rc);
        return 1;
    }

    printf("\n");

    /* Verify output files */
    printf("Verifying output files...\n");

    const char* expected_files[] = {
        "test_output/tragedies_v1.psamc",
        "maps/hamlet.l2u.u32",
        "maps/hamlet.u2l.pairs",
        "maps/macbeth.l2u.u32",
        "maps/macbeth.u2l.pairs",
        NULL
    };

    int all_exist = 1;
    for (int i = 0; expected_files[i]; ++i) {
        if (file_exists(expected_files[i])) {
            printf("  ✓ %s\n", expected_files[i]);
        } else {
            printf("  ❌ MISSING: %s\n", expected_files[i]);
            all_exist = 0;
        }
    }

    if (!all_exist) {
        fprintf(stderr, "\n❌ Some output files are missing!\n");
        return 1;
    }

    /* Inspect binary contents via CLI */
    printf("\nInspecting composite metadata...\n");
    if (run_command("./build/psam inspect --model test_output/tragedies_v1.psamc > test_output/inspect_tragedies.json") != 0) {
        fprintf(stderr, "❌ psam inspect failed\n");
        return 1;
    }

    FILE* f = fopen("test_output/inspect_tragedies.json", "r");
    if (!f) {
        fprintf(stderr, "❌ Unable to read inspect output\n");
        return 1;
    }

    char line[1024];
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        fprintf(stderr, "❌ Inspect output empty\n");
        return 1;
    }
    fclose(f);

    if (!strstr(line, "\"type\":\"psamc\"")) {
        fprintf(stderr, "❌ Inspect output missing psamc type\n");
        return 1;
    }
    if (!strstr(line, "\"layers\":")) {
        fprintf(stderr, "❌ Inspect output missing layers field\n");
        return 1;
    }

    printf("  ✓ inspect reports psamc metadata\n");

    printf("\nRunning predict with composite...\n");
    rc = run_command("./build/psam predict --model test_output/tragedies_v1.psamc --ctx-ids 0,1,2 --top_k 5 > /dev/null");
    if (rc != 0) {
        fprintf(stderr, "❌ psam predict failed with exit code %d\n", rc);
        return 1;
    }
    printf("  ✓ predict command succeeded\n");

    printf("\n🎉 V1 Aligned Composite CLI smoke test PASSED!\n");
    printf("\nGenerated files:\n");
    printf("  Composite: test_output/tragedies_v1.psamc\n");
    printf("  Maps: test_output/maps/*.{l2u.u32,u2l.pairs}\n");
    printf("\nNext: Try prediction with the composite:\n");
    printf("  ./build/psam predict --model test_output/tragedies_v1.psamc \\\n");
    printf("    --prompt \"When shall we three meet again\" --top_k 10\n");

    return 0;
}
