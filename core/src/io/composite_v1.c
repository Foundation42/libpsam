/**
 * composite_v1.c - Aligned composite persistence (v1 MVP)
 *
 * Simple JSON-based format with mandatory vocabulary alignment.
 * No checksums, no headers - just the essential data for save/load/predict.
 */

#define _POSIX_C_SOURCE 200809L

#include "../../include/psam_composite.h"
#include "../../include/psam_vocab_alignment.h"
#include "../../include/psam.h"
#include "../../include/psam_export.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>

/* ========== Simple JSON Writer ========== */

static void json_write_string(FILE* f, const char* key, const char* value, bool comma) {
    fprintf(f, "  \"%s\": \"%s\"%s\n", key, value, comma ? "," : "");
}

static void json_write_uint(FILE* f, const char* key, uint32_t value, bool comma) {
    fprintf(f, "  \"%s\": %u%s\n", key, value, comma ? "," : "");
}

static void json_write_float(FILE* f, const char* key, float value, bool comma) {
    fprintf(f, "  \"%s\": %.6f%s\n", key, value, comma ? "," : "");
}

/* ========== Binary Map I/O ========== */

/**
 * Write local-to-unified dense map (*.l2u.u32)
 * Format: raw little-endian uint32_t array [local_vocab_size]
 */
static int write_l2u_map(const char* path, const uint32_t* l2u, uint32_t local_vocab_size) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "ERROR: Failed to open %s for writing: %s\n", path, strerror(errno));
        return -1;
    }

    size_t written = fwrite(l2u, sizeof(uint32_t), local_vocab_size, f);
    fclose(f);

    if (written != local_vocab_size) {
        fprintf(stderr, "ERROR: Failed to write complete l2u map to %s\n", path);
        return -1;
    }

    return 0;
}

/**
 * Write unified-to-local sparse map (*.u2l.pairs)
 * Format: sorted pairs of (uint32_t unified_id, uint32_t local_id)
 */
static int write_u2l_map(const char* path, const uint32_t* pairs, uint32_t pair_count) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "ERROR: Failed to open %s for writing: %s\n", path, strerror(errno));
        return -1;
    }

    size_t written = fwrite(pairs, sizeof(uint32_t), pair_count * 2, f);
    fclose(f);

    if (written != pair_count * 2) {
        fprintf(stderr, "ERROR: Failed to write complete u2l map to %s\n", path);
        return -1;
    }

    return 0;
}

/**
 * Read local-to-unified dense map (*.l2u.u32)
 */
static uint32_t* read_l2u_map(const char* path, uint32_t expected_size, uint32_t* out_size) PSAM_UNUSED;
static uint32_t* read_l2u_map(const char* path, uint32_t expected_size, uint32_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Failed to open %s: %s\n", path, strerror(errno));
        return NULL;
    }

    /* Check file size */
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size != (long)(expected_size * sizeof(uint32_t))) {
        fprintf(stderr, "ERROR: %s size mismatch (expected %u bytes, got %ld)\n",
                path, expected_size * 4, file_size);
        fclose(f);
        return NULL;
    }

    uint32_t* map = malloc(file_size);
    if (!map) {
        fclose(f);
        return NULL;
    }

    size_t read = fread(map, 1, file_size, f);
    fclose(f);

    if (read != (size_t)file_size) {
        fprintf(stderr, "ERROR: Failed to read complete l2u map from %s\n", path);
        free(map);
        return NULL;
    }

    *out_size = expected_size;
    return map;
}

/**
 * Read unified-to-local sparse map (*.u2l.pairs)
 */
static uint32_t* read_u2l_map(const char* path, uint32_t* out_pair_count) PSAM_UNUSED;
static uint32_t* read_u2l_map(const char* path, uint32_t* out_pair_count) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Failed to open %s: %s\n", path, strerror(errno));
        return NULL;
    }

    /* Check file size is multiple of 8 (two uint32_t per pair) */
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size % 8 != 0) {
        fprintf(stderr, "ERROR: %s size (%ld) is not a multiple of 8\n", path, file_size);
        fclose(f);
        return NULL;
    }

    uint32_t* pairs = malloc(file_size);
    if (!pairs) {
        fclose(f);
        return NULL;
    }

    size_t read = fread(pairs, 1, file_size, f);
    fclose(f);

    if (read != (size_t)file_size) {
        fprintf(stderr, "ERROR: Failed to read complete u2l map from %s\n", path);
        free(pairs);
        return NULL;
    }

    *out_pair_count = file_size / 8;
    return pairs;
}

/* ========== Composite Save (MVP) ========== */

/**
 * Save aligned composite to .psamc v1 JSON format
 *
 * Assumes composite has alignment data already built and ready to persist.
 * Writes:
 * - .psamc JSON manifest
 * - *.l2u.u32 map files (one per layer)
 * - *.u2l.pairs map files (one per layer)
 */
int psam_composite_save_v1(
    const char* psamc_path,
    const char* created_by,
    const char* unified_vocab_path,
    psam_unknown_policy_t unknown_policy,
    psam_coverage_rule_t coverage_rule,
    const psamc_sampler_defaults_t* sampler,
    size_t layer_count,
    const char** layer_ids,
    const char** layer_model_paths,
    const float* layer_weights,
    const float* layer_biases,
    const uint32_t* layer_local_vocab_sizes,
    const uint32_t** layer_l2u_maps,
    const uint32_t** layer_u2l_pairs,
    const uint32_t* layer_u2l_pair_counts,
    const char** layer_l2u_paths,
    const char** layer_u2l_paths
) {
    FILE* f = fopen(psamc_path, "w");
    if (!f) {
        fprintf(stderr, "ERROR: Failed to create %s: %s\n", psamc_path, strerror(errno));
        return -1;
    }

    /* Write JSON header */
    fprintf(f, "{\n");
    json_write_uint(f, "version", 1, true);
    json_write_string(f, "created_by", created_by ? created_by : "libpsam", true);

    /* Sampler defaults */
    fprintf(f, "  \"sampler\": {\n");
    const char* transform_str = "zscore";
    if (sampler && sampler->logit_transform == PSAM_LOGIT_RAW) transform_str = "raw";
    else if (sampler && sampler->logit_transform == PSAM_LOGIT_CALIBRATED) transform_str = "calibrated";
    fprintf(f, "    \"logit_transform\": \"%s\",\n", transform_str);
    fprintf(f, "    \"temperature\": %.6f,\n", sampler ? sampler->temperature : 1.0f);
    fprintf(f, "    \"top_k\": %d,\n", sampler ? sampler->top_k : 50);
    fprintf(f, "    \"top_p\": %.6f,\n", sampler ? sampler->top_p : 0.95f);
    fprintf(f, "    \"seed\": %lu\n", (unsigned long)(sampler ? sampler->seed : 42));
    fprintf(f, "  },\n");

    /* Unified vocab */
    fprintf(f, "  \"unified_vocab\": { \"path\": \"%s\" },\n", unified_vocab_path);

    /* Unknown policy */
    const char* policy_str = (unknown_policy == PSAM_UNKNOWN_MAP_UNK) ? "unk" : "skip";
    json_write_string(f, "unknown_policy", policy_str, true);

    /* Coverage weight */
    const char* coverage_str = "none";
    if (coverage_rule == PSAM_COVER_LINEAR) coverage_str = "linear";
    else if (coverage_rule == PSAM_COVER_SQRT) coverage_str = "sqrt";
    json_write_string(f, "coverage_weight", coverage_str, true);

    /* Layers */
    fprintf(f, "  \"layers\": [\n");
    for (size_t i = 0; i < layer_count; ++i) {
        fprintf(f, "    {\n");
        json_write_string(f, "id", layer_ids[i], true);
        json_write_string(f, "path", layer_model_paths[i], true);
        json_write_float(f, "weight", layer_weights[i], true);
        json_write_float(f, "bias", layer_biases[i], true);
        json_write_uint(f, "local_vocab_size", layer_local_vocab_sizes[i], true);
        json_write_string(f, "l2u_path", layer_l2u_paths[i], true);
        json_write_string(f, "u2l_path", layer_u2l_paths[i], false);
        fprintf(f, "    }%s\n", (i + 1 < layer_count) ? "," : "");

        /* Write binary map files */
        if (write_l2u_map(layer_l2u_paths[i], layer_l2u_maps[i], layer_local_vocab_sizes[i]) != 0) {
            fclose(f);
            return -1;
        }
        if (write_u2l_map(layer_u2l_paths[i], layer_u2l_pairs[i], layer_u2l_pair_counts[i]) != 0) {
            fclose(f);
            return -1;
        }
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");

    fclose(f);
    return 0;
}

/* ========== Composite Load (MVP) ========== */

/* TODO: Implement JSON parser and loader */
/* This will be implemented next after we test the save path */
