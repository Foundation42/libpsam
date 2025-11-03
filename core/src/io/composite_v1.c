/**
 * composite_v1.c - Aligned composite persistence (v1)
 *
 * Binary .psamc writer that persists vocabulary alignment metadata,
 * SHA-256 digests, and sampler defaults alongside layered topology data.
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
    const char** layer_u2l_paths,
    uint32_t unified_vocab_size
) {
    (void)sampler; /* Sampler persistence will be added in a later revision */

    if (!psamc_path || layer_count == 0 || !layer_ids || !layer_model_paths ||
        !layer_weights || !layer_biases || !layer_local_vocab_sizes ||
        !layer_l2u_maps || !layer_u2l_pairs || !layer_u2l_pair_counts ||
        !layer_l2u_paths || !layer_u2l_paths) {
        fprintf(stderr, "ERROR: psam_composite_save_v1 received invalid arguments\n");
        return -1;
    }

    for (size_t i = 0; i < layer_count; ++i) {
        if (!layer_l2u_maps[i] || !layer_u2l_pairs[i] ||
            !layer_l2u_paths[i] || !layer_u2l_paths[i]) {
            fprintf(stderr, "ERROR: Layer %zu missing alignment buffers\n", i);
            return -1;
        }
    }

    /* Emit alignment map binaries */
    for (size_t i = 0; i < layer_count; ++i) {
        if (write_l2u_map(layer_l2u_paths[i], layer_l2u_maps[i], layer_local_vocab_sizes[i]) != 0) {
            return -1;
        }
        if (write_u2l_map(layer_u2l_paths[i], layer_u2l_pairs[i], layer_u2l_pair_counts[i]) != 0) {
            return -1;
        }
    }

    int rc = -1;
    psamc_manifest_t manifest = {0};
    psamc_topology_t topology = {0};
    psam_alignment_info_t alignment = {0};

    manifest.num_references = (uint32_t)layer_count;
    manifest.refs = calloc(layer_count, sizeof(psamc_model_ref_t));
    if (!manifest.refs) {
        goto cleanup;
    }

    for (size_t i = 0; i < layer_count; ++i) {
        const char* model_path = layer_model_paths[i];
        struct stat st = {0};
        if (stat(model_path, &st) != 0) {
            fprintf(stderr, "ERROR: Failed to stat model '%s': %s\n", model_path, strerror(errno));
            goto cleanup;
        }

        psamc_model_ref_t* ref = &manifest.refs[i];
        snprintf(ref->url, PSAMC_MAX_URL_LENGTH, "%s", model_path);
        ref->size = (uint64_t)st.st_size;
        if (psamc_sha256_file(model_path, &ref->sha256) != 0) {
            fprintf(stderr, "ERROR: Failed to hash model '%s'\n", model_path);
            goto cleanup;
        }
        if (layer_ids[i]) {
            snprintf(ref->model_id, PSAMC_MAX_MODEL_ID, "%s", layer_ids[i]);
        }
    }

    manifest.created_timestamp = (uint64_t)time(NULL);
    snprintf(manifest.created_by, PSAMC_CREATED_BY_MAX, "%s", created_by && created_by[0] ? created_by : "libpsam");

    topology.base_weight = layer_weights[0];
    topology.base_ref_index = 0;
    if (layer_count > 1) {
        topology.layer_count = (uint32_t)(layer_count - 1);
        topology.layers = calloc(topology.layer_count, sizeof(psamc_layer_entry_t));
        if (!topology.layers) {
            goto cleanup;
        }
        for (uint32_t i = 0; i < topology.layer_count; ++i) {
            const char* id = layer_ids[i + 1];
            psamc_layer_entry_t* entry = &topology.layers[i];
            if (id) {
                snprintf(entry->layer_id, PSAM_LAYER_ID_MAX, "%s", id);
            }
            entry->weight = layer_weights[i + 1];
            entry->bias = layer_biases[i + 1];
            entry->ref_index = i + 1;
        }
    }

    alignment.unified_vocab_size = unified_vocab_size;
    alignment.unified_unk = unified_vocab_size ? unified_vocab_size - 1 : 0;
    alignment.unknown_policy = unknown_policy;
    alignment.coverage_rule = coverage_rule;
    alignment.layer_count = (uint32_t)layer_count;
    alignment.layers = calloc(layer_count, sizeof(psam_layer_map_t));
    if (!alignment.layers) {
        goto cleanup;
    }

    if (unified_vocab_path && unified_vocab_path[0] != '\0') {
        struct stat vocab_st = {0};
        if (stat(unified_vocab_path, &vocab_st) == 0) {
            alignment.unified_vocab_size_bytes = (uint64_t)vocab_st.st_size;
        }
        snprintf(alignment.unified_vocab_path, PSAMC_MAX_URL_LENGTH, "%s", unified_vocab_path);
        if (psamc_sha256_file(unified_vocab_path, &alignment.unified_vocab_hash) != 0) {
            fprintf(stderr, "ERROR: Failed to hash unified vocab '%s'\n", unified_vocab_path);
            goto cleanup;
        }
    }

    for (size_t i = 0; i < layer_count; ++i) {
        psam_layer_map_t* map = &alignment.layers[i];
        if (layer_ids[i]) {
            snprintf(map->layer_id, PSAM_LAYER_ID_MAX, "%s", layer_ids[i]);
        }
        map->local_vocab_size = layer_local_vocab_sizes[i];
        map->local_unk = layer_local_vocab_sizes[i] ? layer_local_vocab_sizes[i] - 1 : 0;
        map->u2l_pairs_count = layer_u2l_pair_counts[i];
        map->coverage = (unified_vocab_size > 0)
            ? (float)layer_u2l_pair_counts[i] / (float)unified_vocab_size
            : 0.0f;

        snprintf(map->l2u_path, PSAMC_MAX_URL_LENGTH, "%s", layer_l2u_paths[i]);
        snprintf(map->u2l_path, PSAMC_MAX_URL_LENGTH, "%s", layer_u2l_paths[i]);

        struct stat map_stat = {0};
        if (stat(layer_l2u_paths[i], &map_stat) == 0) {
            map->l2u_size_bytes = (uint64_t)map_stat.st_size;
        }
        if (psamc_sha256_file(layer_l2u_paths[i], &map->l2u_hash) != 0) {
            fprintf(stderr, "ERROR: Failed to hash map '%s'\n", layer_l2u_paths[i]);
            goto cleanup;
        }

        if (stat(layer_u2l_paths[i], &map_stat) == 0) {
            map->u2l_size_bytes = (uint64_t)map_stat.st_size;
        }
        if (psamc_sha256_file(layer_u2l_paths[i], &map->u2l_hash) != 0) {
            fprintf(stderr, "ERROR: Failed to hash map '%s'\n", layer_u2l_paths[i]);
            goto cleanup;
        }
    }

    psamc_hyperparams_t hyperparams = PSAMC_PRESET_BALANCED_CONFIG;

    if (psamc_save(psamc_path, NULL, &hyperparams, &manifest, &topology, &alignment) != 0) {
        goto cleanup;
    }

    rc = 0;

cleanup:
    free(manifest.refs);
    manifest.refs = NULL;
    free(topology.layers);
    topology.layers = NULL;
    free(alignment.layers);
    alignment.layers = NULL;
    return rc;
}

/* ========== Composite Load (MVP) ========== */

/* TODO: Implement binary loader for v1 composites */
