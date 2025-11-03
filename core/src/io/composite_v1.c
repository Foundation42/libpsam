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
#include <ctype.h>
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

/* ========== Loader Helpers ========== */

static bool path_is_absolute(const char* path) {
    if (!path || path[0] == '\0') {
        return false;
    }
#ifdef _WIN32
    if (path[0] == '/' || path[0] == '\\') {
        return true;
    }
    if (isalpha((unsigned char)path[0]) && path[1] == ':' && (path[2] == '/' || path[2] == '\\')) {
        return true;
    }
    return false;
#else
    return path[0] == '/';
#endif
}

static char* resolve_reference_path(const char* composite_path, const char* ref_path) {
    if (!ref_path || ref_path[0] == '\0') {
        return NULL;
    }

    if (path_is_absolute(ref_path)) {
        return strdup(ref_path);
    }

    if (!composite_path) {
        return strdup(ref_path);
    }

    const char* last_slash = strrchr(composite_path, '/');
#ifdef _WIN32
    const char* last_backslash = strrchr(composite_path, '\\');
    if (!last_slash || (last_backslash && last_backslash > last_slash)) {
        last_slash = last_backslash;
    }
#endif

    size_t dir_len = last_slash ? (size_t)(last_slash - composite_path + 1) : 0;
    size_t ref_len = strlen(ref_path);
    char* resolved = malloc(dir_len + ref_len + 1);
    if (!resolved) {
        return NULL;
    }

    if (dir_len > 0) {
        memcpy(resolved, composite_path, dir_len);
    }
    memcpy(resolved + dir_len, ref_path, ref_len + 1);
    return resolved;
}

static size_t extract_directory(const char* path, char* out_dir, size_t out_len) {
    if (!out_dir || out_len == 0) {
        return 0;
    }
    out_dir[0] = '\0';
    if (!path) {
        return 0;
    }

    const char* last_slash = strrchr(path, '/');
#ifdef _WIN32
    const char* last_backslash = strrchr(path, '\\');
    if (!last_slash || (last_backslash && last_backslash > last_slash)) {
        last_slash = last_backslash;
    }
#endif

    if (!last_slash) {
        return 0;
    }

    size_t len = (size_t)(last_slash - path + 1); /* include separator */
    if (len >= out_len) {
        len = out_len - 1;
    }
    memcpy(out_dir, path, len);
    out_dir[len] = '\0';
    return len;
}

static void normalize_reference_path(
    const char* base_dir,
    const char* ref_path,
    char* out_path,
    size_t out_len
) {
    if (!out_path || out_len == 0) {
        return;
    }

    out_path[0] = '\0';

    if (!ref_path || ref_path[0] == '\0') {
        return;
    }

    if (!base_dir || base_dir[0] == '\0' || path_is_absolute(ref_path)) {
        snprintf(out_path, out_len, "%s", ref_path);
        return;
    }

    size_t base_len = strlen(base_dir);
    if (base_len > 0 && strncmp(ref_path, base_dir, base_len) == 0) {
        const char* trimmed = ref_path + base_len;
        while (*trimmed == '/' || *trimmed == '\\') {
            ++trimmed;
        }
        if (*trimmed == '\0') {
            snprintf(out_path, out_len, ".");
        } else {
            snprintf(out_path, out_len, "%s", trimmed);
        }
        return;
    }

    if (ref_path[0] == '.' && (ref_path[1] == '/' || ref_path[1] == '\\')) {
        ref_path += 2;
        while (*ref_path == '/' || *ref_path == '\\') {
            ++ref_path;
        }
    }

    snprintf(out_path, out_len, "%s", ref_path);
}

static bool hash_is_nonzero(const sha256_hash_t* hash) {
    if (!hash) {
        return false;
    }
    for (size_t i = 0; i < PSAMC_SOURCE_HASH_SIZE; ++i) {
        if (hash->hash[i] != 0) {
            return true;
        }
    }
    return false;
}

static int verify_external_file(
    const char* path,
    uint64_t expected_size,
    const sha256_hash_t* expected_hash,
    bool verify_integrity,
    bool log_errors
) {
    struct stat st;
    if (stat(path, &st) != 0) {
        if (log_errors) {
            fprintf(stderr, "ERROR: Failed to stat '%s': %s\n", path, strerror(errno));
        }
        return -1;
    }

    if (expected_size > 0 && (uint64_t)st.st_size != expected_size) {
        if (log_errors) {
            fprintf(stderr, "ERROR: Size mismatch for '%s' (expected %llu, got %llu)\n",
                    path,
                    (unsigned long long)expected_size,
                    (unsigned long long)st.st_size);
        }
        return -1;
    }

    if (verify_integrity && expected_hash && hash_is_nonzero(expected_hash)) {
        sha256_hash_t actual = {0};
        if (psamc_sha256_file(path, &actual) != 0) {
            if (log_errors) {
                fprintf(stderr, "ERROR: Failed to hash '%s'\n", path);
            }
            return -1;
        }
        if (memcmp(actual.hash, expected_hash->hash, PSAMC_SOURCE_HASH_SIZE) != 0) {
            if (log_errors) {
                fprintf(stderr, "ERROR: SHA-256 mismatch for '%s'\n", path);
            }
            return -1;
        }
    }

    return 0;
}

static char** load_unified_vocab_tokens(const char* path, uint32_t expected_size) {
    if (!path) {
        return NULL;
    }

    if (expected_size == 0) {
        return NULL;
    }

    FILE* f = fopen(path, "r");
    if (!f) {
        return NULL;
    }

    char** tokens = calloc(expected_size, sizeof(char*));
    if (!tokens) {
        fclose(f);
        return NULL;
    }

    char line[8192];
    uint32_t count = 0;
    while (count < expected_size && fgets(line, sizeof(line), f)) {
        char* tab = strchr(line, '\t');
        if (!tab) {
            fprintf(stderr, "ERROR: Malformed vocab line: %s\n", line);
            goto error;
        }
        *tab = '\0';
        uint32_t id = 0;
        if (sscanf(line, "%u", &id) != 1 || id != count) {
            fprintf(stderr, "ERROR: Unexpected vocab id %u (expected %u)\n", id, count);
            goto error;
        }
        char* token = tab + 1;
        char* newline = strpbrk(token, "\r\n");
        if (newline) {
            *newline = '\0';
        }
        tokens[count] = strdup(token);
        if (!tokens[count]) {
            goto error;
        }
        count++;
    }

    fclose(f);

    if (count != expected_size) {
        fprintf(stderr, "ERROR: Unified vocab expected %u tokens, found %u\n", expected_size, count);
        goto error_cleanup;
    }

    return tokens;

error:
    fclose(f);
error_cleanup:
    for (uint32_t i = 0; i < expected_size; ++i) {
        free(tokens[i]);
    }
    free(tokens);
    return NULL;
}

static int load_layer_map(
    const char* composite_path,
    const psam_layer_map_t* disk_map,
    bool verify_integrity,
    psam_vocab_remap_t* out_remap
) {
    if (!disk_map || !out_remap) {
        return -1;
    }

    memset(out_remap, 0, sizeof(*out_remap));
    out_remap->local_vocab_size = disk_map->local_vocab_size;
    out_remap->coverage = disk_map->coverage;

    char* l2u_resolved = resolve_reference_path(composite_path, disk_map->l2u_path);
    const char* l2u_path = l2u_resolved ? l2u_resolved : disk_map->l2u_path;

    char* u2l_resolved = resolve_reference_path(composite_path, disk_map->u2l_path);
    const char* u2l_path = u2l_resolved ? u2l_resolved : disk_map->u2l_path;

    if (disk_map->local_vocab_size > 0) {
        if (!l2u_path) {
            fprintf(stderr, "ERROR: Missing l2u path for layer '%s'\n", disk_map->layer_id);
            goto fail;
        }

        if (verify_external_file(l2u_path, disk_map->l2u_size_bytes, &disk_map->l2u_hash, verify_integrity, false) != 0) {
            if (l2u_resolved) {
                free(l2u_resolved);
                l2u_resolved = NULL;
                l2u_path = disk_map->l2u_path;
                if (verify_external_file(l2u_path, disk_map->l2u_size_bytes, &disk_map->l2u_hash, verify_integrity, true) != 0) {
                    goto fail;
                }
            } else {
                goto fail;
            }
        }

        size_t expected_bytes = (size_t)disk_map->local_vocab_size * sizeof(uint32_t);
        FILE* f = fopen(l2u_path, "rb");
        if (!f) {
            fprintf(stderr, "ERROR: Failed to open %s: %s\n", l2u_path, strerror(errno));
            goto fail;
        }
        uint32_t* buffer = malloc(expected_bytes);
        if (!buffer) {
            fclose(f);
            goto fail;
        }
        size_t read = fread(buffer, 1, expected_bytes, f);
        fclose(f);
        if (read != expected_bytes) {
            fprintf(stderr, "ERROR: Failed to read complete l2u map from %s\n", l2u_path);
            free(buffer);
            goto fail;
        }
        out_remap->local_to_unified = buffer;
    }

    if (disk_map->u2l_pairs_count > 0) {
        if (!u2l_path) {
            fprintf(stderr, "ERROR: Missing u2l path for layer '%s'\n", disk_map->layer_id);
            goto fail;
        }

        if (verify_external_file(u2l_path, disk_map->u2l_size_bytes, &disk_map->u2l_hash, verify_integrity, false) != 0) {
            if (u2l_resolved) {
                free(u2l_resolved);
                u2l_resolved = NULL;
                u2l_path = disk_map->u2l_path;
                if (verify_external_file(u2l_path, disk_map->u2l_size_bytes, &disk_map->u2l_hash, verify_integrity, true) != 0) {
                    goto fail;
                }
            } else {
                goto fail;
            }
        }

        size_t total_pairs = (size_t)disk_map->u2l_pairs_count;
        size_t raw_count = total_pairs * 2;
        size_t raw_bytes = raw_count * sizeof(uint32_t);

        FILE* f = fopen(u2l_path, "rb");
        if (!f) {
            fprintf(stderr, "ERROR: Failed to open %s: %s\n", u2l_path, strerror(errno));
            goto fail;
        }
        uint32_t* raw_pairs = malloc(raw_bytes);
        if (!raw_pairs) {
            fclose(f);
            goto fail;
        }
        size_t read = fread(raw_pairs, 1, raw_bytes, f);
        fclose(f);
        if (read != raw_bytes) {
            fprintf(stderr, "ERROR: Failed to read complete u2l map from %s\n", u2l_path);
            free(raw_pairs);
            goto fail;
        }

        psam_vocab_sparse_entry_t* entries = malloc(total_pairs * sizeof(psam_vocab_sparse_entry_t));
        if (!entries) {
            free(raw_pairs);
            goto fail;
        }
        for (size_t i = 0; i < total_pairs; ++i) {
            entries[i].unified_id = raw_pairs[i * 2 + 0];
            entries[i].local_id = raw_pairs[i * 2 + 1];
        }
        free(raw_pairs);
        out_remap->unified_to_local_sparse = entries;
        out_remap->unified_to_local_count = (uint32_t)total_pairs;
    }

    free(l2u_resolved);
    free(u2l_resolved);
    return 0;

fail:
    free(l2u_resolved);
    free(u2l_resolved);
    free(out_remap->local_to_unified);
    out_remap->local_to_unified = NULL;
    free(out_remap->unified_to_local_sparse);
    out_remap->unified_to_local_sparse = NULL;
    out_remap->unified_to_local_count = 0;
    return -1;
}

static psam_model_t* load_model_from_ref(const char* composite_path, const psamc_model_ref_t* ref) {
    if (!ref || ref->url[0] == '\0') {
        return NULL;
    }

    char* resolved = resolve_reference_path(composite_path, ref->url);
    const char* path = resolved ? resolved : ref->url;
    psam_model_t* model = psam_load(path);
    if (!model && resolved) {
        model = psam_load(ref->url);
    }
    if (!model) {
        fprintf(stderr, "ERROR: Failed to load model '%s'\n", path);
    }
    free(resolved);
    return model;
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
    char composite_dir[PSAMC_MAX_URL_LENGTH] = {0};

    extract_directory(psamc_path, composite_dir, sizeof(composite_dir));

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
        normalize_reference_path(composite_dir, model_path, ref->url, PSAMC_MAX_URL_LENGTH);
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
        normalize_reference_path(composite_dir, unified_vocab_path, alignment.unified_vocab_path, PSAMC_MAX_URL_LENGTH);
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

        normalize_reference_path(composite_dir, layer_l2u_paths[i], map->l2u_path, PSAMC_MAX_URL_LENGTH);
        normalize_reference_path(composite_dir, layer_u2l_paths[i], map->u2l_path, PSAMC_MAX_URL_LENGTH);

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

    if (psamc_save(psamc_path, NULL, &hyperparams, &manifest, &topology, &alignment, sampler) != 0) {
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

/* ========== Composite Load ========== */

psam_composite_aligned_t* psam_composite_load_aligned(
    const char* psamc_path,
    bool verify_integrity
) {
    if (!psamc_path) {
        return NULL;
    }

    psamc_composite_t* spec = psamc_load(psamc_path, verify_integrity);
    if (!spec) {
        return NULL;
    }

    if (spec->alignment.layer_count == 0 || !spec->alignment.layers) {
        psamc_free(spec);
        return NULL;
    }

    psam_vocab_alignment_t* alignment = calloc(1, sizeof(psam_vocab_alignment_t));
    if (!alignment) {
        psamc_free(spec);
        return NULL;
    }

    alignment->unified_vocab_size = spec->alignment.unified_vocab_size;
    alignment->unknown_policy = spec->alignment.unknown_policy;
    alignment->num_layers = spec->alignment.layer_count;
    alignment->owns_unified_tokens = true;
    alignment->layer_remaps = calloc(alignment->num_layers, sizeof(psam_vocab_remap_t));
    if (!alignment->layer_remaps) {
        psam_vocab_alignment_destroy(alignment);
        psamc_free(spec);
        return NULL;
    }

    char* unified_path = resolve_reference_path(psamc_path, spec->alignment.unified_vocab_path);
    const char* vocab_path = unified_path ? unified_path : spec->alignment.unified_vocab_path;
    if (alignment->unified_vocab_size > 0) {
        alignment->unified_tokens = load_unified_vocab_tokens(vocab_path, alignment->unified_vocab_size);
        if (!alignment->unified_tokens && unified_path) {
            alignment->unified_tokens = load_unified_vocab_tokens(spec->alignment.unified_vocab_path, alignment->unified_vocab_size);
        }
        if (!alignment->unified_tokens) {
            if (unified_path) {
                fprintf(stderr,
                        "ERROR: Failed to load unified vocab '%s' (resolved '%s')\n",
                        spec->alignment.unified_vocab_path,
                        unified_path);
            } else {
                fprintf(stderr,
                        "ERROR: Failed to load unified vocab '%s'\n",
                        spec->alignment.unified_vocab_path);
            }
            free(unified_path);
            psam_vocab_alignment_destroy(alignment);
            psamc_free(spec);
            return NULL;
        }
    } else {
        alignment->owns_unified_tokens = false;
        alignment->unified_tokens = NULL;
    }
    free(unified_path);

    bool load_ok = true;
    for (uint32_t i = 0; i < alignment->num_layers; ++i) {
        psam_vocab_remap_t* remap = &alignment->layer_remaps[i];
        const psam_layer_map_t* map = &spec->alignment.layers[i];
        if (load_layer_map(psamc_path, map, verify_integrity, remap) != 0) {
            load_ok = false;
            break;
        }
    }

    if (!load_ok) {
        psam_vocab_alignment_destroy(alignment);
        psamc_free(spec);
        return NULL;
    }

    uint32_t base_index = spec->topology.base_ref_index;
    if (base_index >= spec->manifest.num_references) {
        base_index = 0;
    }

    psam_model_t* base_model = load_model_from_ref(psamc_path, &spec->manifest.refs[base_index]);
    if (!base_model) {
        psam_vocab_alignment_destroy(alignment);
        psamc_free(spec);
        return NULL;
    }

    psam_composite_aligned_t* composite = psam_create_composite_aligned(
        base_model,
        alignment,
        true,
        true
    );

    if (!composite) {
        psam_destroy(base_model);
        psam_vocab_alignment_destroy(alignment);
        psamc_free(spec);
        return NULL;
    }

    psam_composite_aligned_set_unknown_policy(composite, spec->alignment.unknown_policy);
    psam_composite_aligned_set_coverage_rule(composite, spec->alignment.coverage_rule);
    if (psam_composite_aligned_set_base_weight(composite, spec->topology.base_weight) != 0) {
        psam_composite_aligned_destroy(composite);
        psamc_free(spec);
        return NULL;
    }

    for (uint32_t i = 0; i < spec->topology.layer_count; ++i) {
        const psamc_layer_entry_t* entry = &spec->topology.layers[i];
        uint32_t ref_index = entry->ref_index;
        if (ref_index >= spec->manifest.num_references) {
            continue;
        }
        psam_model_t* overlay = load_model_from_ref(psamc_path, &spec->manifest.refs[ref_index]);
        if (!overlay) {
            psam_composite_aligned_destroy(composite);
            psamc_free(spec);
            return NULL;
        }
        char fallback_id[PSAM_LAYER_ID_MAX];
        const char* layer_id = entry->layer_id[0] != '\0'
            ? entry->layer_id
            : (snprintf(fallback_id, sizeof(fallback_id), "layer-%u", i + 1), fallback_id);

        if (psam_composite_aligned_add_layer(composite, layer_id, overlay, entry->weight, true) != 0) {
            psam_destroy(overlay);
            psam_composite_aligned_destroy(composite);
            psamc_free(spec);
            return NULL;
        }
        psam_composite_aligned_update_layer_bias(composite, layer_id, entry->bias);
    }

    /* Apply sampler defaults if present (natively stored in spec) */
    psam_sampler_t sampler_defaults = {
        .transform = spec->sampler_defaults.logit_transform,
        .temperature = spec->sampler_defaults.temperature,
        .top_k = spec->sampler_defaults.top_k,
        .top_p = spec->sampler_defaults.top_p,
        .seed = spec->sampler_defaults.seed
    };
    psam_composite_set_sampler_defaults(composite->composite, &sampler_defaults);

    psamc_free(spec);
    return composite;
}
