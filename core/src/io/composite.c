/**
 * composite.c - Composite model format (.psamc) implementation
 *
 * Provides integrity checking, external references, and hyperparameter storage.
 */

#define _POSIX_C_SOURCE 200809L

#include "../../include/psam_composite.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(_WIN32)
#include <io.h>
#define psam_fseeko _fseeki64
#define psam_ftello _ftelli64
typedef int64_t psam_off_t;
#else
#define psam_fseeko fseeko
#define psam_ftello ftello
typedef off_t psam_off_t;
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>

/* External SHA-256 functions */
extern int sha256_file(const char* path, uint8_t* out_hash);
extern void sha256_data(const uint8_t* data, size_t len, uint8_t* out_hash);

/* ===== On-disk structures (packed layout) ===== */

typedef struct {
    char url[PSAMC_MAX_URL_LENGTH];
    uint8_t sha256[PSAMC_SOURCE_HASH_SIZE];
    uint64_t size;
    uint16_t version_major;
    uint16_t version_minor;
    uint16_t version_patch;
    uint16_t reserved;
    char model_id[PSAMC_MAX_MODEL_ID];
    uint8_t reserved2[32];
} psamc_model_ref_disk_t;

typedef struct {
    char label[PSAMC_SOURCE_LABEL_MAX];
    char uri[PSAMC_SOURCE_URI_MAX];
    char license[PSAMC_SOURCE_LICENSE_MAX];
} psamc_source_disk_t;

typedef struct {
    uint32_t preset;
    float alpha;
    float min_evidence;
    uint32_t top_k;
    uint8_t enable_ppmi;
    uint8_t enable_idf;
    uint8_t reserved_flags[2];
    float ppmi_k;
    float idf_smoothing;
    uint32_t top_k_function;
    uint32_t top_k_content;
    uint32_t top_k_rare;
    float edge_dropout;
    uint8_t reserved[24];
} psamc_hyperparams_disk_t;

typedef struct {
    float base_weight;
    uint32_t base_ref_index;
    uint32_t layer_count;
    uint8_t reserved[20];
} psamc_layer_section_header_t;

typedef struct {
    char layer_id[PSAM_LAYER_ID_MAX];
    float weight;
    float bias;         /* NEW: Layer bias offset */
    uint32_t ref_index;
    uint8_t reserved[20];
} psamc_layer_entry_disk_t;

/* Sampler defaults (new in v1.1) */
typedef struct {
    uint32_t logit_transform;  /* psam_logit_transform_t */
   float temperature;
   int32_t top_k;
   float top_p;
   uint64_t seed;
   uint8_t reserved[12];
} psamc_sampler_disk_t;

typedef struct {
    uint32_t unknown_policy;
    uint32_t coverage_rule;
    uint32_t unified_vocab_size;
    uint32_t unified_unk;
    uint32_t layer_count;
    uint32_t reserved;
    uint64_t unified_vocab_size_bytes;
    char unified_vocab_path[PSAMC_MAX_URL_LENGTH];
    uint8_t unified_vocab_hash[PSAMC_SOURCE_HASH_SIZE];
    uint8_t reserved2[32];
} psamc_alignment_header_disk_t;

typedef struct {
    char layer_id[PSAM_LAYER_ID_MAX];
    uint32_t local_vocab_size;
    uint32_t local_unk;
    uint32_t u2l_pairs_count;
    float coverage;
    uint8_t reserved[12];
    uint64_t l2u_size_bytes;
    char l2u_path[PSAMC_MAX_URL_LENGTH];
    uint8_t l2u_hash[PSAMC_SOURCE_HASH_SIZE];
    uint64_t u2l_size_bytes;
    char u2l_path[PSAMC_MAX_URL_LENGTH];
    uint8_t u2l_hash[PSAMC_SOURCE_HASH_SIZE];
} psamc_alignment_layer_disk_t;

/* ===== Helper utilities ===== */

const psamc_hyperparams_t* psamc_get_preset(psamc_preset_t preset) {
    switch (preset) {
        case PSAMC_PRESET_FAST:
            return &PSAMC_PRESET_FAST_CONFIG;
        case PSAMC_PRESET_ACCURATE:
            return &PSAMC_PRESET_ACCURATE_CONFIG;
        case PSAMC_PRESET_TINY:
            return &PSAMC_PRESET_TINY_CONFIG;
        case PSAMC_PRESET_BALANCED:
        default:
            return &PSAMC_PRESET_BALANCED_CONFIG;
    }
}

int psamc_sha256_file(const char* path, sha256_hash_t* out_hash) {
    return sha256_file(path, out_hash->hash);
}

static int sha256_equal(const sha256_hash_t* a, const sha256_hash_t* b) {
    return memcmp(a->hash, b->hash, PSAMC_SOURCE_HASH_SIZE) == 0;
}

static size_t manifest_section_size(const psamc_manifest_t* manifest) {
    if (!manifest) {
        return 0;
    }

    size_t size = 0;
    size += sizeof(uint32_t); /* num_references */
    size += sizeof(uint32_t); /* source_count */
    size += sizeof(uint64_t); /* created_timestamp */
    size += PSAMC_CREATED_BY_MAX;
    size += PSAMC_SOURCE_HASH_SIZE; /* source_hash */
    size += 32; /* reserved */
    size += (size_t)manifest->num_references * sizeof(psamc_model_ref_disk_t);
    size += (size_t)manifest->source_count * sizeof(psamc_source_disk_t);
    size += PSAMC_SOURCE_HASH_SIZE; /* self_hash */
    return size;
}

static int write_manifest_section(FILE* f, const psamc_manifest_t* manifest, uint64_t* self_hash_rel_offset) {
    if (!f || !manifest) {
        return -1;
    }

    long section_start = ftell(f);
    if (section_start < 0) {
        return -1;
    }

    uint32_t num_refs = manifest->num_references;
    uint32_t source_count = manifest->source_count;

    if (fwrite(&num_refs, sizeof(uint32_t), 1, f) != 1 ||
        fwrite(&source_count, sizeof(uint32_t), 1, f) != 1 ||
        fwrite(&manifest->created_timestamp, sizeof(uint64_t), 1, f) != 1) {
        return -1;
    }

    char created_by[PSAMC_CREATED_BY_MAX] = {0};
    if (manifest->created_by[0] != '\0') {
        snprintf(created_by, PSAMC_CREATED_BY_MAX, "%s", manifest->created_by);
    }
    if (fwrite(created_by, sizeof(char), PSAMC_CREATED_BY_MAX, f) != PSAMC_CREATED_BY_MAX) {
        return -1;
    }

    if (fwrite(manifest->source_hash.hash, sizeof(uint8_t), PSAMC_SOURCE_HASH_SIZE, f) != PSAMC_SOURCE_HASH_SIZE) {
        return -1;
    }

    uint8_t reserved_block[32] = {0};
    if (fwrite(reserved_block, sizeof(uint8_t), sizeof(reserved_block), f) != sizeof(reserved_block)) {
        return -1;
    }

    for (uint32_t i = 0; i < num_refs; i++) {
        psamc_model_ref_disk_t disk = {0};
        const psamc_model_ref_t* ref = &manifest->refs[i];
        if (ref->url[0] != '\0') {
            snprintf(disk.url, PSAMC_MAX_URL_LENGTH, "%s", ref->url);
        }
        memcpy(disk.sha256, ref->sha256.hash, PSAMC_SOURCE_HASH_SIZE);
        disk.size = ref->size;
        disk.version_major = ref->version.major;
        disk.version_minor = ref->version.minor;
        disk.version_patch = ref->version.patch;
        if (ref->model_id[0] != '\0') {
            snprintf(disk.model_id, PSAMC_MAX_MODEL_ID, "%s", ref->model_id);
        }
        if (fwrite(&disk, sizeof(psamc_model_ref_disk_t), 1, f) != 1) {
            return -1;
        }
    }

    for (uint32_t i = 0; i < source_count; i++) {
        psamc_source_disk_t disk = {0};
        const psamc_source_t* src = &manifest->sources[i];
        if (src->label[0] != '\0') {
            snprintf(disk.label, PSAMC_SOURCE_LABEL_MAX, "%s", src->label);
        }
        if (src->uri[0] != '\0') {
            snprintf(disk.uri, PSAMC_SOURCE_URI_MAX, "%s", src->uri);
        }
        if (src->license[0] != '\0') {
            snprintf(disk.license, PSAMC_SOURCE_LICENSE_MAX, "%s", src->license);
        }
        if (fwrite(&disk, sizeof(psamc_source_disk_t), 1, f) != 1) {
            return -1;
        }
    }

    long before_hash = ftell(f);
    if (before_hash < 0) {
        return -1;
    }

    uint8_t zero_hash[PSAMC_SOURCE_HASH_SIZE] = {0};
    if (fwrite(zero_hash, sizeof(uint8_t), PSAMC_SOURCE_HASH_SIZE, f) != PSAMC_SOURCE_HASH_SIZE) {
        return -1;
    }

    if (self_hash_rel_offset) {
        *self_hash_rel_offset = (uint64_t)(before_hash - section_start);
    }

    return 0;
}

static int write_config_section(FILE* f, const psamc_hyperparams_t* hyperparams) {
    if (!f || !hyperparams) {
        return -1;
    }

    psamc_hyperparams_disk_t disk;
    memset(&disk, 0, sizeof(disk));

    disk.preset = (uint32_t)hyperparams->preset;
    disk.alpha = hyperparams->alpha;
    disk.min_evidence = hyperparams->min_evidence;
    disk.top_k = hyperparams->top_k;
    disk.enable_ppmi = hyperparams->enable_ppmi ? 1 : 0;
    disk.enable_idf = hyperparams->enable_idf ? 1 : 0;
    disk.ppmi_k = hyperparams->ppmi_k;
    disk.idf_smoothing = hyperparams->idf_smoothing;
    disk.top_k_function = hyperparams->top_k_function;
    disk.top_k_content = hyperparams->top_k_content;
    disk.top_k_rare = hyperparams->top_k_rare;
    disk.edge_dropout = hyperparams->edge_dropout;

    if (fwrite(&disk, sizeof(psamc_hyperparams_disk_t), 1, f) != 1) {
        return -1;
    }

    return 0;
}

static size_t layer_section_size(const psamc_topology_t* topology) {
    if (!topology) {
        return 0;
    }

    size_t size = sizeof(psamc_layer_section_header_t);
    size += (size_t)topology->layer_count * sizeof(psamc_layer_entry_disk_t);
    return size;
}

static void init_topology_defaults(psamc_topology_t* topology) {
    if (!topology) {
        return;
    }
    topology->base_weight = 1.0f;
    topology->base_ref_index = 0;
    topology->layer_count = 0;
    topology->layers = NULL;
}

static void free_topology(psamc_topology_t* topology) {
    if (!topology) {
        return;
    }
    free(topology->layers);
    topology->layers = NULL;
    topology->layer_count = 0;
    topology->base_weight = 1.0f;
    topology->base_ref_index = 0;
}

static void init_sampler_defaults(psamc_sampler_defaults_t* sampler) {
    if (!sampler) {
        return;
    }
    sampler->logit_transform = PSAM_LOGIT_ZSCORE;
    sampler->temperature = 1.0f;
    sampler->top_k = 50;
    sampler->top_p = 0.95f;
    sampler->seed = 42;
}

static int write_sampler_section(FILE* f, const psamc_sampler_defaults_t* sampler) {
    if (!f || !sampler) {
        return -1;
    }

    psamc_sampler_disk_t disk;
    memset(&disk, 0, sizeof(disk));
    disk.logit_transform = (uint32_t)sampler->logit_transform;
    disk.temperature = sampler->temperature;
    disk.top_k = sampler->top_k;
    disk.top_p = sampler->top_p;
    disk.seed = sampler->seed;

    return fwrite(&disk, sizeof(psamc_sampler_disk_t), 1, f) == 1 ? 0 : -1;
}

static int read_sampler_section(FILE* f, const psamc_section_entry_t* section, psamc_sampler_defaults_t* out_sampler) {
    if (!f || !section || !out_sampler) {
        return -1;
    }

    if (psam_fseeko(f, (psam_off_t)section->offset, SEEK_SET) != 0) {
        return -1;
    }

    psamc_sampler_disk_t disk;
    if (fread(&disk, sizeof(psamc_sampler_disk_t), 1, f) != 1) {
        return -1;
    }

    memset(out_sampler, 0, sizeof(*out_sampler));
    out_sampler->logit_transform = (psam_logit_transform_t)disk.logit_transform;
    out_sampler->temperature = disk.temperature;
    out_sampler->top_k = disk.top_k;
    out_sampler->top_p = disk.top_p;
    out_sampler->seed = disk.seed;

    return 0;
}

static void init_alignment_defaults(psam_alignment_info_t* alignment) {
    if (!alignment) {
        return;
    }
    memset(alignment, 0, sizeof(*alignment));
}

static size_t alignment_section_size(const psam_alignment_info_t* alignment) {
    if (!alignment || alignment->layer_count == 0) {
        return 0;
    }
    size_t size = sizeof(psamc_alignment_header_disk_t);
    size += (size_t)alignment->layer_count * sizeof(psamc_alignment_layer_disk_t);
    return size;
}

static int write_alignment_section(FILE* f, const psam_alignment_info_t* alignment) {
    if (!f || !alignment || alignment->layer_count == 0 || !alignment->layers) {
        return -1;
    }

    psamc_alignment_header_disk_t header = {0};
    header.unknown_policy = (uint32_t)alignment->unknown_policy;
    header.coverage_rule = (uint32_t)alignment->coverage_rule;
    header.unified_vocab_size = alignment->unified_vocab_size;
    header.unified_unk = alignment->unified_unk;
    header.layer_count = alignment->layer_count;
    header.unified_vocab_size_bytes = alignment->unified_vocab_size_bytes;
    if (alignment->unified_vocab_path[0] != '\0') {
        snprintf(header.unified_vocab_path, PSAMC_MAX_URL_LENGTH, "%s", alignment->unified_vocab_path);
    }
    memcpy(header.unified_vocab_hash, alignment->unified_vocab_hash.hash, PSAMC_SOURCE_HASH_SIZE);

    if (fwrite(&header, sizeof(header), 1, f) != 1) {
        return -1;
    }

    for (uint32_t i = 0; i < alignment->layer_count; ++i) {
        const psam_layer_map_t* map = &alignment->layers[i];
        psamc_alignment_layer_disk_t disk = {0};
        if (map->layer_id[0] != '\0') {
            snprintf(disk.layer_id, PSAM_LAYER_ID_MAX, "%s", map->layer_id);
        }
        disk.local_vocab_size = map->local_vocab_size;
        disk.local_unk = map->local_unk;
        disk.u2l_pairs_count = map->u2l_pairs_count;
        disk.coverage = map->coverage;
        disk.l2u_size_bytes = map->l2u_size_bytes;
        disk.u2l_size_bytes = map->u2l_size_bytes;
        if (map->l2u_path[0] != '\0') {
            snprintf(disk.l2u_path, PSAMC_MAX_URL_LENGTH, "%s", map->l2u_path);
        }
        if (map->u2l_path[0] != '\0') {
            snprintf(disk.u2l_path, PSAMC_MAX_URL_LENGTH, "%s", map->u2l_path);
        }
        memcpy(disk.l2u_hash, map->l2u_hash.hash, PSAMC_SOURCE_HASH_SIZE);
        memcpy(disk.u2l_hash, map->u2l_hash.hash, PSAMC_SOURCE_HASH_SIZE);

        if (fwrite(&disk, sizeof(disk), 1, f) != 1) {
            return -1;
        }
    }

    return 0;
}

static int read_alignment_section(FILE* f, const psamc_section_entry_t* section, psam_alignment_info_t* out_alignment) {
    if (!f || !section || !out_alignment) {
        return -1;
    }

    if (psam_fseeko(f, (psam_off_t)section->offset, SEEK_SET) != 0) {
        return -1;
    }

    psamc_alignment_header_disk_t header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        return -1;
    }

    init_alignment_defaults(out_alignment);
    out_alignment->unknown_policy = (psam_unknown_policy_t)header.unknown_policy;
    out_alignment->coverage_rule = (psam_coverage_rule_t)header.coverage_rule;
    out_alignment->unified_vocab_size = header.unified_vocab_size;
    out_alignment->unified_unk = header.unified_unk;
    out_alignment->layer_count = header.layer_count;
    out_alignment->unified_vocab_size_bytes = header.unified_vocab_size_bytes;
    snprintf(out_alignment->unified_vocab_path, PSAMC_MAX_URL_LENGTH, "%s", header.unified_vocab_path);
    memcpy(out_alignment->unified_vocab_hash.hash, header.unified_vocab_hash, PSAMC_SOURCE_HASH_SIZE);

    if (header.layer_count == 0) {
        return 0;
    }

    out_alignment->layers = calloc(header.layer_count, sizeof(psam_layer_map_t));
    if (!out_alignment->layers) {
        return -1;
    }

    for (uint32_t i = 0; i < header.layer_count; ++i) {
        psamc_alignment_layer_disk_t disk;
        if (fread(&disk, sizeof(disk), 1, f) != 1) {
            free(out_alignment->layers);
            out_alignment->layers = NULL;
            out_alignment->layer_count = 0;
            return -1;
        }

        psam_layer_map_t* map = &out_alignment->layers[i];
        snprintf(map->layer_id, PSAM_LAYER_ID_MAX, "%s", disk.layer_id);
        map->local_vocab_size = disk.local_vocab_size;
        map->local_unk = disk.local_unk;
        map->u2l_pairs_count = disk.u2l_pairs_count;
        map->coverage = disk.coverage;
        map->l2u = NULL;
        map->u2l_pairs = NULL;
        snprintf(map->l2u_path, PSAMC_MAX_URL_LENGTH, "%s", disk.l2u_path);
        snprintf(map->u2l_path, PSAMC_MAX_URL_LENGTH, "%s", disk.u2l_path);
        memcpy(map->l2u_hash.hash, disk.l2u_hash, PSAMC_SOURCE_HASH_SIZE);
        memcpy(map->u2l_hash.hash, disk.u2l_hash, PSAMC_SOURCE_HASH_SIZE);
        map->l2u_size_bytes = disk.l2u_size_bytes;
        map->u2l_size_bytes = disk.u2l_size_bytes;
    }

    return 0;
}

static void free_alignment(psam_alignment_info_t* alignment) {
    if (!alignment) {
        return;
    }
    free(alignment->layers);
    alignment->layers = NULL;
    alignment->layer_count = 0;
    alignment->unified_vocab_size = 0;
}

static int write_layer_section(FILE* f, const psamc_topology_t* topology) {
    if (!f || !topology) {
        return -1;
    }

    psamc_layer_section_header_t header = {0};
    header.base_weight = topology->base_weight;
    header.base_ref_index = topology->base_ref_index;
    header.layer_count = topology->layer_count;

    if (fwrite(&header, sizeof(header), 1, f) != 1) {
        return -1;
    }

    for (uint32_t i = 0; i < topology->layer_count; ++i) {
        psamc_layer_entry_disk_t disk = {0};
        const psamc_layer_entry_t* entry = &topology->layers[i];
        if (entry->layer_id[0] != '\0') {
            snprintf(disk.layer_id, PSAM_LAYER_ID_MAX, "%s", entry->layer_id);
        }
        disk.weight = entry->weight;
        disk.bias = entry->bias;
        disk.ref_index = entry->ref_index;
        if (fwrite(&disk, sizeof(disk), 1, f) != 1) {
            return -1;
        }
    }

    return 0;
}

static int compute_self_hash_excluding(const char* path, uint64_t offset, size_t length, sha256_hash_t* out_hash) {
    if (!path || !out_hash) {
        return -1;
    }

    FILE* f = fopen(path, "rb");
    if (!f) {
        return -1;
    }

    if (psam_fseeko(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }

    psam_off_t file_size = psam_ftello(f);
    if (file_size < 0) {
        fclose(f);
        return -1;
    }
    if ((uint64_t)file_size < offset + length) {
        fclose(f);
        return -1;
    }
    if (psam_fseeko(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }

    uint8_t* buffer = malloc((size_t)file_size);
    if (!buffer) {
        fclose(f);
        return -1;
    }

    size_t read_bytes = fread(buffer, 1, (size_t)file_size, f);
    fclose(f);
    if (read_bytes != (size_t)file_size) {
        free(buffer);
        return -1;
    }

    memset(buffer + offset, 0, length);
    sha256_data(buffer, (size_t)file_size, out_hash->hash);
    free(buffer);
    return 0;
}

static int verify_model_reference(const psamc_model_ref_t* ref) {
    if (!ref || ref->url[0] == '\0') {
        return -1;
    }

    FILE* f = fopen(ref->url, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Referenced model not found: %s\n", ref->url);
        return -1;
    }

    if (psam_fseeko(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    uint64_t actual_size = (uint64_t)psam_ftello(f);
    fclose(f);

    if (actual_size != ref->size) {
        fprintf(stderr, "ERROR: Size mismatch for %s\n", ref->url);
        fprintf(stderr, "  Expected: %llu bytes\n", (unsigned long long)ref->size);
        fprintf(stderr, "  Actual:   %llu bytes\n", (unsigned long long)actual_size);
        return -1;
    }

    sha256_hash_t actual_hash;
    if (psamc_sha256_file(ref->url, &actual_hash) != 0) {
        fprintf(stderr, "ERROR: Failed to compute hash for %s\n", ref->url);
        return -1;
    }

    if (!sha256_equal(&actual_hash, &ref->sha256)) {
        fprintf(stderr, "ERROR: SHA-256 mismatch for %s\n", ref->url);
        fprintf(stderr, "  Expected hash: ");
        for (size_t i = 0; i < PSAMC_SOURCE_HASH_SIZE; i++) {
            fprintf(stderr, "%02x", ref->sha256.hash[i]);
        }
        fprintf(stderr, "\n  Actual hash:   ");
        for (size_t i = 0; i < PSAMC_SOURCE_HASH_SIZE; i++) {
            fprintf(stderr, "%02x", actual_hash.hash[i]);
        }
        fprintf(stderr, "\n");
        return -1;
    }

    return 0;
}

int psamc_verify_manifest(const psamc_manifest_t* manifest) {
    if (!manifest) {
        return -1;
    }

    for (uint32_t i = 0; i < manifest->num_references; i++) {
        if (verify_model_reference(&manifest->refs[i]) != 0) {
            return -1;
        }
    }

    return 0;
}

/* ===== Composite save/load ===== */

int psamc_save(
    const char* path,
    const psam_model_t* base_model,
    const psamc_hyperparams_t* hyperparams,
    const psamc_manifest_t* manifest,
    const psamc_topology_t* topology,
    const psam_alignment_info_t* alignment,
    const psamc_sampler_defaults_t* sampler_overrides
) {
    (void)base_model; /* Placeholder for future embedded support */

    if (!path || !hyperparams || !topology) {
        return -1;
    }

    uint32_t num_sections = 2; /* config + sampler */
    if (manifest) {
        num_sections++;
    }
    if (topology) {
        num_sections++;
    }
    size_t alignment_size_bytes = alignment_section_size(alignment);
    if (alignment_size_bytes > 0) {
        num_sections++;
    }

    psamc_section_entry_t* sections = calloc(num_sections, sizeof(psamc_section_entry_t));
    if (!sections) {
        return -1;
    }

    uint64_t header_size = sizeof(psamc_header_t);
    uint64_t table_size = (uint64_t)num_sections * sizeof(psamc_section_entry_t);
    uint64_t offset = header_size + table_size;

    size_t section_index = 0;
    uint64_t manifest_self_hash_offset_abs = 0;
    uint64_t manifest_section_size_bytes = manifest_section_size(manifest);

    if (manifest) {
        sections[section_index].type = PSAMC_SECTION_MANIFEST;
        sections[section_index].flags = 0;
        sections[section_index].offset = offset;
        sections[section_index].size = manifest_section_size_bytes;
        offset += manifest_section_size_bytes;
        section_index++;
    }

    if (topology) {
        sections[section_index].type = PSAMC_SECTION_LAYER;
        sections[section_index].flags = 0;
        sections[section_index].offset = offset;
        sections[section_index].size = layer_section_size(topology);
        offset += sections[section_index].size;
        section_index++;
    }

    if (alignment_size_bytes > 0) {
        sections[section_index].type = PSAMC_SECTION_ALIGNMENT;
        sections[section_index].flags = 0;
        sections[section_index].offset = offset;
        sections[section_index].size = alignment_size_bytes;
        offset += sections[section_index].size;
        section_index++;
    }

    sections[section_index].type = PSAMC_SECTION_CONFIG;
    sections[section_index].flags = 0;
    sections[section_index].offset = offset;
    sections[section_index].size = sizeof(psamc_hyperparams_disk_t);
    offset += sections[section_index].size;
    section_index++;

    /* Sampler section */
    sections[section_index].type = PSAMC_SECTION_SAMPLER;
    sections[section_index].flags = 0;
    sections[section_index].offset = offset;
    sections[section_index].size = sizeof(psamc_sampler_disk_t);
    offset += sections[section_index].size;

    psamc_header_t header = {0};
    header.magic = PSAMC_MAGIC;
    header.version = PSAMC_VERSION;
    header.flags = manifest ? PSAMC_FLAG_HAS_MANIFEST : 0;
    header.num_sections = num_sections;
    header.file_size = offset;

    FILE* f = fopen(path, "wb");
    if (!f) {
        free(sections);
        return -1;
    }

    if (fwrite(&header, sizeof(psamc_header_t), 1, f) != 1 ||
        fwrite(sections, sizeof(psamc_section_entry_t), num_sections, f) != num_sections) {
        fclose(f);
        free(sections);
        return -1;
    }

    if (manifest) {
        uint64_t rel_offset = 0;
        if (write_manifest_section(f, manifest, &rel_offset) != 0) {
            fclose(f);
            free(sections);
            return -1;
        }
        manifest_self_hash_offset_abs = sections[0].offset + rel_offset;
    }

    if (topology && write_layer_section(f, topology) != 0) {
        fclose(f);
        free(sections);
        return -1;
    }

    if (alignment_size_bytes > 0 && write_alignment_section(f, alignment) != 0) {
        fclose(f);
        free(sections);
        return -1;
    }

    if (write_config_section(f, hyperparams) != 0) {
        fclose(f);
        free(sections);
        return -1;
    }

    /* Write sampler defaults */
    psamc_sampler_defaults_t sampler_defaults;
    if (sampler_overrides) {
        sampler_defaults = *sampler_overrides;
    } else {
        init_sampler_defaults(&sampler_defaults);
    }
    if (write_sampler_section(f, &sampler_defaults) != 0) {
        fclose(f);
        free(sections);
        return -1;
    }

    fclose(f);

    if (manifest) {
        sha256_hash_t computed;
        if (compute_self_hash_excluding(path, manifest_self_hash_offset_abs, PSAMC_SOURCE_HASH_SIZE, &computed) != 0) {
            free(sections);
            return -1;
        }

        FILE* patch = fopen(path, "rb+");
        if (!patch) {
            free(sections);
            return -1;
        }
        if (psam_fseeko(patch, (psam_off_t)manifest_self_hash_offset_abs, SEEK_SET) != 0) {
            fclose(patch);
            free(sections);
            return -1;
        }
        if (fwrite(computed.hash, sizeof(uint8_t), PSAMC_SOURCE_HASH_SIZE, patch) != PSAMC_SOURCE_HASH_SIZE) {
            fclose(patch);
            free(sections);
            return -1;
        }
        fclose(patch);
    }

    free(sections);
    return 0;
}

static void free_manifest(psamc_manifest_t* manifest) {
    if (!manifest) {
        return;
    }
    free(manifest->refs);
    manifest->refs = NULL;
    manifest->num_references = 0;
    free(manifest->sources);
    manifest->sources = NULL;
    manifest->source_count = 0;
}

static int read_manifest_section(FILE* f, const psamc_section_entry_t* section, psamc_manifest_t* out_manifest, uint64_t* self_hash_offset_abs) {
    if (!f || !section || !out_manifest) {
        return -1;
    }

    if (psam_fseeko(f, (psam_off_t)section->offset, SEEK_SET) != 0) {
        return -1;
    }

    memset(out_manifest, 0, sizeof(*out_manifest));

    uint32_t num_refs = 0;
    uint32_t source_count = 0;

    if (fread(&num_refs, sizeof(uint32_t), 1, f) != 1 ||
        fread(&source_count, sizeof(uint32_t), 1, f) != 1 ||
        fread(&out_manifest->created_timestamp, sizeof(uint64_t), 1, f) != 1) {
        return -1;
    }

    char created_by_raw[PSAMC_CREATED_BY_MAX];
    if (fread(created_by_raw, sizeof(char), PSAMC_CREATED_BY_MAX, f) != PSAMC_CREATED_BY_MAX) {
        return -1;
    }
    created_by_raw[PSAMC_CREATED_BY_MAX - 1] = '\0';
    snprintf(out_manifest->created_by, PSAMC_CREATED_BY_MAX, "%s", created_by_raw);

    if (fread(out_manifest->source_hash.hash, sizeof(uint8_t), PSAMC_SOURCE_HASH_SIZE, f) != PSAMC_SOURCE_HASH_SIZE) {
        return -1;
    }

    uint8_t reserved_block[32];
    if (fread(reserved_block, sizeof(uint8_t), sizeof(reserved_block), f) != sizeof(reserved_block)) {
        return -1;
    }

    if (num_refs > 0) {
        out_manifest->refs = calloc(num_refs, sizeof(psamc_model_ref_t));
        if (!out_manifest->refs) {
            return -1;
        }
    }

    for (uint32_t i = 0; i < num_refs; i++) {
        psamc_model_ref_disk_t disk;
        if (fread(&disk, sizeof(psamc_model_ref_disk_t), 1, f) != 1) {
            free_manifest(out_manifest);
            return -1;
        }

        psamc_model_ref_t* ref = &out_manifest->refs[i];
        snprintf(ref->url, PSAMC_MAX_URL_LENGTH, "%s", disk.url);
        memcpy(ref->sha256.hash, disk.sha256, PSAMC_SOURCE_HASH_SIZE);
        ref->size = disk.size;
        ref->version.major = disk.version_major;
        ref->version.minor = disk.version_minor;
        ref->version.patch = disk.version_patch;
        snprintf(ref->model_id, PSAMC_MAX_MODEL_ID, "%s", disk.model_id);
    }

    if (source_count > 0) {
        out_manifest->sources = calloc(source_count, sizeof(psamc_source_t));
        if (!out_manifest->sources) {
            free_manifest(out_manifest);
            return -1;
        }
    }

    for (uint32_t i = 0; i < source_count; i++) {
        psamc_source_disk_t disk;
        if (fread(&disk, sizeof(psamc_source_disk_t), 1, f) != 1) {
            free_manifest(out_manifest);
            return -1;
        }
        psamc_source_t* src = &out_manifest->sources[i];
        snprintf(src->label, PSAMC_SOURCE_LABEL_MAX, "%s", disk.label);
        snprintf(src->uri, PSAMC_SOURCE_URI_MAX, "%s", disk.uri);
        snprintf(src->license, PSAMC_SOURCE_LICENSE_MAX, "%s", disk.license);
    }

    uint64_t absolute_self_hash_offset = section->offset + section->size - PSAMC_SOURCE_HASH_SIZE;
    if (psam_fseeko(f, (psam_off_t)absolute_self_hash_offset, SEEK_SET) != 0) {
        free_manifest(out_manifest);
        return -1;
    }
    if (fread(out_manifest->self_hash.hash, sizeof(uint8_t), PSAMC_SOURCE_HASH_SIZE, f) != PSAMC_SOURCE_HASH_SIZE) {
        free_manifest(out_manifest);
        return -1;
    }

    out_manifest->num_references = num_refs;
    out_manifest->source_count = source_count;

    if (self_hash_offset_abs) {
        *self_hash_offset_abs = absolute_self_hash_offset;
    }

    return 0;
}

static int read_config_section(FILE* f, const psamc_section_entry_t* section, psamc_hyperparams_t* out_config) {
    if (!f || !section || !out_config) {
        return -1;
    }

    if (psam_fseeko(f, (psam_off_t)section->offset, SEEK_SET) != 0) {
        return -1;
    }

    psamc_hyperparams_disk_t disk;
    if (fread(&disk, sizeof(psamc_hyperparams_disk_t), 1, f) != 1) {
        return -1;
    }

    memset(out_config, 0, sizeof(*out_config));
    out_config->preset = (psamc_preset_t)disk.preset;
    out_config->alpha = disk.alpha;
    out_config->min_evidence = disk.min_evidence;
    out_config->top_k = disk.top_k;
    out_config->enable_ppmi = disk.enable_ppmi != 0;
    out_config->enable_idf = disk.enable_idf != 0;
    out_config->ppmi_k = disk.ppmi_k;
    out_config->idf_smoothing = disk.idf_smoothing;
    out_config->top_k_function = disk.top_k_function;
    out_config->top_k_content = disk.top_k_content;
    out_config->top_k_rare = disk.top_k_rare;
    out_config->edge_dropout = disk.edge_dropout;

    return 0;
}

static int read_layer_section(FILE* f, const psamc_section_entry_t* section, psamc_topology_t* out_topology) {
    if (!f || !section || !out_topology) {
        return -1;
    }

    if (psam_fseeko(f, (psam_off_t)section->offset, SEEK_SET) != 0) {
        return -1;
    }

    psamc_layer_section_header_t header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        return -1;
    }

    free_topology(out_topology);
    out_topology->base_weight = header.base_weight;
    out_topology->base_ref_index = header.base_ref_index;
    out_topology->layer_count = header.layer_count;

    if (header.layer_count == 0) {
        out_topology->layers = NULL;
        return 0;
    }

    out_topology->layers = calloc(header.layer_count, sizeof(psamc_layer_entry_t));
    if (!out_topology->layers) {
        out_topology->layer_count = 0;
        return -1;
    }

    for (uint32_t i = 0; i < header.layer_count; ++i) {
        psamc_layer_entry_disk_t disk;
        if (fread(&disk, sizeof(disk), 1, f) != 1) {
            free_topology(out_topology);
            return -1;
        }

        psamc_layer_entry_t* entry = &out_topology->layers[i];
        snprintf(entry->layer_id, PSAM_LAYER_ID_MAX, "%s", disk.layer_id);
        entry->weight = disk.weight;
        entry->bias = disk.bias;  /* Load bias (will be 0.0 for old files) */
        entry->ref_index = disk.ref_index;
    }

    return 0;
}

static int synthesize_topology_from_manifest(const psamc_manifest_t* manifest, psamc_topology_t* topology) {
    if (!manifest || !topology) {
        return -1;
    }

    free_topology(topology);
    topology->base_weight = 1.0f;
    topology->base_ref_index = 0;

    if (manifest->num_references <= 1) {
        topology->layer_count = 0;
        topology->layers = NULL;
        return 0;
    }

    uint32_t overlays = manifest->num_references - 1;
    topology->layers = calloc(overlays, sizeof(psamc_layer_entry_t));
    if (!topology->layers) {
        topology->layer_count = 0;
        return -1;
    }

    topology->layer_count = overlays;
    for (uint32_t i = 0; i < overlays; ++i) {
        const psamc_model_ref_t* ref = &manifest->refs[i + 1];
        psamc_layer_entry_t* entry = &topology->layers[i];
        if (ref->model_id[0] != '\0') {
            snprintf(entry->layer_id, PSAM_LAYER_ID_MAX, "%s", ref->model_id);
        } else {
            snprintf(entry->layer_id, PSAM_LAYER_ID_MAX, "layer-%u", i);
        }
        entry->weight = 1.0f;
        entry->ref_index = i + 1;
    }

    return 0;
}

psamc_composite_t* psamc_load(const char* path, bool verify_integrity) {
    if (!path) {
        return NULL;
    }

    FILE* f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }

    psamc_header_t header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    if (header.magic != PSAMC_MAGIC || header.version != PSAMC_VERSION) {
        fclose(f);
        return NULL;
    }

    if (header.num_sections == 0 || header.num_sections > 16) {
        fclose(f);
        return NULL;
    }

    psamc_section_entry_t* sections = calloc(header.num_sections, sizeof(psamc_section_entry_t));
    if (!sections) {
        fclose(f);
        return NULL;
    }

    if (fread(sections, sizeof(psamc_section_entry_t), header.num_sections, f) != header.num_sections) {
        free(sections);
        fclose(f);
        return NULL;
    }

    psamc_manifest_t manifest = {0};
    psamc_hyperparams_t config = {0};
    psamc_topology_t topology;
    init_topology_defaults(&topology);
    psamc_sampler_defaults_t sampler_defaults;
    init_sampler_defaults(&sampler_defaults);
    psam_alignment_info_t alignment_info;
    init_alignment_defaults(&alignment_info);

    bool manifest_loaded = false;
    bool config_loaded = false;
    bool alignment_loaded = false;
    uint64_t manifest_hash_offset = 0;

    for (uint32_t i = 0; i < header.num_sections; i++) {
        const psamc_section_entry_t* section = &sections[i];
        switch (section->type) {
            case PSAMC_SECTION_MANIFEST:
                if (read_manifest_section(f, section, &manifest, &manifest_hash_offset) != 0) {
                    free(sections);
                    fclose(f);
                    return NULL;
                }
                manifest_loaded = true;
                break;
            case PSAMC_SECTION_CONFIG:
                if (read_config_section(f, section, &config) != 0) {
                    free(sections);
                    fclose(f);
                    free_manifest(&manifest);
                    free_alignment(&alignment_info);
                    return NULL;
                }
                config_loaded = true;
                break;
            case PSAMC_SECTION_LAYER:
                if (read_layer_section(f, section, &topology) != 0) {
                    free(sections);
                    fclose(f);
                    free_manifest(&manifest);
                    free_alignment(&alignment_info);
                    return NULL;
                }
                break;
            case PSAMC_SECTION_ALIGNMENT:
                if (read_alignment_section(f, section, &alignment_info) != 0) {
                    free(sections);
                    fclose(f);
                    free_manifest(&manifest);
                    free_topology(&topology);
                    return NULL;
                }
                alignment_loaded = true;
                break;
            case PSAMC_SECTION_SAMPLER:
                if (read_sampler_section(f, section, &sampler_defaults) != 0) {
                    /* Non-fatal: use defaults if sampler section is malformed */
                    init_sampler_defaults(&sampler_defaults);
                }
                break;
            default:
                /* Skip unknown sections for forward compatibility */
                break;
        }
    }

    free(sections);

    if (!config_loaded) {
        free_manifest(&manifest);
        free_topology(&topology);
        fclose(f);
        free_alignment(&alignment_info);
        return NULL;
    }

    if (!manifest_loaded) {
        free_topology(&topology);
        fclose(f);
        free_alignment(&alignment_info);
        return NULL;
    }

    fclose(f);

    sha256_hash_t computed;
    if (compute_self_hash_excluding(path, manifest_hash_offset, PSAMC_SOURCE_HASH_SIZE, &computed) != 0) {
        free_manifest(&manifest);
        free_topology(&topology);
        free_alignment(&alignment_info);
        return NULL;
    }

    if (!sha256_equal(&computed, &manifest.self_hash)) {
        fprintf(stderr, "ERROR: .psamc self-hash mismatch (file may be corrupted)\\n");
        free_manifest(&manifest);
        free_topology(&topology);
        free_alignment(&alignment_info);
        return NULL;
    }

    if (verify_integrity && psamc_verify_manifest(&manifest) != 0) {
        free_manifest(&manifest);
        free_topology(&topology);
        free_alignment(&alignment_info);
        return NULL;
    }

    if (topology.layer_count == 0 || topology.base_ref_index >= manifest.num_references) {
        synthesize_topology_from_manifest(&manifest, &topology);
    }

    psamc_composite_t* composite = calloc(1, sizeof(psamc_composite_t));
    if (!composite) {
        free_manifest(&manifest);
        free_topology(&topology);
        free_alignment(&alignment_info);
        return NULL;
    }

    composite->hyperparams = config;
    composite->manifest = manifest;
    composite->topology = topology;
    composite->sampler_defaults = sampler_defaults;
    if (alignment_loaded) {
        composite->alignment = alignment_info;
    } else {
        init_alignment_defaults(&composite->alignment);
    }

    return composite;
}

void psamc_free(psamc_composite_t* composite) {
    if (!composite) {
        return;
    }
    free_manifest(&composite->manifest);
    free_topology(&composite->topology);
    free_alignment(&composite->alignment);
    free(composite);
}

static int fill_model_reference(psamc_model_ref_t* ref, const char* path, const char* model_id) {
    if (!ref || !path) {
        return -1;
    }

    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "ERROR: Failed to stat '%s': %s\n", path, strerror(errno));
        return -1;
    }

    if (psamc_sha256_file(path, &ref->sha256) != 0) {
        fprintf(stderr, "ERROR: Failed to hash '%s'\n", path);
        return -1;
    }

    snprintf(ref->url, PSAMC_MAX_URL_LENGTH, "%s", path);
    ref->size = (uint64_t)st.st_size;
    if (model_id && model_id[0] != '\0') {
        snprintf(ref->model_id, PSAMC_MAX_MODEL_ID, "%s", model_id);
    } else {
        ref->model_id[0] = '\0';
    }
    ref->version.major = 0;
    ref->version.minor = 0;
    ref->version.patch = 0;
    return 0;
}

int psam_composite_save_file(
    const char* path,
    const char* created_by,
    const psamc_hyperparams_t* hyperparams,
    float base_weight,
    const char* base_model_path,
    size_t layer_count,
    const psam_composite_layer_file_t* layers
) {
    if (!path || !base_model_path || (layer_count > 0 && !layers)) {
        return -1;
    }

    psamc_manifest_t manifest;
    memset(&manifest, 0, sizeof(manifest));
    manifest.num_references = (uint32_t)(layer_count + 1);
    manifest.refs = calloc(manifest.num_references, sizeof(psamc_model_ref_t));
    if (!manifest.refs) {
        return -1;
    }

    const psamc_hyperparams_t* config = hyperparams ? hyperparams : &PSAMC_PRESET_BALANCED_CONFIG;

    if (fill_model_reference(&manifest.refs[0], base_model_path, "base") != 0) {
        free(manifest.refs);
        return -1;
    }

    psamc_topology_t topology;
    init_topology_defaults(&topology);
    topology.base_weight = base_weight;
    topology.base_ref_index = 0;
    topology.layer_count = (uint32_t)layer_count;
    if (layer_count > 0) {
        topology.layers = calloc(layer_count, sizeof(psamc_layer_entry_t));
        if (!topology.layers) {
            free(manifest.refs);
            return -1;
        }
    }

    for (size_t i = 0; i < layer_count; ++i) {
        const psam_composite_layer_file_t* desc = &layers[i];
        char generated_id[PSAM_LAYER_ID_MAX];
        const char* layer_id = desc->id && desc->id[0] != '\0' ? desc->id : NULL;
        if (!layer_id) {
            snprintf(generated_id, sizeof(generated_id), "layer-%zu", i);
            layer_id = generated_id;
        }

        if (fill_model_reference(&manifest.refs[i + 1], desc->path, layer_id) != 0) {
            free(manifest.refs);
            free_topology(&topology);
            return -1;
        }

        psamc_layer_entry_t* entry = &topology.layers[i];
        snprintf(entry->layer_id, PSAM_LAYER_ID_MAX, "%s", layer_id);
        entry->weight = desc->weight;
        entry->ref_index = (uint32_t)(i + 1);
    }

    manifest.created_timestamp = (uint64_t)time(NULL);
    if (created_by && created_by[0] != '\0') {
        snprintf(manifest.created_by, PSAMC_CREATED_BY_MAX, "%s", created_by);
    } else {
        snprintf(manifest.created_by, PSAMC_CREATED_BY_MAX, "libpsam");
    }

    psamc_sampler_defaults_t sampler_defaults;
    init_sampler_defaults(&sampler_defaults);

    int rc = psamc_save(path, NULL, config, &manifest, &topology, NULL, &sampler_defaults);

    free(manifest.refs);
    free_topology(&topology);

    return rc;
}
#include <sys/types.h>
