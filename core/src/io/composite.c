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
    const psamc_manifest_t* manifest
) {
    (void)base_model; /* Placeholder for future embedded support */

    if (!path || !hyperparams) {
        return -1;
    }

    uint32_t num_sections = 1; /* config */
    if (manifest) {
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

    sections[section_index].type = PSAMC_SECTION_CONFIG;
    sections[section_index].flags = 0;
    sections[section_index].offset = offset;
    sections[section_index].size = sizeof(psamc_hyperparams_disk_t);
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

    if (write_config_section(f, hyperparams) != 0) {
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

    if (fseeko(f, (off_t)section->offset, SEEK_SET) != 0) {
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
    bool manifest_loaded = false;
    bool config_loaded = false;
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
                    return NULL;
                }
                config_loaded = true;
                break;
            default:
                /* Skip unknown sections for forward compatibility */
                break;
        }
    }

    free(sections);

    if (!config_loaded) {
        free_manifest(&manifest);
        fclose(f);
        return NULL;
    }

    fclose(f);

    if (manifest_loaded) {
        sha256_hash_t computed;
        if (compute_self_hash_excluding(path, manifest_hash_offset, PSAMC_SOURCE_HASH_SIZE, &computed) != 0) {
            free_manifest(&manifest);
            return NULL;
        }

        if (!sha256_equal(&computed, &manifest.self_hash)) {
            fprintf(stderr, "ERROR: .psamc self-hash mismatch (file may be corrupted)\n");
            free_manifest(&manifest);
            return NULL;
        }

        if (verify_integrity && psamc_verify_manifest(&manifest) != 0) {
            free_manifest(&manifest);
            return NULL;
        }
    } else if (verify_integrity) {
        fprintf(stderr, "ERROR: Manifest required for integrity verification.\n");
        free_manifest(&manifest);
        return NULL;
    }

    psamc_composite_t* composite = calloc(1, sizeof(psamc_composite_t));
    if (!composite) {
        free_manifest(&manifest);
        return NULL;
    }

    composite->hyperparams = config;
    composite->manifest = manifest;

    return composite;
}

void psamc_free(psamc_composite_t* composite) {
    if (!composite) {
        return;
    }
    free_manifest(&composite->manifest);
    free(composite);
}
