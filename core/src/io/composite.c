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

/* External SHA-256 functions */
extern int sha256_file(const char* path, uint8_t* out_hash);
extern void sha256_data(const uint8_t* data, size_t len, uint8_t* out_hash);

/* Get preset configuration */
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

/* Compute SHA-256 of a file */
int psamc_sha256_file(const char* path, sha256_hash_t* out_hash) {
    return sha256_file(path, out_hash->hash);
}

/* Compare two SHA-256 hashes */
static int sha256_equal(const sha256_hash_t* a, const sha256_hash_t* b) {
    return memcmp(a->hash, b->hash, 32) == 0;
}

/* Verify a single model reference */
static int verify_model_reference(const psamc_model_ref_t* ref) {
    /* Check if file exists and get size */
    FILE* f = fopen(ref->url, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Referenced model not found: %s\n", ref->url);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    uint64_t actual_size = (uint64_t)ftell(f);
    fclose(f);

    /* Verify size */
    if (actual_size != ref->size) {
        fprintf(stderr, "ERROR: Size mismatch for %s\n", ref->url);
        fprintf(stderr, "  Expected: %llu bytes\n", (unsigned long long)ref->size);
        fprintf(stderr, "  Actual:   %llu bytes\n", (unsigned long long)actual_size);
        return -1;
    }

    /* Compute and verify SHA-256 */
    sha256_hash_t actual_hash;
    if (psamc_sha256_file(ref->url, &actual_hash) != 0) {
        fprintf(stderr, "ERROR: Failed to compute hash for %s\n", ref->url);
        return -1;
    }

    if (!sha256_equal(&actual_hash, &ref->sha256)) {
        fprintf(stderr, "ERROR: SHA-256 mismatch for %s\n", ref->url);
        fprintf(stderr, "  This prevents 'works on my machine' bugs.\n");
        fprintf(stderr, "  Expected hash: ");
        for (int i = 0; i < 32; i++) {
            fprintf(stderr, "%02x", ref->sha256.hash[i]);
        }
        fprintf(stderr, "\n  Actual hash:   ");
        for (int i = 0; i < 32; i++) {
            fprintf(stderr, "%02x", actual_hash.hash[i]);
        }
        fprintf(stderr, "\n");
        return -1;
    }

    return 0;
}

/* Verify integrity of all external references */
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

/* Save composite model (stub - full implementation would write sections) */
int psamc_save(
    const char* path,
    const void* base_model,
    const psamc_hyperparams_t* hyperparams,
    const psamc_manifest_t* manifest
) {
    if (!path || !base_model) {
        return -1;
    }

    FILE* f = fopen(path, "wb");
    if (!f) {
        return -1;
    }

    /* Write header */
    psamc_header_t header = {0};
    header.magic = PSAMC_MAGIC;
    header.version = PSAMC_VERSION;
    header.flags = manifest ? PSAMC_FLAG_HAS_MANIFEST : 0;
    header.num_sections = 0;  /* Count sections */

    if (hyperparams) header.num_sections++;
    if (manifest) header.num_sections++;
    header.num_sections++;  /* Base model always present */

    if (fwrite(&header, sizeof(header), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    /* TODO: Write section table and sections */
    /* This is a stub - full implementation would:
     * 1. Build section table
     * 2. Write manifest section
     * 3. Write config section
     * 4. Write base model section
     * 5. Write layer sections
     * 6. Update header with file_size
     */

    fclose(f);
    return 0;
}

/* Load composite model (stub - full implementation would read sections) */
void* psamc_load(const char* path, bool verify_integrity) {
    if (!path) {
        return NULL;
    }

    FILE* f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }

    /* Read and validate header */
    psamc_header_t header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    if (header.magic != PSAMC_MAGIC) {
        fprintf(stderr, "ERROR: Invalid .psamc file (bad magic number)\n");
        fclose(f);
        return NULL;
    }

    if (header.version != PSAMC_VERSION) {
        fprintf(stderr, "ERROR: Unsupported .psamc version %u (expected %u)\n",
                header.version, PSAMC_VERSION);
        fclose(f);
        return NULL;
    }

    /* TODO: Read section table and sections */
    /* Full implementation would:
     * 1. Read section table
     * 2. Find and read manifest section
     * 3. If verify_integrity, call psamc_verify_manifest()
     * 4. Read config section
     * 5. Read base model
     * 6. Read and apply layers
     */

    fclose(f);
    return NULL;
}
