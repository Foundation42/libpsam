/**
 * psam_composite.h - Composite model format (.psamc)
 *
 * Extensible format for model composition with integrity verification,
 * external references, and hyperparameter storage.
 *
 * Format Overview:
 * ┌─────────────────────────────────────────┐
 * │ Header (magic, version, flags)          │
 * ├─────────────────────────────────────────┤
 * │ Section Table (offsets, sizes, types)   │
 * ├─────────────────────────────────────────┤
 * │ Manifest Section                        │
 * │  - References with SHA-256 + semver     │
 * │  - Size verification                    │
 * ├─────────────────────────────────────────┤
 * │ Config Section                          │
 * │  - Hyperparameters                      │
 * │  - Presets (fast/accurate/tiny)         │
 * ├─────────────────────────────────────────┤
 * │ Base Model Section                      │
 * │  - Embedded .psam or reference          │
 * ├─────────────────────────────────────────┤
 * │ Layer Sections (repeatable)             │
 * │  - Layer metadata + model data          │
 * ├─────────────────────────────────────────┤
 * │ Extensions Section (reserved)           │
 * └─────────────────────────────────────────┘
 */

#ifndef PSAM_COMPOSITE_H
#define PSAM_COMPOSITE_H

#include <stdint.h>
#include <stdbool.h>

typedef struct psam_model psam_model_t;

#ifdef __cplusplus
extern "C" {
#endif

/* Magic number: "PSMC" (PSAM Composite) */
#define PSAMC_MAGIC 0x434D5350

/* Current format version */
#define PSAMC_VERSION 1

/* Feature flags (bitmask for optional features) */
#define PSAMC_FLAG_COMPRESSED    (1 << 0)  /* Model data is compressed */
#define PSAMC_FLAG_ENCRYPTED     (1 << 1)  /* Model data is encrypted (reserved) */
#define PSAMC_FLAG_QUANTIZED     (1 << 2)  /* Weights are quantized (reserved) */
#define PSAMC_FLAG_HAS_MANIFEST  (1 << 3)  /* Has integrity manifest */

/* String/field size limits */
#define PSAMC_MAX_URL_LENGTH     256
#define PSAMC_MAX_MODEL_ID       64
#define PSAMC_CREATED_BY_MAX     128
#define PSAMC_SOURCE_HASH_SIZE   32
#define PSAMC_SOURCE_LABEL_MAX   64
#define PSAMC_SOURCE_URI_MAX     256
#define PSAMC_SOURCE_LICENSE_MAX 128

/* Section types */
typedef enum {
    PSAMC_SECTION_MANIFEST = 1,    /* Integrity manifest */
    PSAMC_SECTION_CONFIG = 2,      /* Hyperparameters and presets */
    PSAMC_SECTION_BASE_MODEL = 3,  /* Base model data */
    PSAMC_SECTION_LAYER = 4,       /* Overlay layer */
    PSAMC_SECTION_METADATA = 5,    /* Human-readable metadata (name, desc, author) */
    PSAMC_SECTION_EXTENSIONS = 99, /* Reserved for future use */
} psamc_section_type_t;

/* File header */
typedef struct {
    uint32_t magic;         /* PSAMC_MAGIC */
    uint32_t version;       /* Format version */
    uint32_t flags;         /* Feature flags */
    uint32_t num_sections;  /* Number of sections */
    uint64_t file_size;     /* Total file size for validation */
    uint8_t reserved[16];   /* Reserved for future use */
} psamc_header_t;

/* Section table entry */
typedef struct {
    uint32_t type;          /* Section type */
    uint32_t flags;         /* Section-specific flags */
    uint64_t offset;        /* Offset from start of file */
    uint64_t size;          /* Size in bytes */
    uint8_t reserved[8];    /* Reserved */
} psamc_section_entry_t;

/* SHA-256 hash */
typedef struct {
    uint8_t hash[PSAMC_SOURCE_HASH_SIZE];       /* 256 bits */
} sha256_hash_t;

/* Semantic version */
typedef struct {
    uint16_t major;
    uint16_t minor;
    uint16_t patch;
    uint16_t reserved;      /* For pre-release tags in future */
} semver_t;

/* Model reference (for external models) */
typedef struct {
    char url[PSAMC_MAX_URL_LENGTH];    /* URL or path to external model */
    sha256_hash_t sha256;              /* Expected SHA-256 hash */
    uint64_t size;                     /* Expected size in bytes */
    semver_t version;                  /* Semantic version requirement */
    char model_id[PSAMC_MAX_MODEL_ID]; /* Unique identifier */
    uint8_t reserved[32];              /* Reserved */
} psamc_model_ref_t;

/* Hyperparameter presets */
typedef enum {
    PSAMC_PRESET_CUSTOM = 0,    /* User-defined */
    PSAMC_PRESET_FAST = 1,      /* Fast inference, lower quality */
    PSAMC_PRESET_BALANCED = 2,  /* Default balanced preset */
    PSAMC_PRESET_ACCURATE = 3,  /* High accuracy, slower */
    PSAMC_PRESET_TINY = 4,      /* Minimal memory, fast */
} psamc_preset_t;

/* Hyperparameters configuration */
typedef struct {
    psamc_preset_t preset;      /* Preset identifier */

    /* Core parameters */
    float alpha;                /* Distance decay (default: 0.1) */
    float min_evidence;         /* Minimum edge count (default: 1.0) */
    uint32_t top_k;             /* Top-K predictions (default: 32) */

    /* PPMI parameters */
    bool enable_ppmi;           /* Enable PPMI (default: true) */
    float ppmi_k;               /* PPMI smoothing constant (reserved) */

    /* IDF parameters */
    bool enable_idf;            /* Enable IDF (default: true) */
    float idf_smoothing;        /* IDF smoothing (default: 1.0) */

    /* Top-K per POS class (future extension) */
    uint32_t top_k_function;    /* Top-K for function words (reserved) */
    uint32_t top_k_content;     /* Top-K for content words (reserved) */
    uint32_t top_k_rare;        /* Top-K for rare words (reserved) */

    /* Edge pruning */
    float edge_dropout;         /* Edge dropout rate 0-1 (default: 0.0) */

    uint8_t reserved[32];       /* Reserved for future parameters */
} psamc_hyperparams_t;

/* Optional dataset/source metadata entries */
typedef struct {
    char label[PSAMC_SOURCE_LABEL_MAX];    /* Short name or description */
    char uri[PSAMC_SOURCE_URI_MAX];        /* Reference location or URL */
    char license[PSAMC_SOURCE_LICENSE_MAX];/* License or usage terms */
} psamc_source_t;

/* Manifest for integrity checking */
typedef struct {
    uint32_t num_references;    /* Number of external model references */
    psamc_model_ref_t* refs;    /* Array of references */
    sha256_hash_t self_hash;    /* Hash of this composite (excluding this field) */
    sha256_hash_t source_hash;  /* Hash of training data source */
    uint64_t created_timestamp; /* Unix timestamp */
    char created_by[PSAMC_CREATED_BY_MAX]; /* Creator/tool identification */
    uint32_t source_count;      /* Optional dataset metadata entries */
    psamc_source_t* sources;    /* Array of dataset metadata entries */
    uint8_t reserved[24];       /* Reserved */
} psamc_manifest_t;

/* Layer metadata */
typedef struct {
    char layer_id[64];          /* Unique layer identifier */
    float weight;               /* Layer blending weight */
    bool is_embedded;           /* True if model data is embedded, false if reference */
    union {
        psamc_model_ref_t ref;  /* External reference */
        uint64_t data_offset;   /* Offset to embedded data (within this section) */
    };
    uint8_t reserved[32];       /* Reserved */
} psamc_layer_meta_t;

typedef struct {
    psamc_manifest_t manifest;
    psamc_hyperparams_t hyperparams;
} psamc_composite_t;

/* Preset configurations */
static const psamc_hyperparams_t PSAMC_PRESET_FAST_CONFIG = {
    .preset = PSAMC_PRESET_FAST,
    .alpha = 0.15f,              /* Faster decay */
    .min_evidence = 2.0f,        /* More aggressive pruning */
    .top_k = 16,                 /* Fewer predictions */
    .enable_ppmi = true,
    .enable_idf = false,         /* Skip IDF for speed */
    .edge_dropout = 0.1f,        /* Light pruning */
};

static const psamc_hyperparams_t PSAMC_PRESET_BALANCED_CONFIG = {
    .preset = PSAMC_PRESET_BALANCED,
    .alpha = 0.1f,
    .min_evidence = 1.0f,
    .top_k = 32,
    .enable_ppmi = true,
    .enable_idf = true,
    .edge_dropout = 0.0f,
};

static const psamc_hyperparams_t PSAMC_PRESET_ACCURATE_CONFIG = {
    .preset = PSAMC_PRESET_ACCURATE,
    .alpha = 0.05f,              /* Slower decay, longer context */
    .min_evidence = 0.5f,        /* Keep more edges */
    .top_k = 64,                 /* More predictions */
    .enable_ppmi = true,
    .enable_idf = true,
    .edge_dropout = 0.0f,
};

static const psamc_hyperparams_t PSAMC_PRESET_TINY_CONFIG = {
    .preset = PSAMC_PRESET_TINY,
    .alpha = 0.2f,               /* Fast decay */
    .min_evidence = 3.0f,        /* Aggressive pruning */
    .top_k = 8,                  /* Minimal predictions */
    .enable_ppmi = true,
    .enable_idf = false,
    .edge_dropout = 0.2f,        /* Heavy pruning */
};

/* API Functions */

/**
 * Save a composite model with manifest and integrity checking
 */
int psamc_save(
    const char* path,
    const psam_model_t* base_model,
    const psamc_hyperparams_t* hyperparams,
    const psamc_manifest_t* manifest
);

/**
 * Load a composite model with integrity verification
 *
 * @param path Path to .psamc file
 * @param verify_integrity If true, verify SHA-256 checksums & manifest references
 * @return Composite handle or NULL on error
 */
psamc_composite_t* psamc_load(const char* path, bool verify_integrity);

/**
 * Release resources allocated by psamc_load
 */
void psamc_free(psamc_composite_t* composite);

/**
 * Verify integrity of external references in manifest
 *
 * @return 0 on success, -1 on mismatch
 */
int psamc_verify_manifest(const psamc_manifest_t* manifest);

/**
 * Compute SHA-256 hash of a file
 */
int psamc_sha256_file(const char* path, sha256_hash_t* out_hash);

/**
 * Get hyperparameters for a preset
 */
const psamc_hyperparams_t* psamc_get_preset(psamc_preset_t preset);

#ifdef __cplusplus
}
#endif

#endif /* PSAM_COMPOSITE_H */
