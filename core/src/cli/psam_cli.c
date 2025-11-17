/**
 * psam_cli.c - Command line interface for libpsam
 *
 * Provides thin utilities for building, composing, predicting, and inspecting PSAM models.
 */

#define _POSIX_C_SOURCE 200809L

#include "psam.h"
#include "psam_composite.h"
#include "psam_vocab_alignment.h"
#include "psam_export.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <math.h>
#include <float.h>

/* Windows compatibility for S_ISDIR */
#ifdef _WIN32
  #ifndef S_ISDIR
    #define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
  #endif
#endif

#define CLI_VERSION "0.1.0"

#define EXIT_OK 0
#define EXIT_BAD_ARGS 2
#define EXIT_FILE_MISSING 3
#define EXIT_CHECKSUM_FAIL 4
#define EXIT_INTERNAL 5

typedef struct {
    char** data;
    size_t size;
    size_t capacity;
} string_list_t;

typedef struct {
    uint32_t* data;
    size_t size;
    size_t capacity;
} u32_list_t;

typedef struct {
    char* token;
    uint32_t id;
} vocab_entry_t;

typedef struct {
    char** id_to_token;
   size_t size;
   vocab_entry_t* entries;
   size_t entry_count;
} vocab_t;

typedef struct {
    psam_composite_t* layered;
    psam_composite_aligned_t* aligned;
} cli_composite_handle_t;

static void cli_composite_handle_destroy(cli_composite_handle_t* handle) {
    if (!handle) {
        return;
    }
    if (handle->aligned) {
        psam_composite_aligned_destroy(handle->aligned);
        handle->aligned = NULL;
    }
    if (handle->layered) {
        psam_composite_destroy(handle->layered);
        handle->layered = NULL;
    }
}

static cli_composite_handle_t cli_composite_handle_load(const char* path, bool verify_integrity) {
    cli_composite_handle_t handle = {0};
    handle.aligned = psam_composite_load_aligned(path, verify_integrity);
    if (handle.aligned) {
        return handle;
    }
    handle.layered = psam_composite_load_file(path, verify_integrity);
    return handle;
}

static bool cli_composite_handle_is_aligned(const cli_composite_handle_t* handle) {
    return handle && handle->aligned != NULL;
}

static bool cli_composite_handle_is_empty(const cli_composite_handle_t* handle) {
    return !handle || (!handle->aligned && !handle->layered);
}

static void print_error(const char* fmt, ...) {
    va_list args;
    fprintf(stderr, "psam: ");
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

static void print_usage(void) {
    printf("psam CLI %s\n\n", CLI_VERSION);
    printf("Usage: psam <command> [options]\n\n");
    printf("Commands:\n");
    printf("  build      Train a PSAM model from text\n");
    printf("  compose    Create a .psamc composite manifest\n");
    printf("  predict    Predict next tokens given context\n");
    printf("  generate   Sample a continuation using top-k/top-p\n");
    printf("  explain    Explain why a candidate token was chosen\n");
    printf("  analyze    Report model statistics\n");
    printf("  inspect    Display model/composite metadata\n");
    printf("  tokenize   Convert text to token IDs using a vocab file\n");
    printf("  ids        Convert token IDs back to text\n");
    printf("\nUse 'psam <command> --help' for command-specific options.\n");
}

/* ==== dynamic arrays ==== */

static int string_list_append(string_list_t* list, const char* value) {
    if (list->size == list->capacity) {
        size_t new_cap = list->capacity ? list->capacity * 2 : 64;
        char** new_data = realloc(list->data, new_cap * sizeof(char*));
        if (!new_data) {
            return -1;
        }
        list->data = new_data;
        list->capacity = new_cap;
    }
    list->data[list->size] = strdup(value);
    if (!list->data[list->size]) {
        return -1;
    }
    list->size++;
    return 0;
}

static void string_list_free(string_list_t* list) {
    if (!list) return;
    for (size_t i = 0; i < list->size; ++i) {
        free(list->data[i]);
    }
    free(list->data);
    list->data = NULL;
    list->size = list->capacity = 0;
}

static int u32_list_append(u32_list_t* list, uint32_t value) {
    if (list->size == list->capacity) {
        size_t new_cap = list->capacity ? list->capacity * 2 : 32;
        uint32_t* new_data = realloc(list->data, new_cap * sizeof(uint32_t));
        if (!new_data) {
            return -1;
        }
        list->data = new_data;
        list->capacity = new_cap;
    }
    list->data[list->size++] = value;
    return 0;
}

static void u32_list_free(u32_list_t* list) {
    if (!list) return;
    free(list->data);
    list->data = NULL;
    list->size = list->capacity = 0;
}

static double rng_uniform(uint64_t* state) {
    *state = (*state * 6364136223846793005ULL + 1442695040888963407ULL);
    return ((*state >> 11) & 0x1FFFFFFFFFFFFFULL) * (1.0 / 9007199254740992.0);
}

/* ==== sampler helpers ==== */

static psam_logit_transform_t parse_logit_transform(const char* str) {
    if (strcmp(str, "raw") == 0) return PSAM_LOGIT_RAW;
    if (strcmp(str, "zscore") == 0) return PSAM_LOGIT_ZSCORE;
    if (strcmp(str, "calibrated") == 0) return PSAM_LOGIT_CALIBRATED;
    if (strcmp(str, "legacy") == 0) return PSAM_LOGIT_LEGACY;
    return PSAM_LOGIT_ZSCORE; /* default */
}

static int sample_prediction(const psam_prediction_t* preds,
                             int num_preds,
                             uint32_t top_k,
                             float temperature,
                             float top_p,
                             uint64_t* rng_state,
                             uint32_t* out_token) {
    if (!preds || num_preds <= 0 || !out_token) {
        return -1;
    }

    int candidate_count = num_preds;
    if (top_k > 0 && (int)top_k < candidate_count) {
        candidate_count = (int)top_k;
    }
    if (candidate_count <= 0) {
        candidate_count = num_preds;
    }

    double* logits = NULL;
    double* probs = malloc((size_t)candidate_count * sizeof(double));
    if (!probs) {
        return -1;
    }

    bool use_calibrated = false;
    for (int i = 0; i < candidate_count; ++i) {
        if (preds[i].calibrated_prob > 0.0f) {
            use_calibrated = true;
            break;
        }
    }

    double sum = 0.0;
    if (use_calibrated) {
        for (int i = 0; i < candidate_count; ++i) {
            probs[i] = preds[i].calibrated_prob;
            sum += probs[i];
        }
    } else {
        logits = malloc((size_t)candidate_count * sizeof(double));
        if (!logits) {
            free(probs);
            return -1;
        }

        double inv_temp = 1.0 / (double)temperature;
        double max_logit = -DBL_MAX;
        for (int i = 0; i < candidate_count; ++i) {
            double logit = (double)preds[i].score * inv_temp;
            logits[i] = logit;
            if (logit > max_logit) {
                max_logit = logit;
            }
        }

        for (int i = 0; i < candidate_count; ++i) {
            double val = exp(logits[i] - max_logit);
            probs[i] = val;
            sum += val;
        }
    }

    if (sum <= 0.0) {
        free(logits);
        free(probs);
        *out_token = preds[0].token;
        return 0;
    }

    for (int i = 0; i < candidate_count; ++i) {
        probs[i] /= sum;
    }

    int trimmed_count = candidate_count;
    if (top_p > 0.0f && top_p < 0.999999f) {
        double cumulative = 0.0;
        trimmed_count = 0;
        for (int i = 0; i < candidate_count; ++i) {
            cumulative += probs[i];
            trimmed_count++;
            if (cumulative >= (double)top_p) {
                break;
            }
        }
        if (trimmed_count < 1) {
            trimmed_count = 1;
        }
        double renorm = 0.0;
        for (int i = 0; i < trimmed_count; ++i) {
            renorm += probs[i];
        }
        if (renorm > 0.0) {
            for (int i = 0; i < trimmed_count; ++i) {
                probs[i] /= renorm;
            }
        }
    }

    double r = rng_uniform(rng_state);
    double cumulative = 0.0;
    for (int i = 0; i < trimmed_count; ++i) {
        cumulative += probs[i];
        if (r <= cumulative || i == trimmed_count - 1) {
            *out_token = preds[i].token;
            free(probs);
            free(logits);
            return 0;
        }
    }

    *out_token = preds[trimmed_count - 1].token;
    free(probs);
    free(logits);
    return 0;
}

/* ==== utility helpers ==== */

static int compare_str_ptr(const void* a, const void* b) {
    const char* const* sa = a;
    const char* const* sb = b;
    return strcmp(*sa, *sb);
}

static int compare_entry_token(const void* a, const void* b) {
    const vocab_entry_t* ea = a;
    const vocab_entry_t* eb = b;
    return strcmp(ea->token, eb->token);
}

static void trim_trailing_newline(char* s) {
    if (!s) return;
    size_t len = strlen(s);
    while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r')) {
        s[--len] = '\0';
    }
}

static bool has_suffix(const char* str, const char* suffix) {
    size_t len = strlen(str);
    size_t slen = strlen(suffix);
    if (len < slen) return false;
    return strcmp(str + len - slen, suffix) == 0;
}

static int parse_uint32(const char* text, uint32_t* out) {
    if (!text || !out) return -1;
    char* endptr = NULL;
    errno = 0;
    unsigned long val = strtoul(text, &endptr, 10);
    if (errno != 0 || endptr == text || *endptr != '\0' || val > UINT32_MAX) {
        return -1;
    }
    *out = (uint32_t)val;
    return 0;
}

static int parse_float(const char* text, float* out) {
    if (!text || !out) return -1;
    char* endptr = NULL;
    errno = 0;
    float val = strtof(text, &endptr);
    if (errno != 0 || endptr == text || *endptr != '\0') {
        return -1;
    }
    *out = val;
    return 0;
}

/* ==== vocabulary ==== */

static void vocab_free(vocab_t* vocab) {
    if (!vocab) return;
    if (vocab->id_to_token) {
        for (size_t i = 0; i < vocab->size; ++i) {
            free(vocab->id_to_token[i]);
        }
        free(vocab->id_to_token);
    }
    if (vocab->entries) {
        for (size_t i = 0; i < vocab->entry_count; ++i) {
            free(vocab->entries[i].token);
        }
        free(vocab->entries);
    }
    vocab->id_to_token = NULL;
    vocab->entries = NULL;
    vocab->size = vocab->entry_count = 0;
}

static int vocab_load(const char* path, vocab_t* vocab) {
    memset(vocab, 0, sizeof(*vocab));
    FILE* f = fopen(path, "r");
    if (!f) {
        print_error("failed to open vocab file '%s': %s", path, strerror(errno));
        return EXIT_FILE_MISSING;
    }

    size_t entry_cap = 64;
    vocab_entry_t* entries = malloc(entry_cap * sizeof(vocab_entry_t));
    if (!entries) {
        fclose(f);
        return EXIT_INTERNAL;
    }

    size_t entry_count = 0;
    uint32_t max_id = 0;
    char buffer[8192];

    while (fgets(buffer, sizeof(buffer), f)) {
        char* tab = strchr(buffer, '\t');
        if (!tab) continue;
        *tab = '\0';
        char* token_str = tab + 1;
        trim_trailing_newline(token_str);
        uint32_t id;
        if (parse_uint32(buffer, &id) != 0) {
            continue;
        }
        if (entry_count == entry_cap) {
            entry_cap *= 2;
            vocab_entry_t* new_entries = realloc(entries, entry_cap * sizeof(vocab_entry_t));
            if (!new_entries) {
                fclose(f);
                for (size_t i = 0; i < entry_count; ++i) free(entries[i].token);
                free(entries);
                return EXIT_INTERNAL;
            }
            entries = new_entries;
        }
        entries[entry_count].id = id;
        entries[entry_count].token = strdup(token_str);
        if (!entries[entry_count].token) {
            fclose(f);
            for (size_t i = 0; i < entry_count; ++i) free(entries[i].token);
            free(entries);
            return EXIT_INTERNAL;
        }
        if (id > max_id) {
            max_id = id;
        }
        entry_count++;
    }

    fclose(f);

    size_t vocab_size = (size_t)max_id + 1;
    char** id_to_token = calloc(vocab_size, sizeof(char*));
    if (!id_to_token) {
        for (size_t i = 0; i < entry_count; ++i) free(entries[i].token);
        free(entries);
        return EXIT_INTERNAL;
    }
    for (size_t i = 0; i < entry_count; ++i) {
        if (entries[i].id < vocab_size) {
            id_to_token[entries[i].id] = strdup(entries[i].token);
        }
    }

    qsort(entries, entry_count, sizeof(vocab_entry_t), compare_entry_token);

    vocab->id_to_token = id_to_token;
    vocab->size = vocab_size;
    vocab->entries = entries;
    vocab->entry_count = entry_count;

    return EXIT_OK;
}

static int vocab_build_from_alignment(const psam_vocab_alignment_t* alignment, vocab_t* vocab) {
    if (!alignment || !vocab) {
        return EXIT_BAD_ARGS;
    }
    if (alignment->unified_vocab_size == 0 || !alignment->unified_tokens) {
        return EXIT_BAD_ARGS;
    }

    memset(vocab, 0, sizeof(*vocab));

    size_t size = alignment->unified_vocab_size;
    char** id_to_token = calloc(size, sizeof(char*));
    vocab_entry_t* entries = malloc(size * sizeof(vocab_entry_t));
    if (!id_to_token || !entries) {
        free(id_to_token);
        free(entries);
        return EXIT_INTERNAL;
    }

    size_t entry_count = 0;
    for (uint32_t i = 0; i < alignment->unified_vocab_size; ++i) {
        const char* token = alignment->unified_tokens[i];
        if (!token) {
            continue;
        }
        char* id_copy = strdup(token);
        char* entry_copy = strdup(token);
        if (!id_copy || !entry_copy) {
            free(id_copy);
            free(entry_copy);
            for (size_t j = 0; j < size; ++j) {
                free(id_to_token[j]);
            }
            free(id_to_token);
            for (size_t j = 0; j < entry_count; ++j) {
                free(entries[j].token);
            }
            free(entries);
            return EXIT_INTERNAL;
        }
        id_to_token[i] = id_copy;
        entries[entry_count].id = i;
        entries[entry_count].token = entry_copy;
        entry_count++;
    }

    if (entry_count == 0) {
        for (size_t j = 0; j < size; ++j) {
            free(id_to_token[j]);
        }
        free(id_to_token);
        free(entries);
        return EXIT_BAD_ARGS;
    }

    qsort(entries, entry_count, sizeof(vocab_entry_t), compare_entry_token);

    vocab->id_to_token = id_to_token;
    vocab->size = size;
    vocab->entries = entries;
    vocab->entry_count = entry_count;
    return EXIT_OK;
}

static int vocab_lookup_id(const vocab_t* vocab, const char* token, uint32_t* out_id) {
    vocab_entry_t key;
    key.token = (char*)token;
    key.id = 0;
    vocab_entry_t* found = bsearch(&key, vocab->entries, vocab->entry_count,
                                   sizeof(vocab_entry_t), compare_entry_token);
    if (!found) return -1;
    *out_id = found->id;
    return 0;
}

static const char* vocab_lookup_token(const vocab_t* vocab, uint32_t id) {
    if (!vocab || id >= vocab->size) return NULL;
    return vocab->id_to_token[id];
}

/* ==== token parsing ==== */

static int parse_ctx_ids(const char* text, u32_list_t* out) {
    char* copy = strdup(text);
    if (!copy) return EXIT_INTERNAL;
    char* token = strtok(copy, ",");
    while (token) {
        while (isspace((unsigned char)*token)) token++;
        uint32_t id;
        if (parse_uint32(token, &id) != 0) {
            free(copy);
            return EXIT_BAD_ARGS;
        }
        if (u32_list_append(out, id) != 0) {
            free(copy);
            return EXIT_INTERNAL;
        }
        token = strtok(NULL, ",");
    }
    free(copy);
    return EXIT_OK;
}

static int tokenize_with_vocab(const vocab_t* vocab, const char* text, u32_list_t* out, bool require_all) {
    char* copy = strdup(text);
    if (!copy) return EXIT_INTERNAL;
    char* token = strtok(copy, " \t\r\n");
    while (token) {
        uint32_t id;
        if (vocab_lookup_id(vocab, token, &id) != 0) {
            if (require_all) {
                print_error("unknown token '%s'", token);
                free(copy);
                return EXIT_BAD_ARGS;
            }
        } else {
            if (u32_list_append(out, id) != 0) {
                free(copy);
                return EXIT_INTERNAL;
            }
        }
        token = strtok(NULL, " \t\r\n");
    }
    free(copy);
    return EXIT_OK;
}

static void print_hex(const uint8_t* data, size_t len, char* out, size_t out_len) {
    size_t needed = len * 2 + 1;
    if (out_len < needed) {
        if (out_len > 0) out[0] = '\0';
        return;
    }
    for (size_t i = 0; i < len; ++i) {
        sprintf(out + i * 2, "%02x", data[i]);
    }
    out[len * 2] = '\0';
}

/* ==== build command ==== */

typedef struct {
    string_list_t inputs;
    const char* output_model;
    const char* output_vocab;
    const char* input_vocab;   /* Optional: pre-built vocabulary TSV */
    uint32_t window;
    uint32_t top_k;
    float alpha;
} build_options_t;

static void build_usage(void) {
    printf("Usage: psam build --input file.txt [--input other.txt] --out model.psam --vocab-out vocab.tsv [options]\n");
    printf("Options:\n");
    printf("  --vocab-in <tsv>    Use pre-built vocabulary (skips vocab discovery)\n");
    printf("  --window <n>        Context window (default 8)\n");
    printf("  --top_k <n>         Top-K predictions stored during inference (default 32)\n");
    printf("  --alpha <f>         Distance decay parameter (default 0.1)\n");
}

/* Load vocabulary from TSV file (format: "id\ttoken\n") */
static int load_vocab_from_tsv(const char* path, char*** out_tokens, size_t* out_count) {
    FILE* f = fopen(path, "r");
    if (!f) {
        print_error("failed to open vocab '%s': %s", path, strerror(errno));
        return EXIT_FILE_MISSING;
    }

    string_list_t temp_tokens = {0};
    char line[4096];
    uint32_t expected_id = 0;

    while (fgets(line, sizeof(line), f)) {
        /* Parse: "id\ttoken\n" */
        char* tab = strchr(line, '\t');
        if (!tab) {
            print_error("malformed vocab line (missing tab): %s", line);
            string_list_free(&temp_tokens);
            fclose(f);
            return EXIT_BAD_ARGS;
        }

        /* Verify ID is sequential - temporarily null-terminate at tab for parsing */
        *tab = '\0';
        uint32_t id = 0;
        if (parse_uint32(line, &id) != 0 || id != expected_id) {
            print_error("vocab IDs must be sequential starting at 0, got %u expected %u", id, expected_id);
            string_list_free(&temp_tokens);
            fclose(f);
            return EXIT_BAD_ARGS;
        }
        *tab = '\t';  /* Restore tab for error messages */
        expected_id++;

        /* Extract token (skip tab, trim newline) */
        char* token_start = tab + 1;
        size_t token_len = strlen(token_start);
        if (token_len > 0 && token_start[token_len - 1] == '\n') {
            token_start[token_len - 1] = '\0';
        }

        if (string_list_append(&temp_tokens, token_start) != 0) {
            string_list_free(&temp_tokens);
            fclose(f);
            return EXIT_INTERNAL;
        }
    }
    fclose(f);

    if (temp_tokens.size == 0) {
        print_error("empty vocabulary file");
        return EXIT_BAD_ARGS;
    }

    *out_tokens = temp_tokens.data;
    *out_count = temp_tokens.size;
    return EXIT_OK;
}

static int read_tokens_from_file(const char* path, string_list_t* tokens) {
    FILE* f = fopen(path, "r");
    if (!f) {
        print_error("failed to open input '%s': %s", path, strerror(errno));
        return EXIT_FILE_MISSING;
    }
    char buffer[4096];
    int rc = EXIT_OK;
    while (fgets(buffer, sizeof(buffer), f)) {
        char* tok = strtok(buffer, " \t\r\n");
        while (tok) {
            if (string_list_append(tokens, tok) != 0) {
                rc = EXIT_INTERNAL;
                goto done;
            }
            tok = strtok(NULL, " \t\r\n");
        }
    }
done:
    fclose(f);
    return rc;
}

static int build_command(int argc, char** argv) {
    build_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.window = 8;
    opts.top_k = 32;
    opts.alpha = 0.1f;

#define RETURN_BUILD(code) do { string_list_free(&opts.inputs); return (code); } while (0)

    for (int i = 2; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--input") == 0) {
            if (++i >= argc) {
                build_usage();
                RETURN_BUILD(EXIT_BAD_ARGS);
            }
            if (string_list_append(&opts.inputs, argv[i]) != 0) {
                RETURN_BUILD(EXIT_INTERNAL);
            }
        } else if (strcmp(arg, "--out") == 0) {
            if (++i >= argc) {
                build_usage();
                RETURN_BUILD(EXIT_BAD_ARGS);
            }
            opts.output_model = argv[i];
        } else if (strcmp(arg, "--vocab-out") == 0) {
            if (++i >= argc) {
                build_usage();
                RETURN_BUILD(EXIT_BAD_ARGS);
            }
            opts.output_vocab = argv[i];
        } else if (strcmp(arg, "--vocab-in") == 0) {
            if (++i >= argc) {
                build_usage();
                RETURN_BUILD(EXIT_BAD_ARGS);
            }
            opts.input_vocab = argv[i];
        } else if (strcmp(arg, "--window") == 0) {
            if (++i >= argc || parse_uint32(argv[i], &opts.window) != 0) {
                build_usage();
                RETURN_BUILD(EXIT_BAD_ARGS);
            }
        } else if (strcmp(arg, "--top_k") == 0) {
            if (++i >= argc || parse_uint32(argv[i], &opts.top_k) != 0) {
                build_usage();
                RETURN_BUILD(EXIT_BAD_ARGS);
            }
        } else if (strcmp(arg, "--alpha") == 0) {
            if (++i >= argc || parse_float(argv[i], &opts.alpha) != 0) {
                build_usage();
                RETURN_BUILD(EXIT_BAD_ARGS);
            }
        } else if (strcmp(arg, "--help") == 0) {
            build_usage();
            RETURN_BUILD(EXIT_OK);
        } else {
            print_error("unknown option '%s'", arg);
            build_usage();
            RETURN_BUILD(EXIT_BAD_ARGS);
        }
    }

    if (opts.inputs.size == 0 || !opts.output_model) {
        print_error("build requires --input and --out");
        build_usage();
        RETURN_BUILD(EXIT_BAD_ARGS);
    }

    /* If using pre-built vocab, --vocab-out is optional (for convenience, to copy it) */
    /* If discovering vocab, --vocab-out is required */
    if (!opts.input_vocab && !opts.output_vocab) {
        print_error("build requires either --vocab-in (use existing) or --vocab-out (create new)");
        build_usage();
        RETURN_BUILD(EXIT_BAD_ARGS);
    }

    string_list_t tokens = {0};
    for (size_t idx = 0; idx < opts.inputs.size; ++idx) {
        int rc = read_tokens_from_file(opts.inputs.data[idx], &tokens);
        if (rc != EXIT_OK) {
            string_list_free(&tokens);
            RETURN_BUILD(rc);
        }
    }

    if (tokens.size == 0) {
        string_list_free(&tokens);
        print_error("no tokens found in input");
        RETURN_BUILD(EXIT_BAD_ARGS);
    }

    char** sorted = malloc(tokens.size * sizeof(char*));
    if (!sorted) {
        string_list_free(&tokens);
        RETURN_BUILD(EXIT_INTERNAL);
    }
    memcpy(sorted, tokens.data, tokens.size * sizeof(char*));
    qsort(sorted, tokens.size, sizeof(char*), compare_str_ptr);

    size_t unique_count = 0;
    for (size_t i = 0; i < tokens.size; ++i) {
        if (i == 0 || strcmp(sorted[i], sorted[i - 1]) != 0) {
            sorted[unique_count++] = sorted[i];
        }
    }

    /* Build or load vocabulary */
    char** id_to_token = NULL;
    size_t vocab_size = 0;

    if (opts.input_vocab) {
        /* Load pre-built vocabulary from TSV */
        int rc = load_vocab_from_tsv(opts.input_vocab, &id_to_token, &vocab_size);
        if (rc != EXIT_OK) {
            free(sorted);
            string_list_free(&tokens);
            RETURN_BUILD(rc);
        }
        fprintf(stderr, "INFO: Loaded %zu tokens from vocabulary '%s'\n", vocab_size, opts.input_vocab);
        free(sorted);  /* Don't need sorted array if using pre-built vocab */
        sorted = NULL;
    } else {
        /* Discover vocabulary from input - use sorted array */
        if (unique_count == 0) {
            free(sorted);
            string_list_free(&tokens);
            print_error("no unique tokens discovered");
            RETURN_BUILD(EXIT_BAD_ARGS);
        }

        vocab_size = unique_count;
        id_to_token = malloc(vocab_size * sizeof(char*));
        if (!id_to_token) {
            free(sorted);
            string_list_free(&tokens);
            RETURN_BUILD(EXIT_INTERNAL);
        }
        for (size_t i = 0; i < vocab_size; ++i) {
            id_to_token[i] = strdup(sorted[i]);
            if (!id_to_token[i]) {
                for (size_t j = 0; j < i; ++j) free(id_to_token[j]);
                free(id_to_token);
                free(sorted);
                string_list_free(&tokens);
                RETURN_BUILD(EXIT_INTERNAL);
            }
        }
    }

    /* Create model with discovered or loaded vocabulary */
    psam_config_t config = {
        .vocab_size = (uint32_t)vocab_size,
        .window = opts.window,
        .top_k = opts.top_k,
        .alpha = opts.alpha,
        .min_evidence = 1.0f,
        .enable_idf = true,
        .enable_ppmi = true,
        .edge_dropout = 0.0f
    };

    psam_model_t* model = psam_create_with_config(&config);
    if (!model) {
        print_error("failed to create model (vocab_size=%u)", config.vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) free(id_to_token[i]);
        free(id_to_token);
        if (sorted) free(sorted);
        string_list_free(&tokens);
        RETURN_BUILD(EXIT_INTERNAL);
    }

    /* Convert tokens to IDs */
    uint32_t* sequence = malloc(tokens.size * sizeof(uint32_t));
    if (!sequence) {
        psam_destroy(model);
        for (size_t j = 0; j < vocab_size; ++j) free(id_to_token[j]);
        free(id_to_token);
        if (sorted) free(sorted);
        string_list_free(&tokens);
        RETURN_BUILD(EXIT_INTERNAL);
    }

    size_t unknown_tokens = 0;
    for (size_t i = 0; i < tokens.size; ++i) {
        char* tok = tokens.data[i];

        /* Search for token in vocabulary */
        char** found = NULL;
        if (sorted) {
            /* vocab discovery path: use sorted array */
            found = bsearch(&tok, sorted, vocab_size, sizeof(char*), compare_str_ptr);
        } else {
            /* pre-loaded vocab path: search id_to_token array */
            for (size_t v = 0; v < vocab_size; ++v) {
                if (strcmp(id_to_token[v], tok) == 0) {
                    found = &id_to_token[v];
                    break;
                }
            }
        }

        if (!found) {
            if (unknown_tokens == 0) {
                print_error("token '%s' not found in vocabulary (first occurrence)", tok);
            }
            unknown_tokens++;
            /* For now, skip unknown tokens - could also map to UNK */
            sequence[i] = UINT32_MAX;  /* Sentinel for unknown */
            continue;
        }

        uint32_t id;
        if (sorted) {
            id = (uint32_t)(found - sorted);
        } else {
            id = (uint32_t)(found - id_to_token);
        }
        sequence[i] = id;
    }

    if (unknown_tokens > 0) {
        print_error("%zu unknown tokens found (not in vocabulary)", unknown_tokens);
        free(sequence);
        psam_destroy(model);
        for (size_t j = 0; j < vocab_size; ++j) free(id_to_token[j]);
        free(id_to_token);
        if (sorted) free(sorted);
        string_list_free(&tokens);
        RETURN_BUILD(EXIT_BAD_ARGS);
    }

    psam_error_t batch_err = psam_train_batch(model, sequence, tokens.size);
    free(sequence);
    if (batch_err != PSAM_OK) {
        print_error("psam_train_batch failed (%d)", batch_err);
        psam_destroy(model);
        for (size_t j = 0; j < vocab_size; ++j) free(id_to_token[j]);
        free(id_to_token);
        if (sorted) free(sorted);
        string_list_free(&tokens);
        RETURN_BUILD(EXIT_INTERNAL);
    }

    psam_error_t finalize_err = psam_finalize_training(model);
    if (finalize_err != PSAM_OK) {
        print_error("psam_finalize_training failed (%d)", finalize_err);
        psam_destroy(model);
        for (size_t j = 0; j < vocab_size; ++j) free(id_to_token[j]);
        free(id_to_token);
        if (sorted) free(sorted);
        string_list_free(&tokens);
        RETURN_BUILD(EXIT_INTERNAL);
    }

    psam_provenance_t prov = {0};
    prov.created_timestamp = (uint64_t)time(NULL);
    snprintf(prov.created_by, PSAM_CREATED_BY_MAX, "psam-cli/%s", CLI_VERSION);
    if (opts.inputs.size > 0) {
        sha256_hash_t input_hash;
        if (psamc_sha256_file(opts.inputs.data[0], &input_hash) == 0) {
            memcpy(prov.source_hash, input_hash.hash, PSAM_SOURCE_HASH_SIZE);
        }
    }
    if (psam_set_provenance(model, &prov) != PSAM_OK) {
        print_error("failed to set provenance");
        psam_destroy(model);
        for (size_t j = 0; j < vocab_size; ++j) free(id_to_token[j]);
        free(id_to_token);
        if (sorted) free(sorted);
        string_list_free(&tokens);
        RETURN_BUILD(EXIT_INTERNAL);
    }

    psam_error_t save_err = psam_save(model, opts.output_model);
    if (save_err != PSAM_OK) {
        print_error("psam_save failed (%d)", save_err);
        psam_destroy(model);
        for (size_t j = 0; j < vocab_size; ++j) free(id_to_token[j]);
        free(id_to_token);
        if (sorted) free(sorted);
        string_list_free(&tokens);
        RETURN_BUILD(EXIT_INTERNAL);
    }

    /* Save vocabulary if output path specified */
    if (opts.output_vocab) {
        FILE* vocab_file = fopen(opts.output_vocab, "w");
        if (!vocab_file) {
            print_error("failed to write vocab '%s': %s", opts.output_vocab, strerror(errno));
            psam_destroy(model);
            for (size_t j = 0; j < vocab_size; ++j) free(id_to_token[j]);
            free(id_to_token);
            if (sorted) free(sorted);
            string_list_free(&tokens);
            RETURN_BUILD(EXIT_INTERNAL);
        }
        for (size_t i = 0; i < vocab_size; ++i) {
            fprintf(vocab_file, "%zu\t%s\n", i, id_to_token[i]);
        }
        fclose(vocab_file);
    }

    printf("{\"status\":\"ok\",\"model\":\"%s\",\"vocab\":\"%s\",\"tokens\":%zu,\"vocab_size\":%zu}\n",
           opts.output_model, opts.output_vocab ? opts.output_vocab : opts.input_vocab, tokens.size, vocab_size);

    psam_destroy(model);
    for (size_t i = 0; i < vocab_size; ++i) free(id_to_token[i]);
    free(id_to_token);
    if (sorted) free(sorted);
    string_list_free(&tokens);
    string_list_free(&opts.inputs);
    return EXIT_OK;

#undef RETURN_BUILD
}

/* ==== predict command ==== */

typedef struct {
    const char* model_path;
    const char* vocab_path;
    const char* ctx_ids;
    const char* context_text;
    uint32_t top_k;
    float temperature;
    float top_p;
    psam_logit_transform_t logit_transform;
    bool pretty;
} predict_options_t;

static void predict_usage(void) {
    printf("Usage: psam predict --model model.psam (--ctx-ids 1,2,3 | --context \"text\" [--vocab vocab.tsv])\n");
    printf("       [--top_k 5] [--temperature 1.0] [--top_p 0.95] [--logit-transform zscore|raw|legacy|calibrated] [--pretty]\n");
    printf("\nAliases: --prompt can be used instead of --context\n");
    printf("       When targeting aligned composites, the CLI auto-loads the unified vocabulary.\n");
    printf("       Provide --vocab for raw .psam models or composites without vocab metadata.\n");
}

typedef struct {
    const char* model_path;
    const char* vocab_path;
    const char* ctx_ids;
    const char* context_text;
    uint32_t count;
    uint32_t top_k;
    float top_p;
    float temperature;
    psam_logit_transform_t logit_transform;
    uint32_t seed;
    bool seed_provided;
    bool pretty;
    bool quiet;
} generate_options_t;

static void generate_usage(void) {
    printf("Usage: psam generate --model model.psam (--ctx-ids 1,2,3 | --context \"text\" [--vocab vocab.tsv])\n");
    printf("       [--count 32] [--top_k 16] [--top_p 0.95] [--temperature 1.0] [--logit-transform zscore|raw|legacy|calibrated]\n");
    printf("       [--seed N] [--pretty] [--quiet]\n");
    printf("\nAliases: --prompt can be used instead of --context\n");
    printf("       Aligned composites saved with vocab metadata can generate from prompts without --vocab.\n");
    printf("       Supply --vocab for standalone .psam models or unlabeled composites, or use --ctx-ids.\n");
}

static int generate_command(int argc, char** argv) {
    generate_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.count = 32;
    opts.top_k = 16;
    opts.top_p = 0.95f;
    opts.temperature = 1.0f;
    opts.logit_transform = PSAM_LOGIT_ZSCORE;

    for (int i = 2; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--model") == 0) {
            if (++i >= argc) { generate_usage(); return EXIT_BAD_ARGS; }
            opts.model_path = argv[i];
        } else if (strcmp(arg, "--ctx-ids") == 0) {
            if (++i >= argc) { generate_usage(); return EXIT_BAD_ARGS; }
            opts.ctx_ids = argv[i];
        } else if (strcmp(arg, "--context") == 0 || strcmp(arg, "--prompt") == 0) {
            if (++i >= argc) { generate_usage(); return EXIT_BAD_ARGS; }
            opts.context_text = argv[i];
        } else if (strcmp(arg, "--vocab") == 0) {
            if (++i >= argc) { generate_usage(); return EXIT_BAD_ARGS; }
            opts.vocab_path = argv[i];
        } else if (strcmp(arg, "--count") == 0) {
            if (++i >= argc || parse_uint32(argv[i], &opts.count) != 0 || opts.count == 0) {
                generate_usage();
                return EXIT_BAD_ARGS;
            }
        } else if (strcmp(arg, "--top_k") == 0) {
            if (++i >= argc || parse_uint32(argv[i], &opts.top_k) != 0) {
                generate_usage();
                return EXIT_BAD_ARGS;
            }
        } else if (strcmp(arg, "--top_p") == 0) {
            if (++i >= argc || parse_float(argv[i], &opts.top_p) != 0 || opts.top_p <= 0.0f || opts.top_p > 1.0f) {
                generate_usage();
                return EXIT_BAD_ARGS;
            }
        } else if (strcmp(arg, "--temperature") == 0) {
            if (++i >= argc || parse_float(argv[i], &opts.temperature) != 0 || opts.temperature <= 0.0f) {
                generate_usage();
                return EXIT_BAD_ARGS;
            }
        } else if (strcmp(arg, "--logit-transform") == 0) {
            if (++i >= argc) {
                generate_usage();
                return EXIT_BAD_ARGS;
            }
            opts.logit_transform = parse_logit_transform(argv[i]);
        } else if (strcmp(arg, "--seed") == 0) {
            if (++i >= argc || parse_uint32(argv[i], &opts.seed) != 0) {
                generate_usage();
                return EXIT_BAD_ARGS;
            }
            opts.seed_provided = true;
        } else if (strcmp(arg, "--pretty") == 0) {
            opts.pretty = true;
        } else if (strcmp(arg, "--quiet") == 0) {
            opts.quiet = true;
        } else if (strcmp(arg, "--help") == 0) {
            generate_usage();
            return EXIT_OK;
        } else if (strcmp(arg, "--json") == 0) {
            /* default */
        } else {
            print_error("unknown option '%s'", arg);
            generate_usage();
            return EXIT_BAD_ARGS;
        }
    }

    if (!opts.model_path) {
        print_error("generate requires --model");
        generate_usage();
        return EXIT_BAD_ARGS;
    }

    if (!opts.ctx_ids && !opts.context_text) {
        print_error("generate requires either --ctx-ids or --context");
        generate_usage();
        return EXIT_BAD_ARGS;
    }
    bool is_composite = has_suffix(opts.model_path, ".psamc");
    vocab_t vocab = {0};
    u32_list_t context = {0};
    u32_list_t generated = {0};
    psam_model_t* model = NULL;
    cli_composite_handle_t composite_handle = (cli_composite_handle_t){0};
    psam_prediction_t* preds = NULL;
    int exit_code = EXIT_OK;

    if (is_composite) {
        composite_handle = cli_composite_handle_load(opts.model_path, false);
        if (cli_composite_handle_is_empty(&composite_handle)) {
            print_error("failed to load composite model '%s'", opts.model_path);
            exit_code = EXIT_FILE_MISSING;
            goto cleanup;
        }
    } else {
        model = psam_load(opts.model_path);
        if (!model) {
            print_error("failed to load model '%s'", opts.model_path);
            exit_code = EXIT_FILE_MISSING;
            goto cleanup;
        }
    }

    if (opts.context_text) {
        int rc = EXIT_OK;
        if (opts.vocab_path) {
            rc = vocab_load(opts.vocab_path, &vocab);
        } else if (cli_composite_handle_is_aligned(&composite_handle)) {
            const char* inferred_path = psam_composite_aligned_get_unified_vocab_path(composite_handle.aligned);
            if (inferred_path) {
                rc = vocab_load(inferred_path, &vocab);
            }
            if (!inferred_path || rc != EXIT_OK) {
                const psam_vocab_alignment_t* alignment = psam_composite_aligned_get_alignment(composite_handle.aligned);
                rc = alignment ? vocab_build_from_alignment(alignment, &vocab) : EXIT_BAD_ARGS;
                if (rc == EXIT_BAD_ARGS) {
                    print_error("composite does not expose unified vocabulary; supply --vocab");
                }
            }
        } else {
            print_error("--context requires --vocab");
            exit_code = EXIT_BAD_ARGS;
            goto cleanup;
        }
        if (rc != EXIT_OK) {
            exit_code = rc;
            goto cleanup;
        }
    } else if (opts.vocab_path) {
        int rc = vocab_load(opts.vocab_path, &vocab);
        if (rc != EXIT_OK) {
            exit_code = rc;
            goto cleanup;
        }
    }

    if (opts.ctx_ids) {
        int rc = parse_ctx_ids(opts.ctx_ids, &context);
        if (rc != EXIT_OK) {
            exit_code = rc;
            goto cleanup;
        }
    } else {
        int rc = tokenize_with_vocab(&vocab, opts.context_text, &context, true);
        if (rc != EXIT_OK) {
            exit_code = rc;
            goto cleanup;
        }
    }

    uint32_t predict_cap = opts.top_k ? opts.top_k : 64;
    if (predict_cap < 8) predict_cap = 8;
    preds = calloc(predict_cap, sizeof(psam_prediction_t));
    if (!preds) {
        exit_code = EXIT_INTERNAL;
        goto cleanup;
    }

    uint64_t rng_state = opts.seed_provided ? (uint64_t)opts.seed : ((uint64_t)time(NULL) ^ 0x9e3779b97f4a7c15ULL);
    if (rng_state == 0) rng_state = 1;
    bool first_output = true;

    /* Configure sampler */
    psam_sampler_t sampler = {
        .transform = opts.logit_transform,
        .temperature = opts.temperature,
        .top_k = (int)opts.top_k,
        .top_p = opts.top_p,
        .seed = rng_state
    };

    for (uint32_t step = 0; step < opts.count; ++step) {
        /* Use new sampler API */
        sampler.seed = rng_state;  /* Update seed for each step */
        int num_preds;
        if (is_composite) {
            if (cli_composite_handle_is_aligned(&composite_handle)) {
                num_preds = psam_composite_aligned_predict_with_sampler(
                    composite_handle.aligned,
                    context.data,
                    context.size,
                    &sampler,
                    preds,
                    predict_cap
                );
            } else {
                num_preds = psam_composite_predict_with_sampler(
                    composite_handle.layered,
                    context.data,
                    context.size,
                    &sampler,
                    preds,
                    predict_cap
                );
            }
        } else {
            num_preds = psam_predict_with_sampler(model, context.data, context.size, &sampler, preds, predict_cap);
        }
        if (num_preds <= 0) {
            break;
        }

        /* Sample from the computed probabilities */
        uint32_t next_token = 0;
        if (sample_prediction(preds, num_preds, opts.top_k, opts.temperature, opts.top_p, &rng_state, &next_token) != 0) {
            break;
        }

        if (u32_list_append(&generated, next_token) != 0 || u32_list_append(&context, next_token) != 0) {
            print_error("out of memory while generating");
            break;
        }

        if (opts.quiet) {
            if (vocab.size) {
                const char* tok = vocab_lookup_token(&vocab, next_token);
                if (!tok) {
                    printf("%s%u", first_output ? "" : " ", next_token);
                } else {
                    printf("%s%s", first_output ? "" : " ", tok);
                }
            } else {
                printf("%s%u", first_output ? "" : " ", next_token);
            }
            first_output = false;
        }
    }

    if (opts.quiet) {
        printf("\n");
    } else {
        if (opts.pretty) {
            printf("{\n  \"generated\": {\n    \"ids\": [");
        } else {
            printf("{\"generated\":{\"ids\":[");
        }

        for (size_t i = 0; i < generated.size; ++i) {
            if (opts.pretty) {
                printf("%s%u", i == 0 ? "" : ", ", generated.data[i]);
            } else {
                printf("%s%u", i == 0 ? "" : ",", generated.data[i]);
            }
        }

        if (opts.pretty) {
            printf("]");
        } else {
            printf("]");
        }

        if (vocab.size) {
            if (opts.pretty) {
                printf(",\n    \"tokens\": [");
            } else {
                printf(",\"tokens\":[");
            }
            for (size_t i = 0; i < generated.size; ++i) {
                const char* tok = vocab_lookup_token(&vocab, generated.data[i]);
                if (!tok) tok = "<UNK>";
                if (opts.pretty) {
                    printf("%s\"%s\"", i == 0 ? "" : ", ", tok);
                } else {
                    printf("%s\"%s\"", i == 0 ? "" : ",", tok);
                }
            }
            if (opts.pretty) {
                printf("]");
            } else {
                printf("]");
            }
        }

        if (opts.pretty) {
            printf("\n  }\n}\n");
        } else {
            printf("}}\n");
        }
    }

    exit_code = EXIT_OK;

cleanup:
    free(preds);
    if (model) psam_destroy(model);
    cli_composite_handle_destroy(&composite_handle);
    u32_list_free(&generated);
    vocab_free(&vocab);
    u32_list_free(&context);
    return exit_code;
}

static int predict_command(int argc, char** argv) {
    predict_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.top_k = 5;
    opts.temperature = 1.0f;
    opts.top_p = 0.95f;
    opts.logit_transform = PSAM_LOGIT_ZSCORE;

    for (int i = 2; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--model") == 0) {
            if (++i >= argc) { predict_usage(); return EXIT_BAD_ARGS; }
            opts.model_path = argv[i];
        } else if (strcmp(arg, "--ctx-ids") == 0) {
            if (++i >= argc) { predict_usage(); return EXIT_BAD_ARGS; }
            opts.ctx_ids = argv[i];
        } else if (strcmp(arg, "--context") == 0 || strcmp(arg, "--prompt") == 0) {
            if (++i >= argc) { predict_usage(); return EXIT_BAD_ARGS; }
            opts.context_text = argv[i];
        } else if (strcmp(arg, "--vocab") == 0) {
            if (++i >= argc) { predict_usage(); return EXIT_BAD_ARGS; }
            opts.vocab_path = argv[i];
        } else if (strcmp(arg, "--top_k") == 0) {
            if (++i >= argc || parse_uint32(argv[i], &opts.top_k) != 0) {
                predict_usage();
                return EXIT_BAD_ARGS;
            }
        } else if (strcmp(arg, "--temperature") == 0) {
            if (++i >= argc || parse_float(argv[i], &opts.temperature) != 0 || opts.temperature <= 0.0f) {
                predict_usage();
                return EXIT_BAD_ARGS;
            }
        } else if (strcmp(arg, "--top_p") == 0) {
            if (++i >= argc || parse_float(argv[i], &opts.top_p) != 0 || opts.top_p <= 0.0f || opts.top_p > 1.0f) {
                predict_usage();
                return EXIT_BAD_ARGS;
            }
        } else if (strcmp(arg, "--logit-transform") == 0) {
            if (++i >= argc) {
                predict_usage();
                return EXIT_BAD_ARGS;
            }
            opts.logit_transform = parse_logit_transform(argv[i]);
        } else if (strcmp(arg, "--pretty") == 0) {
            opts.pretty = true;
        } else if (strcmp(arg, "--json") == 0) {
            /* default */
        } else if (strcmp(arg, "--help") == 0) {
            predict_usage();
            return EXIT_OK;
        } else {
            print_error("unknown option '%s'", arg);
            predict_usage();
            return EXIT_BAD_ARGS;
        }
    }

    if (!opts.model_path) {
        print_error("predict requires --model");
        predict_usage();
        return EXIT_BAD_ARGS;
    }

    if (!opts.ctx_ids && !opts.context_text) {
        print_error("predict requires either --ctx-ids or --context");
        predict_usage();
        return EXIT_BAD_ARGS;
    }

    bool is_composite = has_suffix(opts.model_path, ".psamc");
    vocab_t vocab = {0};
    u32_list_t context = {0};
    psam_model_t* model = NULL;
    cli_composite_handle_t composite_handle = (cli_composite_handle_t){0};
    psam_prediction_t* preds = NULL;
    int exit_code = EXIT_OK;

    if (is_composite) {
        composite_handle = cli_composite_handle_load(opts.model_path, false);
        if (cli_composite_handle_is_empty(&composite_handle)) {
            print_error("failed to load composite model '%s'", opts.model_path);
            exit_code = EXIT_FILE_MISSING;
            goto cleanup;
        }
    } else {
        model = psam_load(opts.model_path);
        if (!model) {
            print_error("failed to load model '%s'", opts.model_path);
            exit_code = EXIT_FILE_MISSING;
            goto cleanup;
        }
    }

    if (opts.context_text) {
        int rc = EXIT_OK;
        if (opts.vocab_path) {
            rc = vocab_load(opts.vocab_path, &vocab);
        } else if (cli_composite_handle_is_aligned(&composite_handle)) {
            const char* inferred_path = psam_composite_aligned_get_unified_vocab_path(composite_handle.aligned);
            if (inferred_path) {
                rc = vocab_load(inferred_path, &vocab);
            }
            if (!inferred_path || rc != EXIT_OK) {
                const psam_vocab_alignment_t* alignment = psam_composite_aligned_get_alignment(composite_handle.aligned);
                rc = alignment ? vocab_build_from_alignment(alignment, &vocab) : EXIT_BAD_ARGS;
                if (rc == EXIT_BAD_ARGS) {
                    print_error("composite does not expose unified vocabulary; supply --vocab");
                }
            }
        } else {
            print_error("--context requires --vocab");
            exit_code = EXIT_BAD_ARGS;
            goto cleanup;
        }
        if (rc != EXIT_OK) {
            exit_code = rc;
            goto cleanup;
        }
    } else if (opts.vocab_path) {
        int rc = vocab_load(opts.vocab_path, &vocab);
        if (rc != EXIT_OK) {
            exit_code = rc;
            goto cleanup;
        }
    }

    if (opts.ctx_ids) {
        int rc = parse_ctx_ids(opts.ctx_ids, &context);
        if (rc != EXIT_OK) {
            exit_code = rc;
            goto cleanup;
        }
    } else if (opts.context_text) {
        int rc = tokenize_with_vocab(&vocab, opts.context_text, &context, true);
        if (rc != EXIT_OK) {
            exit_code = rc;
            goto cleanup;
        }
    }

    if (context.size == 0) {
        print_error("empty context");
        exit_code = EXIT_BAD_ARGS;
        goto cleanup;
    }

    uint32_t predict_count = opts.top_k ? opts.top_k : 16;
    preds = calloc(predict_count, sizeof(psam_prediction_t));
    if (!preds) {
        exit_code = EXIT_INTERNAL;
        goto cleanup;
    }

    /* Configure sampler */
    psam_sampler_t sampler = {
        .transform = opts.logit_transform,
        .temperature = opts.temperature,
        .top_k = (int)opts.top_k,
        .top_p = opts.top_p,
        .seed = 42
    };

    int num_preds;
    if (is_composite) {
        if (cli_composite_handle_is_aligned(&composite_handle)) {
            num_preds = psam_composite_aligned_predict_with_sampler(
                composite_handle.aligned,
                context.data,
                context.size,
                &sampler,
                preds,
                predict_count
            );
        } else {
            num_preds = psam_composite_predict_with_sampler(
                composite_handle.layered,
                context.data,
                context.size,
                &sampler,
                preds,
                predict_count
            );
        }
    } else {
        num_preds = psam_predict_with_sampler(model, context.data, context.size, &sampler, preds, predict_count);
    }
    if (num_preds < 0) {
        print_error("predict failed (%d)", num_preds);
        exit_code = EXIT_INTERNAL;
        goto cleanup;
    }

    if (opts.pretty) {
        printf("{\n  \"predictions\": [\n");
    } else {
        printf("{\"predictions\":[");
    }
    for (int i = 0; i < num_preds; ++i) {
        const char* token_str = vocab.size ? vocab_lookup_token(&vocab, preds[i].token) : NULL;
        if (opts.pretty) {
            printf("    {\"id\":%u", preds[i].token);
            if (token_str) {
                printf(", \"token\":\"%s\"", token_str);
            }
            printf(", \"score\":%.6f", preds[i].score);
            printf(", \"raw_strength\":%.6f", preds[i].raw_strength);
            printf(", \"support_count\":%u", (unsigned)preds[i].support_count);
            if (preds[i].calibrated_prob > 0.0f) {
                printf(", \"prob\":%.6f", preds[i].calibrated_prob);
            }
            printf("}");
            if (i + 1 < num_preds) printf(",\n"); else printf("\n");
        } else {
            printf("{\"id\":%u", preds[i].token);
            if (token_str) {
                printf(",\"token\":\"%s\"", token_str);
            }
            printf(",\"score\":%.6f", preds[i].score);
            printf(",\"raw_strength\":%.6f", preds[i].raw_strength);
            printf(",\"support_count\":%u", (unsigned)preds[i].support_count);
            if (preds[i].calibrated_prob > 0.0f) {
                printf(",\"prob\":%.6f", preds[i].calibrated_prob);
            }
            printf("}");
            if (i + 1 < num_preds) printf(",");
        }
    }
    if (opts.pretty) {
        printf("  ]\n}\n");
    } else {
        printf("]}\n");
    }
    exit_code = EXIT_OK;

cleanup:
    free(preds);
    if (model) psam_destroy(model);
    cli_composite_handle_destroy(&composite_handle);
    vocab_free(&vocab);
    u32_list_free(&context);
    return exit_code;
}

/* ==== explain command ==== */

typedef struct {
    const char* model_path;
    const char* vocab_path;
    const char* ctx_ids;
    const char* context_text;
    const char* candidate_token;
    const char* candidate_id;
    uint32_t topN;
    bool pretty;
} explain_options_t;

static void explain_usage(void) {
    printf("Usage: psam explain --model model.psam (--ctx-ids 1,2 | --context \"text\" --vocab vocab.tsv)\n");
    printf("                    (--candidate-id ID | --candidate TOKEN) [--topN 16] [--pretty]\n");
}

static int explain_command(int argc, char** argv) {
    explain_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.topN = 16;

    for (int i = 2; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--model") == 0) {
            if (++i >= argc) { explain_usage(); return EXIT_BAD_ARGS; }
            opts.model_path = argv[i];
        } else if (strcmp(arg, "--ctx-ids") == 0) {
            if (++i >= argc) { explain_usage(); return EXIT_BAD_ARGS; }
            opts.ctx_ids = argv[i];
        } else if (strcmp(arg, "--context") == 0) {
            if (++i >= argc) { explain_usage(); return EXIT_BAD_ARGS; }
            opts.context_text = argv[i];
        } else if (strcmp(arg, "--vocab") == 0) {
            if (++i >= argc) { explain_usage(); return EXIT_BAD_ARGS; }
            opts.vocab_path = argv[i];
        } else if (strcmp(arg, "--candidate") == 0) {
            if (++i >= argc) { explain_usage(); return EXIT_BAD_ARGS; }
            opts.candidate_token = argv[i];
        } else if (strcmp(arg, "--candidate-id") == 0) {
            if (++i >= argc) { explain_usage(); return EXIT_BAD_ARGS; }
            opts.candidate_id = argv[i];
        } else if (strcmp(arg, "--topN") == 0) {
            if (++i >= argc || parse_uint32(argv[i], &opts.topN) != 0) {
                explain_usage();
                return EXIT_BAD_ARGS;
            }
        } else if (strcmp(arg, "--pretty") == 0) {
            opts.pretty = true;
        } else if (strcmp(arg, "--help") == 0) {
            explain_usage();
            return EXIT_OK;
        } else {
            print_error("unknown option '%s'", arg);
            explain_usage();
            return EXIT_BAD_ARGS;
        }
    }

    if (!opts.model_path) {
        print_error("explain requires --model");
        explain_usage();
        return EXIT_BAD_ARGS;
    }

    if (!opts.ctx_ids && !opts.context_text) {
        print_error("explain requires either --ctx-ids or --context");
        explain_usage();
        return EXIT_BAD_ARGS;
    }

    if (!opts.candidate_id && !opts.candidate_token) {
        print_error("explain requires --candidate-id or --candidate");
        explain_usage();
        return EXIT_BAD_ARGS;
    }

    vocab_t vocab = {0};
    if ((opts.context_text || opts.candidate_token) && !opts.vocab_path) {
        print_error("--context/--candidate require --vocab");
        return EXIT_BAD_ARGS;
    }
    if (opts.vocab_path) {
        int rc = vocab_load(opts.vocab_path, &vocab);
        if (rc != EXIT_OK) {
            vocab_free(&vocab);
            return rc;
        }
    }

    u32_list_t context = {0};
    if (opts.ctx_ids) {
        int rc = parse_ctx_ids(opts.ctx_ids, &context);
        if (rc != EXIT_OK) {
            vocab_free(&vocab);
            return rc;
        }
    } else {
        int rc = tokenize_with_vocab(&vocab, opts.context_text, &context, true);
        if (rc != EXIT_OK) {
            vocab_free(&vocab);
            return rc;
        }
    }

    uint32_t candidate_id = 0;
    if (opts.candidate_id) {
        if (parse_uint32(opts.candidate_id, &candidate_id) != 0) {
            print_error("invalid candidate id '%s'", opts.candidate_id);
            vocab_free(&vocab);
            u32_list_free(&context);
            return EXIT_BAD_ARGS;
        }
    } else if (opts.candidate_token) {
        if (vocab_lookup_id(&vocab, opts.candidate_token, &candidate_id) != 0) {
            print_error("unknown candidate token '%s'", opts.candidate_token);
            vocab_free(&vocab);
            u32_list_free(&context);
            return EXIT_BAD_ARGS;
        }
    }

    psam_model_t* model = psam_load(opts.model_path);
    if (!model) {
        print_error("failed to load model '%s'", opts.model_path);
        vocab_free(&vocab);
        u32_list_free(&context);
        return EXIT_FILE_MISSING;
    }

    psam_explain_term_t* terms = NULL;
    if (opts.topN > 0) {
        terms = calloc(opts.topN, sizeof(psam_explain_term_t));
        if (!terms) {
            psam_destroy(model);
            vocab_free(&vocab);
            u32_list_free(&context);
            return EXIT_INTERNAL;
        }
    }

    psam_explain_result_t result;
    memset(&result, 0, sizeof(result));

    psam_error_t err = psam_explain(model, context.data, context.size,
                                    candidate_id, terms, opts.topN, &result);
    if (err != PSAM_OK) {
        print_error("psam_explain failed (%d)", err);
        free(terms);
        psam_destroy(model);
        vocab_free(&vocab);
        u32_list_free(&context);
        return EXIT_INTERNAL;
    }

    const char* candidate_str = vocab.size ? vocab_lookup_token(&vocab, candidate_id) : NULL;
    if (opts.pretty) {
        printf("{\n  \"candidate\": {\"id\":%u", candidate_id);
        if (candidate_str) printf(", \"token\":\"%s\"", candidate_str);
        printf("},\n  \"total\": %.6f,\n  \"bias\": %.6f,\n  \"terms\": [\n", result.total_score, result.bias_score);
    } else {
        printf("{\"candidate\":{\"id\":%u", candidate_id);
        if (candidate_str) printf(",\"token\":\"%s\"", candidate_str);
        printf("},\"total\":%.6f,\"bias\":%.6f,\"terms\":[", result.total_score, result.bias_score);
    }

    int term_count = result.term_count;
    if (opts.topN > 0 && term_count > (int)opts.topN) {
        term_count = (int)opts.topN;
    }

    for (int i = 0; i < term_count; ++i) {
        const char* source_str = vocab.size ? vocab_lookup_token(&vocab, terms[i].source_token) : NULL;
        if (opts.pretty) {
            printf("    {\"source\":%u", terms[i].source_token);
            if (source_str) printf(", \"token\":\"%s\"", source_str);
            printf(", \"offset\":%d, \"weight\":%.6f, \"idf\":%.6f, \"decay\":%.6f, \"contribution\":%.6f}",
                   terms[i].rel_offset, terms[i].weight_ppmi, terms[i].idf, terms[i].decay, terms[i].contribution);
            if (i + 1 < term_count) printf(",\n"); else printf("\n");
        } else {
            printf("{\"source\":%u", terms[i].source_token);
            if (source_str) printf(",\"token\":\"%s\"", source_str);
            printf(",\"offset\":%d,\"weight\":%.6f,\"idf\":%.6f,\"decay\":%.6f,\"contribution\":%.6f}",
                   terms[i].rel_offset, terms[i].weight_ppmi, terms[i].idf, terms[i].decay, terms[i].contribution);
            if (i + 1 < term_count) printf(",");
        }
    }

    if (opts.pretty) {
        printf("  ]\n}\n");
    } else {
        printf("]}\n");
    }

    free(terms);
    psam_destroy(model);
    vocab_free(&vocab);
    u32_list_free(&context);
    return EXIT_OK;
}

/* ==== analyze command ==== */

static void analyze_usage(void) {
    printf("Usage: psam analyze --model model.psam [--pretty]\n");
}

static int analyze_command(int argc, char** argv) {
    const char* model_path = NULL;
    bool pretty = false;

    for (int i = 2; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--model") == 0) {
            if (++i >= argc) { analyze_usage(); return EXIT_BAD_ARGS; }
            model_path = argv[i];
        } else if (strcmp(arg, "--pretty") == 0) {
            pretty = true;
        } else if (strcmp(arg, "--help") == 0) {
            analyze_usage();
            return EXIT_OK;
        } else {
            print_error("unknown option '%s'", arg);
            analyze_usage();
            return EXIT_BAD_ARGS;
        }
    }

    if (!model_path) {
        print_error("analyze requires --model");
        analyze_usage();
        return EXIT_BAD_ARGS;
    }

    /* Composite models not supported for analyze (would need to aggregate stats) */
    if (has_suffix(model_path, ".psamc")) {
        print_error("analyze does not support composite models (use inspect instead)");
        return EXIT_BAD_ARGS;
    }

    psam_model_t* model = psam_load(model_path);
    if (!model) {
        print_error("failed to load model '%s'", model_path);
        return EXIT_FILE_MISSING;
    }

    psam_stats_t stats;
    if (psam_get_stats(model, &stats) != PSAM_OK) {
        print_error("psam_get_stats failed");
        psam_destroy(model);
        return EXIT_INTERNAL;
    }

    double avg_degree = stats.row_count ? (double)stats.edge_count / (double)stats.row_count : 0.0;
    double density = (stats.row_count && stats.vocab_size)
        ? (double)stats.edge_count / ((double)stats.row_count * (double)stats.vocab_size)
        : 0.0;

    psam_provenance_t prov;
    memset(&prov, 0, sizeof(prov));
    psam_get_provenance(model, &prov);
    char hash_hex[PSAM_SOURCE_HASH_SIZE * 2 + 1];
    print_hex(prov.source_hash, PSAM_SOURCE_HASH_SIZE, hash_hex, sizeof(hash_hex));

    if (pretty) {
        printf("{\n");
        printf("  \"vocab_size\": %u,\n", stats.vocab_size);
        printf("  \"row_count\": %u,\n", stats.row_count);
        printf("  \"edge_count\": %" PRIu64 ",\n", stats.edge_count);
        printf("  \"average_degree\": %.6f,\n", avg_degree);
        printf("  \"density\": %.6f,\n", density);
        printf("  \"total_tokens\": %" PRIu64 ",\n", stats.total_tokens);
        printf("  \"memory_bytes\": %zu,\n", stats.memory_bytes);
        printf("  \"provenance\": {\n");
        printf("    \"created_timestamp\": %" PRIu64 ",\n", prov.created_timestamp);
        printf("    \"created_by\": \"%s\",\n", prov.created_by);
        printf("    \"source_hash\": \"%s\"\n", hash_hex);
        printf("  }\n");
        printf("}\n");
    } else {
        printf("{\"vocab_size\":%u,\"row_count\":%u,\"edge_count\":%" PRIu64
               ",\"average_degree\":%.6f,\"density\":%.6f,\"total_tokens\":%" PRIu64
               ",\"memory_bytes\":%zu,\"provenance\":{\"created_timestamp\":%" PRIu64
               ",\"created_by\":\"%s\",\"source_hash\":\"%s\"}}\n",
               stats.vocab_size, stats.row_count, stats.edge_count,
               avg_degree, density, stats.total_tokens, stats.memory_bytes,
               prov.created_timestamp, prov.created_by, hash_hex);
    }

    psam_destroy(model);
    return EXIT_OK;
}

/* ==== tokenize & ids ==== */

static void tokenize_usage(void) {
    printf("Usage: psam tokenize --vocab vocab.tsv --context \"text\"\n");
}

static int tokenize_command(int argc, char** argv) {
    const char* vocab_path = NULL;
    const char* text = NULL;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--vocab") == 0) {
            if (++i >= argc) { tokenize_usage(); return EXIT_BAD_ARGS; }
            vocab_path = argv[i];
        } else if (strcmp(argv[i], "--context") == 0) {
            if (++i >= argc) { tokenize_usage(); return EXIT_BAD_ARGS; }
            text = argv[i];
        } else if (strcmp(argv[i], "--help") == 0) {
            tokenize_usage();
            return EXIT_OK;
        } else {
            print_error("unknown option '%s'", argv[i]);
            tokenize_usage();
            return EXIT_BAD_ARGS;
        }
    }
    if (!vocab_path || !text) {
        tokenize_usage();
        return EXIT_BAD_ARGS;
    }
    vocab_t vocab;
    int rc = vocab_load(vocab_path, &vocab);
    if (rc != EXIT_OK) {
        vocab_free(&vocab);
        return rc;
    }
    u32_list_t ids = {0};
    rc = tokenize_with_vocab(&vocab, text, &ids, true);
    if (rc != EXIT_OK) {
        vocab_free(&vocab);
        u32_list_free(&ids);
        return rc;
    }
    printf("{\"ids\":[");
    for (size_t i = 0; i < ids.size; ++i) {
        printf("%u", ids.data[i]);
        if (i + 1 < ids.size) printf(",");
    }
    printf("]}\n");
    vocab_free(&vocab);
    u32_list_free(&ids);
    return EXIT_OK;
}

static void ids_usage(void) {
    printf("Usage: psam ids --vocab vocab.tsv --ids 1,2,3\n");
}

static int ids_command(int argc, char** argv) {
    const char* vocab_path = NULL;
    const char* ids = NULL;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--vocab") == 0) {
            if (++i >= argc) { ids_usage(); return EXIT_BAD_ARGS; }
            vocab_path = argv[i];
        } else if (strcmp(argv[i], "--ids") == 0) {
            if (++i >= argc) { ids_usage(); return EXIT_BAD_ARGS; }
            ids = argv[i];
        } else if (strcmp(argv[i], "--help") == 0) {
            ids_usage();
            return EXIT_OK;
        } else {
            print_error("unknown option '%s'", argv[i]);
            ids_usage();
            return EXIT_BAD_ARGS;
        }
    }
    if (!vocab_path || !ids) {
        ids_usage();
        return EXIT_BAD_ARGS;
    }
    vocab_t vocab;
    int rc = vocab_load(vocab_path, &vocab);
    if (rc != EXIT_OK) {
        vocab_free(&vocab);
        return rc;
    }
    u32_list_t list = {0};
    rc = parse_ctx_ids(ids, &list);
    if (rc != EXIT_OK) {
        vocab_free(&vocab);
        u32_list_free(&list);
        return rc;
    }
    if (list.size == 0) {
        vocab_free(&vocab);
        u32_list_free(&list);
        return EXIT_BAD_ARGS;
    }
    printf("{\"tokens\":[");
    for (size_t i = 0; i < list.size; ++i) {
        const char* tok = vocab_lookup_token(&vocab, list.data[i]);
        if (!tok) tok = "<UNK>";
        printf("\"%s\"", tok);
        if (i + 1 < list.size) printf(",");
    }
    printf("]}\n");
    vocab_free(&vocab);
    u32_list_free(&list);
    return EXIT_OK;
}

/* ==== compose command ==== */

typedef struct {
    char* path;         /* Model path */
    char* version;      /* Version (legacy, unused in v1) */
    char* vocab_path;   /* Vocabulary TSV path (v1 aligned) */
    char* id;           /* Layer ID (v1 aligned) */
    float weight;       /* Layer weight */
    float bias;         /* Layer bias */
} layer_spec_t;

typedef struct {
    layer_spec_t* data;
    size_t size;
    size_t capacity;
} layer_list_t;

static void layer_list_free(layer_list_t* list) {
    if (!list) return;
    for (size_t i = 0; i < list->size; ++i) {
        free(list->data[i].path);
        free(list->data[i].version);
        free(list->data[i].vocab_path);
        free(list->data[i].id);
    }
    free(list->data);
    list->data = NULL;
    list->size = list->capacity = 0;
}

static int layer_list_append(layer_list_t* list, const char* path, const char* version) PSAM_UNUSED;
static int layer_list_append(layer_list_t* list, const char* path, const char* version) {
    if (list->size == list->capacity) {
        size_t new_cap = list->capacity ? list->capacity * 2 : 4;
        layer_spec_t* new_data = realloc(list->data, new_cap * sizeof(layer_spec_t));
        if (!new_data) return -1;
        list->data = new_data;
        list->capacity = new_cap;
    }
    list->data[list->size].path = strdup(path);
    list->data[list->size].version = version ? strdup(version) : NULL;
    list->data[list->size].vocab_path = NULL;
    list->data[list->size].id = NULL;
    list->data[list->size].weight = 1.0f;
    list->data[list->size].bias = 0.0f;
    if (!list->data[list->size].path || (version && !list->data[list->size].version)) {
        free(list->data[list->size].path);
        free(list->data[list->size].version);
        return -1;
    }
    list->size++;
    return 0;
}

/* Add a layer with full v1 parameters */
static int layer_list_add_v1(layer_list_t* list, const char* path, const char* vocab_path,
                               const char* id, float weight, float bias) {
    if (list->size == list->capacity) {
        size_t new_cap = list->capacity ? list->capacity * 2 : 4;
        layer_spec_t* new_data = realloc(list->data, new_cap * sizeof(layer_spec_t));
        if (!new_data) return -1;
        list->data = new_data;
        list->capacity = new_cap;
    }
    list->data[list->size].path = strdup(path);
    list->data[list->size].version = NULL;
    list->data[list->size].vocab_path = vocab_path ? strdup(vocab_path) : NULL;
    list->data[list->size].id = id ? strdup(id) : NULL;
    list->data[list->size].weight = weight;
    list->data[list->size].bias = bias;

    if (!list->data[list->size].path) {
        return -1;
    }
    list->size++;
    return 0;
}

static void compose_usage(void) {
    printf("Usage: psam compose --out file.psamc [options]\n\n");
    printf("V1 Aligned Composite (recommended):\n");
    printf("  --layer <model.psam>    Add layer (repeatable)\n");
    printf("  --vocab <vocab.tsv>     Vocabulary for preceding layer\n");
    printf("  --weight <float>        Weight for preceding layer (default: 1.0)\n");
    printf("  --bias <float>          Bias for preceding layer (default: 0.0)\n");
    printf("  --unified-vocab <tsv>   Unified vocabulary (auto-generated if omitted)\n");
    printf("  --unknown-policy <p>    Policy: unk | skip (default: unk)\n");
    printf("  --coverage-weight <w>   Coverage: none | linear | sqrt (default: none)\n\n");
    printf("Legacy (same-vocab composites):\n");
    printf("  --layer <path>          Layer model\n");
    printf("  --created-by <text>     Creator string\n\n");
    printf("Sampler defaults (optional):\n");
    printf("  --sampler.save          Save sampler defaults to composite\n");
    printf("  --logit-transform <t>   Transform: raw | zscore | calibrated\n");
    printf("  --temperature <float>   Sampling temperature\n");
    printf("  --top-k <int>           Top-k sampling\n");
    printf("  --top-p <float>         Nucleus sampling threshold\n");
    printf("  --seed <int>            Random seed\n\n");
    printf("Other:\n");
    printf("  --force                 Overwrite existing files\n");
    printf("  --help                  Show this help\n");
}

static char* derive_layer_id(const char* path, size_t index) {
    if (!path) {
        char fallback[32];
        snprintf(fallback, sizeof(fallback), "layer-%zu", index);
        return strdup(fallback);
    }

    const char* last_slash = strrchr(path, '/');
#ifdef _WIN32
    const char* last_backslash = strrchr(path, '\\');
    if (!last_slash || (last_backslash && last_backslash > last_slash)) {
        last_slash = last_backslash;
    }
#endif
    const char* name = last_slash ? last_slash + 1 : path;
    if (!name || name[0] == '\0') {
        char fallback[32];
        snprintf(fallback, sizeof(fallback), "layer-%zu", index);
        return strdup(fallback);
    }

    /* Remove .psam extension if present */
    char* id = strdup(name);
    char* dot = strrchr(id, '.');
    if (dot && strcmp(dot, ".psam") == 0) {
        *dot = '\0';
    }
    return id;
}

/* Helper: Ensure directory exists, create if needed */
static int ensure_dir(const char* path) PSAM_UNUSED;
static int ensure_dir(const char* path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        return S_ISDIR(st.st_mode) ? 0 : -1;
    }
#ifdef _WIN32
    return mkdir(path);
#else
    return mkdir(path, 0755);
#endif
}

static int compose_command(int argc, char** argv) {
    const char* out_path = NULL;
    const char* created_by = NULL;
    const char* unified_vocab_path = NULL;
    const char* unknown_policy_str = "unk";
    const char* coverage_weight_str = "none";
    bool force PSAM_UNUSED = false;
    bool v1_mode = false;  /* Detect v1 if --vocab is used */

    /* V1 aligned composite state */
    layer_list_t layers = {0};
    char* pending_layer_path = NULL;
    char* pending_vocab_path = NULL;
    float pending_weight = 1.0f;
    float pending_bias = 0.0f;

    /* Sampler defaults */
    psamc_sampler_defaults_t sampler = {
        .logit_transform = PSAM_LOGIT_ZSCORE,
        .temperature = 1.0f,
        .top_k = 50,
        .top_p = 0.95f,
        .seed = 42
    };
    bool sampler_save = false;

    /* Legacy state */
    psamc_hyperparams_t hyper = PSAMC_PRESET_BALANCED_CONFIG;

    for (int i = 2; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--out") == 0) {
            if (++i >= argc) { compose_usage(); goto cleanup_error; }
            out_path = argv[i];
        } else if (strcmp(arg, "--layer") == 0) {
            /* Flush pending layer if exists */
            if (pending_layer_path) {
                if (!pending_vocab_path && v1_mode) {
                    print_error("--layer %s missing required --vocab", pending_layer_path);
                    goto cleanup_error;
                }
                char* layer_id = derive_layer_id(pending_layer_path, layers.size);
                if (layer_list_add_v1(&layers, pending_layer_path, pending_vocab_path, layer_id,
                                       pending_weight, pending_bias) != 0) {
                    free(layer_id);
                    goto cleanup_error;
                }
                free(layer_id);
                free(pending_layer_path);
                free(pending_vocab_path);
                pending_layer_path = pending_vocab_path = NULL;
                pending_weight = 1.0f;
                pending_bias = 0.0f;
            }
            if (++i >= argc) { compose_usage(); goto cleanup_error; }
            pending_layer_path = strdup(argv[i]);
        } else if (strcmp(arg, "--vocab") == 0) {
            if (++i >= argc) { compose_usage(); goto cleanup_error; }
            if (!pending_layer_path) {
                print_error("--vocab must follow --layer");
                goto cleanup_error;
            }
            pending_vocab_path = strdup(argv[i]);
            v1_mode = true;
        } else if (strcmp(arg, "--weight") == 0) {
            if (++i >= argc || parse_float(argv[i], &pending_weight) != 0) {
                goto cleanup_error;
            }
        } else if (strcmp(arg, "--bias") == 0) {
            if (++i >= argc || parse_float(argv[i], &pending_bias) != 0) {
                goto cleanup_error;
            }
        } else if (strcmp(arg, "--unified-vocab") == 0) {
            if (++i >= argc) { compose_usage(); goto cleanup_error; }
            unified_vocab_path = argv[i];
        } else if (strcmp(arg, "--unknown-policy") == 0) {
            if (++i >= argc) { compose_usage(); goto cleanup_error; }
            unknown_policy_str = argv[i];
        } else if (strcmp(arg, "--coverage-weight") == 0) {
            if (++i >= argc) { compose_usage(); goto cleanup_error; }
            coverage_weight_str = argv[i];
        } else if (strcmp(arg, "--force") == 0) {
            force = true;
        } else if (strcmp(arg, "--sampler.save") == 0) {
            sampler_save = true;
        } else if (strcmp(arg, "--temperature") == 0) {
            if (++i >= argc || parse_float(argv[i], &sampler.temperature) != 0) {
                goto cleanup_error;
            }
        } else if (strcmp(arg, "--top-k") == 0) {
            if (++i >= argc) { goto cleanup_error; }
            sampler.top_k = atoi(argv[i]);
        } else if (strcmp(arg, "--top-p") == 0) {
            if (++i >= argc || parse_float(argv[i], &sampler.top_p) != 0) {
                goto cleanup_error;
            }
        } else if (strcmp(arg, "--seed") == 0) {
            if (++i >= argc) { goto cleanup_error; }
            sampler.seed = (uint64_t)atoll(argv[i]);
        } else if (strcmp(arg, "--help") == 0) {
            compose_usage();
            goto cleanup_ok;
        } else {
            print_error("unknown option '%s'", arg);
            compose_usage();
            goto cleanup_error;
        }
    }

    /* Flush final pending layer */
    if (pending_layer_path) {
        if (!pending_vocab_path && v1_mode) {
            print_error("--layer %s missing required --vocab", pending_layer_path);
            goto cleanup_error;
        }
        char* layer_id = derive_layer_id(pending_layer_path, layers.size);
        if (layer_list_add_v1(&layers, pending_layer_path, pending_vocab_path, layer_id,
                               pending_weight, pending_bias) != 0) {
            free(layer_id);
            goto cleanup_error;
        }
        free(layer_id);
        free(pending_layer_path);
        free(pending_vocab_path);
        pending_layer_path = pending_vocab_path = NULL;
    }

    if (!out_path || layers.size == 0) {
        print_error("compose requires --out and at least one --layer");
        compose_usage();
        goto cleanup_error;
    }

    if (v1_mode) {
        /* V1 Aligned Composite Path */
        printf("Creating v1 aligned composite with %zu layers...\n", layers.size);

        /* 1. Determine unified vocab path */
        char unified_tsv_buf[512];
        const char* unified_tsv = unified_vocab_path;
        if (!unified_tsv) {
            /* Auto-generate unified vocab path */
            ensure_dir("vocabs");
            snprintf(unified_tsv_buf, sizeof(unified_tsv_buf), "vocabs/unified.tsv");
            unified_tsv = unified_tsv_buf;

            /* TODO: Build unified vocab from layer vocabs if not exists */
            /* For now, require user to provide --unified-vocab */
            print_error("Auto-building unified vocab not yet implemented");
            fprintf(stderr, "Please provide --unified-vocab for now\n");
            goto cleanup_error;
        }

        /* 2. Build vocabulary alignment */
        printf("Building vocabulary alignment from %zu layer vocabularies...\n", layers.size);

        const char** vocab_paths = malloc(layers.size * sizeof(char*));
        if (!vocab_paths) goto cleanup_error;
        for (size_t i = 0; i < layers.size; ++i) {
            vocab_paths[i] = layers.data[i].vocab_path;
        }

        uint32_t unified_size = 0;
        psam_vocab_alignment_t* alignment = psam_build_vocab_alignment_from_files(
            vocab_paths, layers.size, NULL, &unified_size);
        free(vocab_paths);

        if (!alignment) {
            print_error("Failed to build vocabulary alignment");
            goto cleanup_error;
        }

        printf("  Unified vocabulary: %u tokens\n", unified_size);

        /* 3. Prepare arrays for psam_composite_save_v1 */
        ensure_dir("maps");

        const char** layer_ids = malloc(layers.size * sizeof(char*));
        const char** layer_paths = malloc(layers.size * sizeof(char*));
        float* weights = malloc(layers.size * sizeof(float));
        float* biases = malloc(layers.size * sizeof(float));
        uint32_t* local_vocab_sizes = malloc(layers.size * sizeof(uint32_t));
        const uint32_t** l2u_maps = malloc(layers.size * sizeof(uint32_t*));
        const uint32_t** u2l_pairs = malloc(layers.size * sizeof(uint32_t*));
        uint32_t* u2l_pair_counts = malloc(layers.size * sizeof(uint32_t));
        char** l2u_paths = malloc(layers.size * sizeof(char*));
        char** u2l_paths = malloc(layers.size * sizeof(char*));

        if (!layer_ids || !layer_paths || !weights || !biases || !local_vocab_sizes ||
            !l2u_maps || !u2l_pairs || !u2l_pair_counts || !l2u_paths || !u2l_paths) {
            psam_vocab_alignment_destroy(alignment);
            free(layer_ids); free(layer_paths); free(weights); free(biases);
            free(local_vocab_sizes); free(l2u_maps); free(u2l_pairs);
            free(u2l_pair_counts); free(l2u_paths); free(u2l_paths);
            goto cleanup_error;
        }

        /* Extract maps and prepare paths */
        for (size_t i = 0; i < layers.size; ++i) {
            layer_ids[i] = layers.data[i].id;
            layer_paths[i] = layers.data[i].path;
            weights[i] = layers.data[i].weight;
            biases[i] = layers.data[i].bias;

            /* Get alignment data for this layer */
            psam_vocab_remap_t* remap = &alignment->layer_remaps[i];

            local_vocab_sizes[i] = remap->local_vocab_size;
            l2u_maps[i] = remap->local_to_unified;

            /* Convert sparse entries to pairs for binary format */
            u2l_pair_counts[i] = remap->unified_to_local_count;
            uint32_t* pairs = malloc(remap->unified_to_local_count * 2 * sizeof(uint32_t));
            if (!pairs) {
                psam_vocab_alignment_destroy(alignment);
                for (size_t j = 0; j < i; ++j) {
                    free(l2u_paths[j]);
                    free(u2l_paths[j]);
                    free((void*)u2l_pairs[j]);
                }
                free(layer_ids); free(layer_paths); free(weights); free(biases);
                free(local_vocab_sizes); free(l2u_maps); free(u2l_pairs);
                free(u2l_pair_counts); free(l2u_paths); free(u2l_paths);
                goto cleanup_error;
            }
            for (uint32_t j = 0; j < remap->unified_to_local_count; ++j) {
                pairs[j * 2 + 0] = remap->unified_to_local_sparse[j].unified_id;
                pairs[j * 2 + 1] = remap->unified_to_local_sparse[j].local_id;
            }
            u2l_pairs[i] = pairs;

            /* Generate map file paths */
            l2u_paths[i] = malloc(512);
            u2l_paths[i] = malloc(512);
            snprintf(l2u_paths[i], 512, "maps/%s.l2u.u32", layer_ids[i]);
            snprintf(u2l_paths[i], 512, "maps/%s.u2l.pairs", layer_ids[i]);

            printf("  Layer %zu (%s): local_vocab=%u, coverage=%.1f%%\n",
                   i, layer_ids[i], local_vocab_sizes[i],
                   psam_vocab_alignment_get_coverage(alignment, i) * 100.0f);
        }

        /* 4. Parse policy and coverage enums */
        psam_unknown_policy_t policy = PSAM_UNKNOWN_MAP_UNK;
        if (strcmp(unknown_policy_str, "skip") == 0) {
            policy = PSAM_UNKNOWN_SKIP;
        }
        alignment->unknown_policy = policy;

        psam_coverage_rule_t coverage = PSAM_COVER_NONE;
        if (strcmp(coverage_weight_str, "linear") == 0) {
            coverage = PSAM_COVER_LINEAR;
        } else if (strcmp(coverage_weight_str, "sqrt") == 0) {
            coverage = PSAM_COVER_SQRT;
        }

        /* 5. Save v1 composite */
        printf("Saving aligned composite to %s...\n", out_path);

        int rc = psam_composite_save_v1(
            out_path,
            created_by ? created_by : "libpsam-cli",
            unified_tsv,
            policy,
            coverage,
            sampler_save ? &sampler : NULL,
            layers.size,
            layer_ids,
            layer_paths,
            weights,
            biases,
            local_vocab_sizes,
            l2u_maps,
            u2l_pairs,
            u2l_pair_counts,
            (const char**)l2u_paths,
            (const char**)u2l_paths,
            unified_size
        );

        /* Cleanup */
        for (size_t i = 0; i < layers.size; ++i) {
            free(l2u_paths[i]);
            free(u2l_paths[i]);
            free((void*)u2l_pairs[i]);  /* Free converted pairs */
        }
        psam_vocab_alignment_destroy(alignment);
        free(layer_ids); free(layer_paths); free(weights); free(biases);
        free(local_vocab_sizes); free(l2u_maps); free(u2l_pairs);
        free(u2l_pair_counts); free(l2u_paths); free(u2l_paths);

        if (rc != 0) {
            print_error("Failed to save aligned composite");
            goto cleanup_error;
        }

        /* 6. Print success summary */
        printf("\n✅ Saved aligned composite: %s\n", out_path);
        printf("   • Unified vocab:  %s (%u tokens)\n", unified_tsv, unified_size);
        printf("   • Unknown policy: %s\n", unknown_policy_str);
        printf("   • Coverage rule:  %s\n", coverage_weight_str);
        printf("   • Layers:         %zu\n", layers.size);
        for (size_t i = 0; i < layers.size; ++i) {
            printf("     - %s (weight=%.3f, bias=%.3f)\n",
                   layers.data[i].id, layers.data[i].weight, layers.data[i].bias);
            printf("       model: %s\n", layers.data[i].path);
            printf("       vocab: %s\n", layers.data[i].vocab_path);
        }
        printf("\nTry:\n  psam predict --model %s --prompt \"test\" --top_k 10\n", out_path);
        goto cleanup_ok;
    }

    /* Legacy path: same-vocabulary composites */
    const char* base_path = layers.data[0].path;
    size_t overlay_count = layers.size > 1 ? layers.size - 1 : 0;
    psam_composite_layer_file_t* overlay_descs = NULL;
    char** overlay_ids = NULL;

    if (overlay_count > 0) {
        overlay_descs = calloc(overlay_count, sizeof(psam_composite_layer_file_t));
        overlay_ids = calloc(overlay_count, sizeof(char*));
        if (!overlay_descs || !overlay_ids) {
            free(overlay_descs);
            free(overlay_ids);
            goto cleanup_error;
        }

        for (size_t i = 0; i < overlay_count; ++i) {
            overlay_descs[i].path = layers.data[i + 1].path;
            overlay_descs[i].weight = 1.0f;
            overlay_ids[i] = derive_layer_id(layers.data[i + 1].path, i);
            overlay_descs[i].id = overlay_ids[i];
        }
    }

    int rc = psam_composite_save_file(
        out_path,
        created_by ? created_by : NULL,
        &hyper,
        1.0f,
        base_path,
        overlay_count,
        overlay_descs
    );

    if (overlay_ids) {
        for (size_t i = 0; i < overlay_count; ++i) {
            free(overlay_ids[i]);
        }
        free(overlay_ids);
    }
    free(overlay_descs);

    if (rc != 0) {
        print_error("psam composite save failed");
        goto cleanup_error;
    }

    printf("{\"status\":\"ok\",\"composite\":\"%s\",\"layers\":%zu}\n", out_path, overlay_count + 1);
    goto cleanup_ok;

cleanup_error:
    free(pending_layer_path);
    free(pending_vocab_path);
    layer_list_free(&layers);
    return EXIT_BAD_ARGS;

cleanup_ok:
    free(pending_layer_path);
    free(pending_vocab_path);
    layer_list_free(&layers);
    return EXIT_OK;
}

/* ==== inspect command ==== */

static void inspect_usage(void) {
    printf("Usage: psam inspect --model path\n");
}

static int inspect_command(int argc, char** argv) {
    const char* path = NULL;
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0) {
            if (++i >= argc) { inspect_usage(); return EXIT_BAD_ARGS; }
            path = argv[i];
        } else if (strcmp(argv[i], "--help") == 0) {
            inspect_usage();
            return EXIT_OK;
        } else {
            print_error("unknown option '%s'", argv[i]);
            inspect_usage();
            return EXIT_BAD_ARGS;
        }
    }
    if (!path) {
        inspect_usage();
        return EXIT_BAD_ARGS;
    }

    if (has_suffix(path, ".psamc")) {
        psamc_composite_t* comp = psamc_load(path, false);
        if (!comp) {
            print_error("failed to load composite '%s'", path);
            return EXIT_FILE_MISSING;
        }
        printf("{\"type\":\"psamc\",\"layers\":%u,\"top_k\":%u,\"alpha\":%.6f}\n",
               comp->manifest.num_references,
               comp->hyperparams.top_k,
               comp->hyperparams.alpha);
        psamc_free(comp);
    } else {
        psam_model_t* model = psam_load(path);
        if (!model) {
            print_error("failed to load model '%s'", path);
            return EXIT_FILE_MISSING;
        }
        psam_stats_t stats;
        psam_get_stats(model, &stats);
        psam_provenance_t prov;
        psam_get_provenance(model, &prov);
        char hash_hex[PSAM_SOURCE_HASH_SIZE * 2 + 1];
        print_hex(prov.source_hash, PSAM_SOURCE_HASH_SIZE, hash_hex, sizeof(hash_hex));

        // Test psam_get_edges API
        psam_edge_t edges[500];
        int num_edges = psam_get_edges(model, UINT32_MAX, 0.0f, 500, edges);

        printf("{\"type\":\"psam\",\"vocab_size\":%u,\"row_count\":%u,\"edge_count\":%" PRIu64 ",\"created_by\":\"%s\",\"source_hash\":\"%s\",\"test_edges\":%d}\n",
               stats.vocab_size, stats.row_count, stats.edge_count,
               prov.created_by, hash_hex, num_edges);

        if (num_edges > 0) {
            fprintf(stderr, "Sample edges (first 5):\n");
            for (int i = 0; i < (num_edges < 5 ? num_edges : 5); i++) {
                fprintf(stderr, "  [%d] %u -> %u: weight=%.3f offset=%d obs=%u\n",
                       i, edges[i].source_token, edges[i].target_token,
                       edges[i].weight, edges[i].offset, edges[i].observations);
            }
        } else {
            fprintf(stderr, "WARNING: psam_get_edges returned %d (no edges extracted)\n", num_edges);
        }

        psam_destroy(model);
    }
    return EXIT_OK;
}

/* ==== main dispatch ==== */

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return EXIT_BAD_ARGS;
    }

    const char* cmd = argv[1];
    if (strcmp(cmd, "build") == 0) {
        return build_command(argc, argv);
    } else if (strcmp(cmd, "predict") == 0) {
        return predict_command(argc, argv);
    } else if (strcmp(cmd, "generate") == 0) {
        return generate_command(argc, argv);
    } else if (strcmp(cmd, "explain") == 0) {
        return explain_command(argc, argv);
    } else if (strcmp(cmd, "analyze") == 0) {
        return analyze_command(argc, argv);
    } else if (strcmp(cmd, "compose") == 0) {
        return compose_command(argc, argv);
    } else if (strcmp(cmd, "inspect") == 0) {
        return inspect_command(argc, argv);
    } else if (strcmp(cmd, "tokenize") == 0) {
        return tokenize_command(argc, argv);
    } else if (strcmp(cmd, "ids") == 0) {
        return ids_command(argc, argv);
    } else if (strcmp(cmd, "eval") == 0 || strcmp(cmd, "export") == 0) {
        print_error("command '%s' is not implemented yet", cmd);
        return EXIT_INTERNAL;
    } else if (strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
        print_usage();
        return EXIT_OK;
    }

    print_error("unknown command '%s'", cmd);
    print_usage();
    return EXIT_BAD_ARGS;
}
