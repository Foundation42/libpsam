/**
 * psam_cli.c - Command line interface for libpsam
 *
 * Provides thin utilities for building, composing, predicting, and inspecting PSAM models.
 */

#define _POSIX_C_SOURCE 200809L

#include "psam.h"
#include "psam_composite.h"

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

    double* logits = malloc((size_t)candidate_count * sizeof(double));
    double* probs = malloc((size_t)candidate_count * sizeof(double));
    if (!logits || !probs) {
        free(logits);
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

    double sum = 0.0;
    for (int i = 0; i < candidate_count; ++i) {
        double val = exp(logits[i] - max_logit);
        probs[i] = val;
        sum += val;
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
            free(logits);
            free(probs);
            return 0;
        }
    }

    *out_token = preds[trimmed_count - 1].token;
    free(logits);
    free(probs);
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
    uint32_t window;
    uint32_t top_k;
    float alpha;
} build_options_t;

static void build_usage(void) {
    printf("Usage: psam build --input file.txt [--input other.txt] --out model.psam --vocab-out vocab.tsv [options]\n");
    printf("Options:\n");
    printf("  --window <n>        Context window (default 8)\n");
    printf("  --top_k <n>         Top-K predictions stored during inference (default 32)\n");
    printf("  --alpha <f>         Distance decay parameter (default 0.1)\n");
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

    if (opts.inputs.size == 0 || !opts.output_model || !opts.output_vocab) {
        print_error("build requires --input, --out, and --vocab-out");
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

    if (unique_count == 0) {
        free(sorted);
        string_list_free(&tokens);
        print_error("no unique tokens discovered");
        return EXIT_BAD_ARGS;
    }

    char** id_to_token = malloc(unique_count * sizeof(char*));
    if (!id_to_token) {
        free(sorted);
        string_list_free(&tokens);
        RETURN_BUILD(EXIT_INTERNAL);
    }
    for (size_t i = 0; i < unique_count; ++i) {
        id_to_token[i] = strdup(sorted[i]);
        if (!id_to_token[i]) {
            for (size_t j = 0; j < i; ++j) free(id_to_token[j]);
            free(id_to_token);
            free(sorted);
            string_list_free(&tokens);
            RETURN_BUILD(EXIT_INTERNAL);
        }
    }

    psam_config_t config = {
        .vocab_size = (uint32_t)unique_count,
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
        for (size_t i = 0; i < unique_count; ++i) free(id_to_token[i]);
        free(id_to_token);
        free(sorted);
        string_list_free(&tokens);
        RETURN_BUILD(EXIT_INTERNAL);
    }

    uint32_t* sequence = malloc(tokens.size * sizeof(uint32_t));
    if (!sequence) {
        psam_destroy(model);
        for (size_t j = 0; j < unique_count; ++j) free(id_to_token[j]);
        free(id_to_token);
        free(sorted);
        string_list_free(&tokens);
        RETURN_BUILD(EXIT_INTERNAL);
    }

    for (size_t i = 0; i < tokens.size; ++i) {
        char* tok = tokens.data[i];
        char** found = bsearch(&tok, sorted, unique_count, sizeof(char*), compare_str_ptr);
        if (!found) {
            print_error("internal error: token lookup failed for '%s'", tok);
            free(sequence);
            psam_destroy(model);
            for (size_t j = 0; j < unique_count; ++j) free(id_to_token[j]);
            free(id_to_token);
            free(sorted);
            string_list_free(&tokens);
            RETURN_BUILD(EXIT_INTERNAL);
        }
        uint32_t id = (uint32_t)(found - sorted);
        sequence[i] = id;
    }

    psam_error_t batch_err = psam_train_batch(model, sequence, tokens.size);
    free(sequence);
    if (batch_err != PSAM_OK) {
        print_error("psam_train_batch failed (%d)", batch_err);
        psam_destroy(model);
        for (size_t j = 0; j < unique_count; ++j) free(id_to_token[j]);
        free(id_to_token);
        free(sorted);
        string_list_free(&tokens);
        RETURN_BUILD(EXIT_INTERNAL);
    }

    psam_error_t finalize_err = psam_finalize_training(model);
    if (finalize_err != PSAM_OK) {
        print_error("psam_finalize_training failed (%d)", finalize_err);
        psam_destroy(model);
        for (size_t j = 0; j < unique_count; ++j) free(id_to_token[j]);
        free(id_to_token);
        free(sorted);
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
        for (size_t j = 0; j < unique_count; ++j) free(id_to_token[j]);
        free(id_to_token);
        free(sorted);
        string_list_free(&tokens);
        RETURN_BUILD(EXIT_INTERNAL);
    }

    psam_error_t save_err = psam_save(model, opts.output_model);
    if (save_err != PSAM_OK) {
        print_error("psam_save failed (%d)", save_err);
        psam_destroy(model);
        for (size_t j = 0; j < unique_count; ++j) free(id_to_token[j]);
        free(id_to_token);
        free(sorted);
        string_list_free(&tokens);
        RETURN_BUILD(EXIT_INTERNAL);
    }

    FILE* vocab_file = fopen(opts.output_vocab, "w");
    if (!vocab_file) {
        print_error("failed to write vocab '%s': %s", opts.output_vocab, strerror(errno));
        psam_destroy(model);
        for (size_t j = 0; j < unique_count; ++j) free(id_to_token[j]);
        free(id_to_token);
        free(sorted);
        string_list_free(&tokens);
        RETURN_BUILD(EXIT_INTERNAL);
    }
    for (size_t i = 0; i < unique_count; ++i) {
        fprintf(vocab_file, "%zu\t%s\n", i, id_to_token[i]);
    }
    fclose(vocab_file);

    printf("{\"status\":\"ok\",\"model\":\"%s\",\"vocab\":\"%s\",\"tokens\":%zu,\"vocab_size\":%zu}\n",
           opts.output_model, opts.output_vocab, tokens.size, unique_count);

    psam_destroy(model);
    for (size_t i = 0; i < unique_count; ++i) free(id_to_token[i]);
    free(id_to_token);
    free(sorted);
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
    bool pretty;
} predict_options_t;

static void predict_usage(void) {
    printf("Usage: psam predict --model model.psam (--ctx-ids 1,2,3 | --context \"text\" --vocab vocab.tsv) [--top_k 5] [--temperature 1.0] [--top_p 1.0] [--pretty]\n");
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
    uint32_t seed;
    bool seed_provided;
    bool pretty;
    bool quiet;
} generate_options_t;

static void generate_usage(void) {
    printf("Usage: psam generate --model model.psam (--ctx-ids 1,2,3 | --context \"text\" --vocab vocab.tsv) [--count 32] [--top_k 16] [--top_p 1.0] [--temperature 1.0] [--seed N] [--pretty] [--quiet]\n");
}

static int generate_command(int argc, char** argv) {
    generate_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.count = 32;
    opts.top_k = 16;
    opts.top_p = 1.0f;
    opts.temperature = 1.0f;

    for (int i = 2; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--model") == 0) {
            if (++i >= argc) { generate_usage(); return EXIT_BAD_ARGS; }
            opts.model_path = argv[i];
        } else if (strcmp(arg, "--ctx-ids") == 0) {
            if (++i >= argc) { generate_usage(); return EXIT_BAD_ARGS; }
            opts.ctx_ids = argv[i];
        } else if (strcmp(arg, "--context") == 0) {
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

    vocab_t vocab = {0};
    if (opts.context_text && !opts.vocab_path) {
        print_error("--context requires --vocab");
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

    psam_model_t* model = psam_load(opts.model_path);
    if (!model) {
        print_error("failed to load model '%s'", opts.model_path);
        vocab_free(&vocab);
        u32_list_free(&context);
        return EXIT_FILE_MISSING;
    }

    uint32_t predict_cap = opts.top_k ? opts.top_k : 64;
    if (predict_cap < 8) predict_cap = 8;
    psam_prediction_t* preds = calloc(predict_cap, sizeof(psam_prediction_t));
    if (!preds) {
        psam_destroy(model);
        vocab_free(&vocab);
        u32_list_free(&context);
        return EXIT_INTERNAL;
    }

    u32_list_t generated = {0};
    uint64_t rng_state = opts.seed_provided ? (uint64_t)opts.seed : ((uint64_t)time(NULL) ^ 0x9e3779b97f4a7c15ULL);
    if (rng_state == 0) rng_state = 1;
    bool first_output = true;

    for (uint32_t step = 0; step < opts.count; ++step) {
        int num_preds = psam_predict(model, context.data, context.size, preds, predict_cap);
        if (num_preds <= 0) {
            break;
        }

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

    free(preds);
    psam_destroy(model);

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

    u32_list_free(&generated);
    vocab_free(&vocab);
    u32_list_free(&context);
    return EXIT_OK;
}

static int predict_command(int argc, char** argv) {
    predict_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.top_k = 5;
    opts.temperature = 1.0f;
    opts.top_p = 1.0f;

    for (int i = 2; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--model") == 0) {
            if (++i >= argc) { predict_usage(); return EXIT_BAD_ARGS; }
            opts.model_path = argv[i];
        } else if (strcmp(arg, "--ctx-ids") == 0) {
            if (++i >= argc) { predict_usage(); return EXIT_BAD_ARGS; }
            opts.ctx_ids = argv[i];
        } else if (strcmp(arg, "--context") == 0) {
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

    vocab_t vocab = {0};
    if (opts.context_text && !opts.vocab_path) {
        print_error("--context requires --vocab");
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
    } else if (opts.context_text) {
        int rc = tokenize_with_vocab(&vocab, opts.context_text, &context, true);
        if (rc != EXIT_OK) {
            vocab_free(&vocab);
            return rc;
        }
    }

    psam_model_t* model = psam_load(opts.model_path);
    if (!model) {
        print_error("failed to load model '%s'", opts.model_path);
        vocab_free(&vocab);
        u32_list_free(&context);
        return EXIT_FILE_MISSING;
    }

    if (context.size == 0) {
        print_error("empty context");
        psam_destroy(model);
        vocab_free(&vocab);
        u32_list_free(&context);
        return EXIT_BAD_ARGS;
    }

    uint32_t predict_count = opts.top_k ? opts.top_k : 16;
    psam_prediction_t* preds = calloc(predict_count, sizeof(psam_prediction_t));
    if (!preds) {
        psam_destroy(model);
        vocab_free(&vocab);
        u32_list_free(&context);
        return EXIT_INTERNAL;
    }

    int num_preds = psam_predict(model, context.data, context.size, preds, predict_count);
    if (num_preds < 0) {
        print_error("psam_predict failed (%d)", num_preds);
        free(preds);
        psam_destroy(model);
        vocab_free(&vocab);
        u32_list_free(&context);
        return EXIT_INTERNAL;
    }

    if (opts.pretty) {
        printf("{\n  \"predictions\": [\n");
    } else {
        printf("{\"predictions\":[");
    }
    for (int i = 0; i < num_preds; ++i) {
        const char* token_str = vocab.size ? vocab_lookup_token(&vocab, preds[i].token) : NULL;
        float score = preds[i].score;
        if (opts.temperature != 1.0f) {
            score = preds[i].score / opts.temperature;
        }
        if (opts.pretty) {
            printf("    {\"id\":%u", preds[i].token);
            if (token_str) {
                printf(", \"token\":\"%s\"", token_str);
            }
            printf(", \"score\":%.6f}", score);
            if (i + 1 < num_preds) printf(",\n"); else printf("\n");
        } else {
            printf("{\"id\":%u", preds[i].token);
            if (token_str) {
                printf(",\"token\":\"%s\"", token_str);
            }
            printf(",\"score\":%.6f}", score);
            if (i + 1 < num_preds) printf(",");
        }
    }
    if (opts.pretty) {
        printf("  ]\n}\n");
    } else {
        printf("]}\n");
    }

    free(preds);
    psam_destroy(model);
    vocab_free(&vocab);
    u32_list_free(&context);
    return EXIT_OK;
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
    char* path;
    char* version;
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
    }
    free(list->data);
    list->data = NULL;
    list->size = list->capacity = 0;
}

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
    if (!list->data[list->size].path || (version && !list->data[list->size].version)) {
        free(list->data[list->size].path);
        free(list->data[list->size].version);
        return -1;
    }
    list->size++;
    return 0;
}

static void compose_usage(void) {
    printf("Usage: psam compose --out composite.psamc --layer base.psam [--layer overlay.psam] [--created-by text]\n");
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
    return strdup(name);
}

static int compose_command(int argc, char** argv) {
    const char* out_path = NULL;
    const char* created_by = NULL;
    layer_list_t layers = {0};
    psamc_hyperparams_t hyper = PSAMC_PRESET_BALANCED_CONFIG;

    for (int i = 2; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--out") == 0) {
            if (++i >= argc) { compose_usage(); layer_list_free(&layers); return EXIT_BAD_ARGS; }
            out_path = argv[i];
        } else if (strcmp(arg, "--layer") == 0) {
            if (++i >= argc) { compose_usage(); layer_list_free(&layers); return EXIT_BAD_ARGS; }
            const char* layer_path = argv[i];
            const char* version = NULL;
            if (i + 1 < argc && strncmp(argv[i + 1], "--", 2) != 0 && strchr(argv[i + 1], '.') != NULL) {
                version = argv[++i];
            }
            if (layer_list_append(&layers, layer_path, version) != 0) {
                layer_list_free(&layers);
                return EXIT_INTERNAL;
            }
        } else if (strcmp(arg, "--created-by") == 0) {
            if (++i >= argc) { compose_usage(); layer_list_free(&layers); return EXIT_BAD_ARGS; }
            created_by = argv[i];
        } else if (strcmp(arg, "--sampler.top_k") == 0) {
            if (++i >= argc || parse_uint32(argv[i], &hyper.top_k) != 0) {
                compose_usage();
                layer_list_free(&layers);
                return EXIT_BAD_ARGS;
            }
        } else if (strcmp(arg, "--sampler.temperature") == 0) {
            if (++i >= argc || parse_float(argv[i], &hyper.alpha) != 0) {
                compose_usage();
                layer_list_free(&layers);
                return EXIT_BAD_ARGS;
            }
        } else if (strcmp(arg, "--help") == 0) {
            compose_usage();
            layer_list_free(&layers);
            return EXIT_OK;
        } else {
            print_error("unknown option '%s'", arg);
            compose_usage();
            layer_list_free(&layers);
            return EXIT_BAD_ARGS;
        }
    }

    if (!out_path || layers.size == 0) {
        print_error("compose requires --out and at least one --layer");
        compose_usage();
        layer_list_free(&layers);
        return EXIT_BAD_ARGS;
    }

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
            layer_list_free(&layers);
            return EXIT_INTERNAL;
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
    layer_list_free(&layers);

    if (rc != 0) {
        print_error("psam composite save failed");
        return EXIT_INTERNAL;
    }

    printf("{\"status\":\"ok\",\"composite\":\"%s\",\"layers\":%zu}\n", out_path, overlay_count + 1);
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
        printf("{\"type\":\"psam\",\"vocab_size\":%u,\"row_count\":%u,\"edge_count\":%" PRIu64 ",\"created_by\":\"%s\",\"source_hash\":\"%s\"}\n",
               stats.vocab_size, stats.row_count, stats.edge_count,
               prov.created_by, hash_hex);
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
