/**
 * training_hash.h - Hash tables for training data structures
 *
 * Custom hash tables optimized for PSAM training, avoiding O(n) linear searches.
 */

#ifndef PSAM_TRAINING_HASH_H
#define PSAM_TRAINING_HASH_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ============================ Edge Hash Table ============================ */

typedef struct edge_entry_t {
    uint32_t target;        /* Key */
    float weight;
    uint32_t count;
    struct edge_entry_t* next;  /* For chaining */
} edge_entry_t;

typedef struct {
    edge_entry_t** buckets;
    uint32_t bucket_count;
    uint32_t entry_count;
} edge_hash_t;

static inline uint32_t hash_u32(uint32_t x) {
    /* MurmurHash3 finalizer */
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
}

static inline edge_hash_t* edge_hash_create(uint32_t initial_capacity) {
    if (initial_capacity < 16) initial_capacity = 16;

    edge_hash_t* ht = calloc(1, sizeof(edge_hash_t));
    if (!ht) return NULL;

    ht->bucket_count = initial_capacity;
    ht->entry_count = 0;
    ht->buckets = calloc(initial_capacity, sizeof(edge_entry_t*));

    if (!ht->buckets) {
        free(ht);
        return NULL;
    }

    return ht;
}

static inline void edge_hash_destroy(edge_hash_t* ht) {
    if (!ht) return;

    for (uint32_t i = 0; i < ht->bucket_count; i++) {
        edge_entry_t* entry = ht->buckets[i];
        while (entry) {
            edge_entry_t* next = entry->next;
            free(entry);
            entry = next;
        }
    }

    free(ht->buckets);
    free(ht);
}

static inline edge_entry_t* edge_hash_find_or_create(edge_hash_t* ht, uint32_t target) {
    uint32_t hash = hash_u32(target);
    uint32_t bucket = hash % ht->bucket_count;

    /* Search chain for existing entry */
    edge_entry_t* entry = ht->buckets[bucket];
    while (entry) {
        if (entry->target == target) {
            return entry;
        }
        entry = entry->next;
    }

    /* Create new entry */
    entry = calloc(1, sizeof(edge_entry_t));
    if (!entry) return NULL;

    entry->target = target;
    entry->weight = 0.0f;
    entry->count = 0;
    entry->next = ht->buckets[bucket];

    ht->buckets[bucket] = entry;
    ht->entry_count++;

    return entry;
}

/* Iterator for edges */
typedef struct {
    edge_hash_t* ht;
    uint32_t bucket_idx;
    edge_entry_t* current;
} edge_iterator_t;

static inline void edge_iterator_init(edge_iterator_t* it, edge_hash_t* ht) {
    it->ht = ht;
    it->bucket_idx = 0;
    it->current = NULL;

    /* Find first non-empty bucket */
    for (uint32_t i = 0; i < ht->bucket_count; i++) {
        if (ht->buckets[i]) {
            it->bucket_idx = i;
            it->current = ht->buckets[i];
            return;
        }
    }
}

static inline edge_entry_t* edge_iterator_next(edge_iterator_t* it) {
    if (!it->current) return NULL;

    edge_entry_t* result = it->current;
    it->current = it->current->next;

    /* If chain exhausted, find next non-empty bucket */
    if (!it->current) {
        for (uint32_t i = it->bucket_idx + 1; i < it->ht->bucket_count; i++) {
            if (it->ht->buckets[i]) {
                it->bucket_idx = i;
                it->current = it->ht->buckets[i];
                break;
            }
        }
    }

    return result;
}

/* ============================ Row Hash Table ============================ */

typedef struct row_accumulator_t row_accumulator_t;

typedef struct row_entry_t {
    uint64_t key;           /* Combined (source << 32 | offset) */
    row_accumulator_t* row;
    struct row_entry_t* next;
} row_entry_t;

typedef struct {
    row_entry_t** buckets;
    uint32_t bucket_count;
    uint32_t entry_count;
} row_hash_t;

static inline uint64_t make_row_key(uint32_t source, uint32_t offset) {
    return ((uint64_t)source << 32) | (uint64_t)offset;
}

static inline uint32_t hash_u64(uint64_t x) {
    /* MurmurHash3-inspired 64-bit hash */
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)x;
}

static inline row_hash_t* row_hash_create(uint32_t initial_capacity) {
    if (initial_capacity < 256) initial_capacity = 256;

    row_hash_t* ht = calloc(1, sizeof(row_hash_t));
    if (!ht) return NULL;

    ht->bucket_count = initial_capacity;
    ht->entry_count = 0;
    ht->buckets = calloc(initial_capacity, sizeof(row_entry_t*));

    if (!ht->buckets) {
        free(ht);
        return NULL;
    }

    return ht;
}

static inline void row_hash_destroy(row_hash_t* ht, void (*row_destructor)(row_accumulator_t*)) {
    if (!ht) return;

    for (uint32_t i = 0; i < ht->bucket_count; i++) {
        row_entry_t* entry = ht->buckets[i];
        while (entry) {
            row_entry_t* next = entry->next;
            if (row_destructor && entry->row) {
                row_destructor(entry->row);
            }
            free(entry);
            entry = next;
        }
    }

    free(ht->buckets);
    free(ht);
}

static inline row_accumulator_t* row_hash_find(row_hash_t* ht, uint32_t source, uint32_t offset) {
    uint64_t key = make_row_key(source, offset);
    uint32_t hash = hash_u64(key);
    uint32_t bucket = hash % ht->bucket_count;

    row_entry_t* entry = ht->buckets[bucket];
    while (entry) {
        if (entry->key == key) {
            return entry->row;
        }
        entry = entry->next;
    }

    return NULL;
}

static inline int row_hash_insert(row_hash_t* ht, uint32_t source, uint32_t offset, row_accumulator_t* row) {
    uint64_t key = make_row_key(source, offset);
    uint32_t hash = hash_u64(key);
    uint32_t bucket = hash % ht->bucket_count;

    /* Create new entry */
    row_entry_t* entry = calloc(1, sizeof(row_entry_t));
    if (!entry) return -1;

    entry->key = key;
    entry->row = row;
    entry->next = ht->buckets[bucket];

    ht->buckets[bucket] = entry;
    ht->entry_count++;

    return 0;
}

/* Iterator for rows */
typedef struct {
    row_hash_t* ht;
    uint32_t bucket_idx;
    row_entry_t* current;
} row_iterator_t;

static inline void row_iterator_init(row_iterator_t* it, row_hash_t* ht) {
    it->ht = ht;
    it->bucket_idx = 0;
    it->current = NULL;

    /* Find first non-empty bucket */
    for (uint32_t i = 0; i < ht->bucket_count; i++) {
        if (ht->buckets[i]) {
            it->bucket_idx = i;
            it->current = ht->buckets[i];
            return;
        }
    }
}

static inline row_accumulator_t* row_iterator_next(row_iterator_t* it) {
    if (!it->current) return NULL;

    row_accumulator_t* result = it->current->row;
    it->current = it->current->next;

    /* If chain exhausted, find next non-empty bucket */
    if (!it->current) {
        for (uint32_t i = it->bucket_idx + 1; i < it->ht->bucket_count; i++) {
            if (it->ht->buckets[i]) {
                it->bucket_idx = i;
                it->current = it->ht->buckets[i];
                break;
            }
        }
    }

    return result;
}

#endif /* PSAM_TRAINING_HASH_H */
