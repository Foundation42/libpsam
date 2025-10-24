/**
 * sha256.c - Standalone SHA-256 implementation (no external dependencies)
 *
 * Based on FIPS 180-4 specification
 * Public domain implementation
 */

#include <stdint.h>
#include <string.h>
#include <stdio.h>

/* SHA-256 constants */
static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

/* Initial hash values */
static const uint32_t H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

/* SHA-256 context */
typedef struct {
    uint32_t state[8];
    uint64_t count;
    uint8_t buffer[64];
} sha256_ctx_t;

/* Bit rotation macros */
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define SHR(x, n) ((x) >> (n))

/* SHA-256 functions */
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define SIGMA0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define SIGMA1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define GAMMA0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3))
#define GAMMA1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10))

static uint32_t read_be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) |
           ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) |
           ((uint32_t)p[3]);
}

static void write_be32(uint8_t* p, uint32_t val) {
    p[0] = (val >> 24) & 0xff;
    p[1] = (val >> 16) & 0xff;
    p[2] = (val >> 8) & 0xff;
    p[3] = val & 0xff;
}

static void sha256_transform(sha256_ctx_t* ctx, const uint8_t* data) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t T1, T2;
    int i;

    /* Prepare message schedule */
    for (i = 0; i < 16; i++) {
        W[i] = read_be32(data + i * 4);
    }
    for (i = 16; i < 64; i++) {
        W[i] = GAMMA1(W[i - 2]) + W[i - 7] + GAMMA0(W[i - 15]) + W[i - 16];
    }

    /* Initialize working variables */
    a = ctx->state[0];
    b = ctx->state[1];
    c = ctx->state[2];
    d = ctx->state[3];
    e = ctx->state[4];
    f = ctx->state[5];
    g = ctx->state[6];
    h = ctx->state[7];

    /* Main loop */
    for (i = 0; i < 64; i++) {
        T1 = h + SIGMA1(e) + CH(e, f, g) + K[i] + W[i];
        T2 = SIGMA0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    /* Add to state */
    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

static void sha256_init(sha256_ctx_t* ctx) {
    memcpy(ctx->state, H0, sizeof(H0));
    ctx->count = 0;
}

static void sha256_update(sha256_ctx_t* ctx, const uint8_t* data, size_t len) {
    size_t i, index, part_len;

    index = (size_t)(ctx->count & 0x3f);
    ctx->count += len;

    part_len = 64 - index;

    if (len >= part_len) {
        memcpy(&ctx->buffer[index], data, part_len);
        sha256_transform(ctx, ctx->buffer);

        for (i = part_len; i + 63 < len; i += 64) {
            sha256_transform(ctx, &data[i]);
        }

        index = 0;
    } else {
        i = 0;
    }

    memcpy(&ctx->buffer[index], &data[i], len - i);
}

static void sha256_final(sha256_ctx_t* ctx, uint8_t* hash) {
    uint8_t bits[8];
    size_t index, pad_len;
    uint64_t bit_count;

    /* Save bit count */
    bit_count = ctx->count * 8;
    for (int i = 0; i < 8; i++) {
        bits[7 - i] = (bit_count >> (i * 8)) & 0xff;
    }

    /* Pad to 56 mod 64 */
    index = (size_t)(ctx->count & 0x3f);
    pad_len = (index < 56) ? (56 - index) : (120 - index);

    uint8_t padding[64];
    padding[0] = 0x80;
    memset(padding + 1, 0, pad_len - 1);

    sha256_update(ctx, padding, pad_len);
    sha256_update(ctx, bits, 8);

    /* Output hash */
    for (int i = 0; i < 8; i++) {
        write_be32(hash + i * 4, ctx->state[i]);
    }
}

/* Public API for file hashing */
int sha256_file(const char* path, uint8_t* out_hash) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        return -1;
    }

    sha256_ctx_t ctx;
    sha256_init(&ctx);

    uint8_t buffer[4096];
    size_t n;

    while ((n = fread(buffer, 1, sizeof(buffer), f)) > 0) {
        sha256_update(&ctx, buffer, n);
    }

    fclose(f);
    sha256_final(&ctx, out_hash);

    return 0;
}

/* Hash arbitrary data */
void sha256_data(const uint8_t* data, size_t len, uint8_t* out_hash) {
    sha256_ctx_t ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, out_hash);
}
