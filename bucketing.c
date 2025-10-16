#define _POSIX_C_SOURCE 200809L
#include "bucketing.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <time.h>
#include <inttypes.h>
#include <math.h>
#include <omp.h>

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

int bucketing_build(const uint32_t *input, uint32_t N,
                    uint32_t total_dpus, uint32_t cap,
                    uint32_t **out_bucketed,
                    uint32_t **out_offsets,
                    uint32_t **out_counts,
                    uint32_t *out_num_buckets)
{
    if (!input || !out_bucketed || !out_offsets || !out_counts || !out_num_buckets)
        return -1;
    if (total_dpus == 0u || cap == 0u)
        return -2;

    omp_set_num_threads(NUM_THREADS);

    const uint64_t capacity = (uint64_t) total_dpus * (uint64_t) cap;
    if ((uint64_t)N > capacity) {
        fprintf(stderr, "[bucketing_build] N=%" PRIu32 " exceeds capacity total_dpus*cap=%" PRIu64 "\n",
                (uint32_t)N, capacity);
        return -3;
    }

    // -------- Global min/max (parallel) --------
    uint32_t vmin = UINT32_MAX, vmax = 0;
    #pragma omp parallel
    {
        uint32_t lmin = UINT32_MAX, lmax = 0;
        #pragma omp for nowait schedule(static)
        for (uint32_t i = 0; i < N; i++) {
            uint32_t v = input[i];
            if (v < lmin) lmin = v;
            if (v > lmax) lmax = v;
        }
        #pragma omp critical
        {
            if (lmin < vmin) vmin = lmin;
            if (lmax > vmax) vmax = lmax;
        }
    }

    uint64_t range = (uint64_t)vmax - (uint64_t)vmin + 1ull;
    if (range == 0ull) range = 1ull;

    // -------- Iterate B until constraints hold --------
    uint32_t B = (total_dpus < 16384u ? 65536u : 4u * total_dpus);
    if (B < total_dpus) B = total_dpus;

    uint32_t *bin_counts = NULL;
    uint32_t *bin_offsets = NULL;

    uint32_t *bucketed = NULL;
    uint32_t *final_offsets = NULL;
    uint32_t *final_counts  = NULL;
    uint32_t  num_buckets   = 0;

    int ok = 0;
    for (int attempt = 0; attempt < 10 && !ok; attempt++) {
        // clear previous attempt
        free(bin_counts);  bin_counts  = NULL;
        free(bin_offsets); bin_offsets = NULL;
        free(bucketed);    bucketed    = NULL;
        free(final_offsets); final_offsets = NULL;
        free(final_counts);  final_counts  = NULL;
        num_buckets = 0;

        // --- histogram ---
        bin_counts  = (uint32_t *)calloc((size_t)B, sizeof(uint32_t));
        bin_offsets = (uint32_t *)malloc((size_t)B * sizeof(uint32_t));
        if (!bin_counts || !bin_offsets) {
            free(bin_counts); free(bin_offsets);
            return -4;
        }

        long double inv_range = (long double)B / (long double)range;

        // Parallel local hist + reduction
        int T = omp_get_max_threads();
        uint32_t **local_bins = (uint32_t **)malloc((size_t)T * sizeof(uint32_t *));
        if (!local_bins) { free(bin_counts); free(bin_offsets); return -4; }
        uint32_t *lb_store = (uint32_t *)calloc((size_t)T * (size_t)B, sizeof(uint32_t));
        if (!lb_store) { free(local_bins); free(bin_counts); free(bin_offsets); return -4; }
        for (int t = 0; t < T; t++) local_bins[t] = lb_store + (size_t)t * (size_t)B;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            uint32_t *lb = local_bins[tid];

            #pragma omp for schedule(static)
            for (uint32_t i = 0; i < N; i++) {
                uint64_t rel = (uint64_t)input[i] - (uint64_t)vmin;
                uint64_t b   = (uint64_t)(inv_range * (long double)rel);
                if (b >= (uint64_t)B) b = (uint64_t)B - 1ull;
                lb[(uint32_t)b] += 1u;
            }
        }

        #pragma omp parallel for schedule(static)
        for (uint32_t b = 0; b < B; b++) {
            uint64_t s = 0;
            for (int t = 0; t < T; t++) s += local_bins[t][b];
            bin_counts[b] = (uint32_t)s;
        }

        free(lb_store);
        free(local_bins);

        // cap constraint
        uint32_t max_bin = 0;
        #pragma omp parallel for reduction(max:max_bin) schedule(static)
        for (uint32_t b = 0; b < B; b++) {
            if (bin_counts[b] > max_bin) max_bin = bin_counts[b];
        }
        if (max_bin > cap) {
            B = (B < (1u<<20) ? (B<<1) : (B + B/2));
            continue;
        }

        // --- prefix
        uint32_t total = 0;
        for (uint32_t b = 0; b < B; b++) { bin_offsets[b] = total; total += bin_counts[b]; }
        if (total != N) { free(bin_counts); free(bin_offsets); return -6; }

        // --- pack (parallel, stable within bin)
        bucketed = (uint32_t *)malloc((size_t)N * sizeof(uint32_t));
        if (!bucketed) { free(bin_counts); free(bin_offsets); return -7; }

        // rebuild local hist to compute per-thread offsets
        T = omp_get_max_threads();
        local_bins = (uint32_t **)malloc((size_t)T * sizeof(uint32_t *));
        if (!local_bins) { free(bucketed); free(bin_counts); free(bin_offsets); return -8; }
        lb_store = (uint32_t *)calloc((size_t)T * (size_t)B, sizeof(uint32_t));
        if (!lb_store) { free(local_bins); free(bucketed); free(bin_counts); free(bin_offsets); return -8; }
        for (int t = 0; t < T; t++) local_bins[t] = lb_store + (size_t)t * (size_t)B;

        long double inv_range2 = inv_range;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            uint32_t *lb = local_bins[tid];

            #pragma omp for schedule(static)
            for (uint32_t i = 0; i < N; i++) {
                uint64_t rel = (uint64_t)input[i] - (uint64_t)vmin;
                uint64_t b   = (uint64_t)(inv_range2 * (long double)rel);
                if (b >= (uint64_t)B) b = (uint64_t)B - 1ull;
                lb[(uint32_t)b] += 1u;
            }
        }

        // per-thread starting offsets per bin
        uint32_t **thread_prefix = (uint32_t **)malloc((size_t)T * sizeof(uint32_t *));
        if (!thread_prefix) { free(lb_store); free(local_bins); free(bucketed); free(bin_counts); free(bin_offsets); return -8; }
        uint32_t *tp_store = (uint32_t *)malloc((size_t)T * (size_t)B * sizeof(uint32_t));
        if (!tp_store) { free(thread_prefix); free(lb_store); free(local_bins); free(bucketed); free(bin_counts); free(bin_offsets); return -8; }
        for (int t = 0; t < T; t++) thread_prefix[t] = tp_store + (size_t)t * (size_t)B;

        #pragma omp parallel for schedule(static)
        for (uint32_t b = 0; b < B; b++) {
            uint32_t run = bin_offsets[b];
            for (int t = 0; t < T; t++) {
                thread_prefix[t][b] = run;
                run += local_bins[t][b];
            }
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            uint32_t *cursor = (uint32_t *)malloc((size_t)B * sizeof(uint32_t));
            uint32_t *cur = cursor ? cursor : thread_prefix[tid];
            memcpy(cur, thread_prefix[tid], (size_t)B * sizeof(uint32_t));

            #pragma omp for schedule(static)
            for (uint32_t i = 0; i < N; i++) {
                uint64_t rel = (uint64_t)input[i] - (uint64_t)vmin;
                uint64_t b   = (uint64_t)(inv_range2 * (long double)rel);
                if (b >= (uint64_t)B) b = (uint64_t)B - 1ull;
                uint32_t pos = cur[(uint32_t)b]++;
                bucketed[pos] = input[i];
            }
            if (cursor) free(cursor);
        }

        free(tp_store);
        free(thread_prefix);
        free(lb_store);
        free(local_bins);

        // --- group bins into buckets (≤ cap, never split a bin)
        uint32_t cap_buckets = (B > total_dpus ? B : total_dpus) + 8u;
        final_offsets = (uint32_t *)malloc((size_t)cap_buckets * sizeof(uint32_t));
        final_counts  = (uint32_t *)malloc((size_t)cap_buckets * sizeof(uint32_t));
        if (!final_offsets || !final_counts) {
            free(final_offsets); free(final_counts);
            free(bucketed); free(bin_counts); free(bin_offsets);
            return -9;
        }

        num_buckets = 0;
        uint32_t acc = 0;
        uint32_t cur_start = (B > 0 ? bin_offsets[0] : 0);

        for (uint32_t b = 0; b < B; b++) {
            uint32_t cnt = bin_counts[b];
            if (acc > 0 && (uint64_t)acc + (uint64_t)cnt > (uint64_t)cap) {
                if (num_buckets >= cap_buckets) {
                    uint32_t new_cap = cap_buckets * 2u;
                    uint32_t *no = (uint32_t *)realloc(final_offsets, (size_t)new_cap * sizeof(uint32_t));
                    uint32_t *nc = (uint32_t *)realloc(final_counts,  (size_t)new_cap * sizeof(uint32_t));
                    if (!no || !nc) { free(no); free(nc); free(final_offsets); free(final_counts); free(bucketed); free(bin_counts); free(bin_offsets); return -10; }
                    final_offsets = no; final_counts = nc; cap_buckets = new_cap;
                }
                final_offsets[num_buckets] = cur_start;
                final_counts[num_buckets]  = acc;
                num_buckets++;
                cur_start = bin_offsets[b];
                acc = 0;
            }
            acc += cnt;
        }
        if (acc > 0 || (B == 0 && N == 0)) {
            if (num_buckets >= cap_buckets) {
                uint32_t new_cap = cap_buckets + 1u;
                uint32_t *no = (uint32_t *)realloc(final_offsets, (size_t)new_cap * sizeof(uint32_t));
                uint32_t *nc = (uint32_t *)realloc(final_counts,  (size_t)new_cap * sizeof(uint32_t));
                if (!no || !nc) { free(no); free(nc); free(final_offsets); free(final_counts); free(bucketed); free(bin_counts); free(bin_offsets); return -11; }
                final_offsets = no; final_counts = nc; cap_buckets = new_cap;
            }
            final_offsets[num_buckets] = cur_start;
            final_counts[num_buckets]  = acc;
            num_buckets++;
        }

        ok = 1;
        break;
    }

    if (!ok) {
        free(bin_counts); free(bin_offsets);
        free(bucketed); free(final_offsets); free(final_counts);
        fprintf(stderr, "[bucketing_build] Could not reach num_buckets ≤ total_dpus after refinements\n");
        return -15;
    }

    // -------- Sanity: buckets partition [0..N) --------
    uint64_t tot = 0;
    for (uint32_t i = 0; i < num_buckets; i++) {
        if (i == 0) {
            if (final_offsets[i] != 0u) {
                fprintf(stderr, "[bucketing_build] first offset %u != 0\n", final_offsets[i]);
                free(final_offsets); free(final_counts);
                free(bucketed); free(bin_counts); free(bin_offsets);
                return -12;
            }
        } else {
            uint32_t prev_end = final_offsets[i-1] + final_counts[i-1];
            if (final_offsets[i] != prev_end) {
                fprintf(stderr, "[bucketing_build] gap/overlap at i=%u (off=%u, prev_end=%u)\n",
                        i, final_offsets[i], prev_end);
                free(final_offsets); free(final_counts);
                free(bucketed); free(bin_counts); free(bin_offsets);
                return -13;
            }
        }
        tot += (uint64_t)final_counts[i];
    }
    if (tot != (uint64_t)N) {
        fprintf(stderr, "[bucketing_build] total=%" PRIu64 " != N=%u\n", tot, N);
        free(final_offsets); free(final_counts);
        free(bucketed); free(bin_counts); free(bin_offsets);
        return -14;
    }

    // Hand off
    *out_bucketed    = bucketed;
    *out_offsets     = final_offsets;
    *out_counts      = final_counts;
    *out_num_buckets = num_buckets;

    free(bin_counts);
    free(bin_offsets);
    return 0;
}

int verify_across_buckets(const uint32_t *bucketed,
                                 const uint32_t *final_offsets,
                                 const uint32_t *final_counts,
                                 uint32_t num_buckets)
{
    for (uint32_t i = 0; i < num_buckets; i++) {
        uint32_t off = final_offsets[i], cnt = final_counts[i];
        for (uint32_t k = 1; k < cnt; k++) {
            if (bucketed[off + k - 1] > bucketed[off + k]) {
                fprintf(stderr, "[VERIFY] Intra-bucket inversion at bucket %u, k=%u: %u > %u\n",
                        i, k, bucketed[off + k - 1], bucketed[off + k]);
                return -1;
            }
        }
        if (i + 1 < num_buckets) {
            uint32_t off2 = final_offsets[i + 1], cnt2 = final_counts[i + 1];
            if (cnt && cnt2) {
                uint32_t last  = bucketed[off + cnt - 1];
                uint32_t first = bucketed[off2];
                if (last > first) {
                    fprintf(stderr, "[VERIFY] Cross-bucket inversion between %u and %u: %u > %u\n",
                            i, i + 1, last, first);
                    return -1;
                }
            }
        }
    }
    return 0;
}

int verify_buckets_host_pre(const uint32_t *bucketed,
                                   const uint32_t *off,
                                   const uint32_t *cnt,
                                   uint32_t nb)
{
    // Verify cross-boundary order before DPUs touch the data.
    for (uint32_t i = 0; i + 1 < nb; i++) {
        uint32_t off0 = off[i],  c0 = cnt[i];
        uint32_t off1 = off[i+1], c1 = cnt[i+1];
        if (c0 == 0 || c1 == 0) continue; // empty buckets shouldn't happen, but skip

        uint32_t last0  = bucketed[off0 + c0 - 1];
        uint32_t first1 = bucketed[off1];

        if (last0 > first1) {
            fprintf(stderr,
                "[PRE] Cross-bucket inversion between %u and %u: %u > %u\n",
                i, i+1, last0, first1);
            // print some context windows
            uint32_t w0s = (c0 >= 8 ? c0 - 8 : 0);
            fprintf(stderr, "     tail of %u:", i);
            for (uint32_t z = 0; z < (c0 - w0s); z++) fprintf(stderr, " %u", bucketed[off0 + w0s + z]);
            fprintf(stderr, "\n     head of %u:", i+1);
            for (uint32_t z = 0; z < (c1 < 8 ? c1 : 8); z++) fprintf(stderr, " %u", bucketed[off1 + z]);
            fprintf(stderr, "\n");
            return -1;
        }
    }
    return 0;
}
