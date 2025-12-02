#define _GNU_SOURCE 1
#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#include "bucketing.h"

#include <fcntl.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <unistd.h> 
#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <time.h>
#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include <sys/syscall.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

int bucketing_build(const uint32_t *input, uint64_t N,
                    uint32_t total_dpus, uint32_t cap,
                    uint32_t **out_bucketed,
                    uint64_t **out_offsets,
                    uint32_t **out_counts,
                    uint32_t *out_num_buckets)
{
    if (!input || !out_bucketed || !out_offsets || !out_counts || !out_num_buckets || cap == 0u) return -1;

    omp_set_dynamic(0);
    omp_set_num_threads(NUM_THREADS);
    const int T = NUM_THREADS;
    const size_t CACHELINE = 64;

    // --- min/max ---
    uint32_t vmin = UINT32_MAX, vmax = 0;
    #pragma omp parallel
    {
        uint32_t lmin = UINT32_MAX, lmax = 0;
        #pragma omp for schedule(static)
        for (uint64_t i = 0; i < N; i++) {
            uint32_t v = input[i];
            if (v < lmin) lmin = v;
            if (v > lmax) lmax = v;
        }
        #pragma omp critical
        { if (lmin < vmin) vmin = lmin; if (lmax > vmax) vmax = lmax; }
    }
    uint64_t range = (uint64_t)vmax - (uint64_t)vmin + 1ull;
    if (range == 0ull) range = 1ull;

    // --- choose bins ---
    uint32_t B = 1u << 12;
    if (B > N) B = (uint32_t)(N ? N : 1u);
    if (B < 1024u) B = 1024u;

    // outputs
    uint32_t *bucketed = (uint32_t *)malloc((size_t)N * sizeof(uint32_t));
    uint64_t *final_offsets = NULL;
    uint32_t *final_counts  = NULL;
    uint32_t  num_buckets   = 0;

    // reusable work buffers (resized as B grows)
    uint32_t *bin_counts  = NULL;
    uint64_t *bin_offsets = NULL;

    // per-thread hist rows, cacheline padded
    typedef struct { uint32_t c; uint32_t pad; } c32;
    c32 **local_bins = NULL;
    c32  *lb_store   = NULL;

    for (int attempt = 0; attempt < 16; attempt++) {
        // (re)alloc sized by current B
        bin_counts  = (uint32_t *)realloc(bin_counts,  (size_t)B * sizeof(uint32_t));
        bin_offsets = (uint64_t *)realloc(bin_offsets, (size_t)B * sizeof(uint64_t));
        memset(bin_counts, 0, (size_t)B * sizeof(uint32_t));

        // per-thread hist layout
        size_t row_bytes = ((B * sizeof(c32) + (CACHELINE-1)) / CACHELINE) * CACHELINE;
        lb_store  = (c32 *)realloc(lb_store,  (size_t)T * row_bytes);
        memset(lb_store, 0, (size_t)T * row_bytes);
        local_bins = (c32 **)realloc(local_bins, (size_t)T * sizeof(c32 *));
        for (int t = 0; t < T; t++) local_bins[t] = (c32 *)((uint8_t*)lb_store + (size_t)t * row_bytes);

        // fixed-point scale: b = ((rel * scale) >> 32)
        const uint64_t scale = (((uint64_t)B) << 32) / range;

        // parallel histogram
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            c32 *lb = local_bins[tid];
            const uint64_t chunk = (N + (uint64_t)T - 1u) / (uint64_t)T;
            uint64_t beg = (uint64_t)tid * chunk;
            uint64_t end = beg + chunk; if (end > N) end = N;

            for (uint64_t i = beg; i < end; i++) {
                uint64_t rel = (uint64_t)input[i] - (uint64_t)vmin;
                uint64_t b   = (rel * scale) >> 32;
                if (b >= (uint64_t)B) b = (uint64_t)B - 1ull;
                lb[b].c += 1u;
            }
        }

        // reduce to bin_counts
        #pragma omp parallel for schedule(static)
        for (uint32_t b = 0; b < B; b++) {
            uint32_t s = 0;
            for (int t = 0; t < T; t++) s += local_bins[t][b].c;
            bin_counts[b] = s;
        }

        // cap check
        uint32_t max_bin = 0;
        #pragma omp parallel for reduction(max:max_bin) schedule(static)
        for (uint32_t b = 0; b < B; b++) if (bin_counts[b] > max_bin) max_bin = bin_counts[b];

        if (max_bin > cap) {
            uint64_t newB = (uint64_t)B << 1;
            if (newB > (uint64_t)N) newB = (uint64_t)(N ? N : B + 1u);
            if (newB == B) newB = B + (B >> 1);
            if (newB == B) break;
            B = (uint32_t)newB;
            continue;
        }

        // prefix over bins
        uint64_t run = 0;
        for (uint32_t b = 0; b < B; b++) { bin_offsets[b] = run; run += bin_counts[b]; }

        // per-thread starts: thread_prefix[t*B + b] = bin_offsets[b] + sum_{u<t} local_bins[u][b]
        uint64_t *thread_prefix = (uint64_t *)malloc((size_t)T * (size_t)B * sizeof(uint64_t));
        #pragma omp parallel for schedule(static)
        for (uint32_t b = 0; b < B; b++) {
            uint64_t acc = bin_offsets[b];
            for (int t = 0; t < T; t++) {
                thread_prefix[(size_t)t * B + b] = acc;
                acc += (uint64_t)local_bins[t][b].c;
            }
        }

        // scatter
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const uint64_t chunk = (N + (uint64_t)T - 1u) / (uint64_t)T;
            uint64_t beg = (uint64_t)tid * chunk;
            uint64_t end = beg + chunk; if (end > N) end = N;

            uint64_t *cur = (uint64_t *)malloc((size_t)B * sizeof(uint64_t));
            uint64_t *base = &thread_prefix[(size_t)tid * B];
            if (cur) memcpy(cur, base, (size_t)B * sizeof(uint64_t)); else cur = base;

            const uint64_t scale2 = scale; // keep in register
            for (uint64_t i = beg; i < end; i++) {
                uint64_t rel = (uint64_t)input[i] - (uint64_t)vmin;
                uint64_t b   = (rel * scale2) >> 32;
                if (b >= (uint64_t)B) b = (uint64_t)B - 1ull;
                uint64_t pos = cur[b]++;
                bucketed[pos] = input[i];
            }
            if (cur != base) free(cur);
        }
        free(thread_prefix);

        // bins -> buckets (≤ cap), coalescing contiguous bins
        uint64_t est_buckets = cap ? ((N + cap - 1) / cap) : 1;
        uint32_t cap_buckets = (uint32_t)(est_buckets + 8u);
        final_offsets = (uint64_t *)malloc((size_t)cap_buckets * sizeof(uint64_t));
        final_counts  = (uint32_t *)malloc((size_t)cap_buckets * sizeof(uint32_t));

        num_buckets = 0;
        uint32_t acc = 0;
        uint64_t cur_start = (B ? bin_offsets[0] : 0);

        for (uint32_t b = 0; b < B; b++) {
            uint32_t cnt = bin_counts[b];
            if (acc && (uint64_t)acc + (uint64_t)cnt > (uint64_t)cap) {
                if (num_buckets >= cap_buckets) {
                    cap_buckets *= 2u;
                    final_offsets = (uint64_t *)realloc(final_offsets, (size_t)cap_buckets * sizeof(uint64_t));
                    final_counts  = (uint32_t *)realloc(final_counts,  (size_t)cap_buckets * sizeof(uint32_t));
                }
                final_offsets[num_buckets] = cur_start;
                final_counts[num_buckets]  = acc;
                num_buckets++;
                cur_start = bin_offsets[b];
                acc = 0;
            }
            acc += cnt;
        }
        if (acc) {
            if (num_buckets >= cap_buckets) {
                cap_buckets += 1u;
                final_offsets = (uint64_t *)realloc(final_offsets, (size_t)cap_buckets * sizeof(uint64_t));
                final_counts  = (uint32_t *)realloc(final_counts,  (size_t)cap_buckets * sizeof(uint32_t));
            }
            final_offsets[num_buckets] = cur_start;
            final_counts[num_buckets]  = acc;
            num_buckets++;
        }

        /* Ensure num_buckets % total_dpus == 0 by splitting ONLY at bin boundaries.
        If not enough boundaries are available, we fall back to increasing B and asking
        the caller to redo (return -2). You can instead loop outside and rerun once with bigger B.
        */
        if (total_dpus > 0 && num_buckets > 0) {
            uint32_t m = total_dpus;
            uint32_t rem = num_buckets % m;
            if (rem != 0) {
                uint32_t need = ((num_buckets + m - 1u) / m) * m - num_buckets;  // extra buckets required

                // Map each bucket to its covering bin range [sb, eb) (bin-aligned buckets already have sb..eb contiguous)
                uint32_t *b_start = (uint32_t *)malloc((size_t)num_buckets * sizeof(uint32_t));
                uint32_t *b_end   = (uint32_t *)malloc((size_t)num_buckets * sizeof(uint32_t));

                // linear scan bins to locate ranges
                uint32_t bi = 0;
                for (uint32_t i = 0; i < num_buckets; i++) {
                    uint64_t off = final_offsets[i];
                    uint64_t end = off + (uint64_t)final_counts[i];

                    while (bi + 1 < B && bin_offsets[bi + 1] <= off) bi++;
                    uint32_t sb = bi;
                    while (bi < B && bin_offsets[bi] + (uint64_t)bin_counts[bi] < end) bi++;
                    uint32_t eb = (bi < B) ? (bi + 1) : B; // exclusive

                    b_start[i] = sb;
                    b_end[i]   = eb;
                }

                // How many *bin-boundary* splits are possible? (each multi-bin bucket with k bins yields up to k-1 splits)
                uint32_t avail_splits = 0;
                for (uint32_t i = 0; i < num_buckets; i++) {
                    uint32_t nbins = (b_end[i] > b_start[i]) ? (b_end[i] - b_start[i]) : 0u;
                    if (nbins > 1u) avail_splits += (nbins - 1u);
                }

                if (need > avail_splits) {
                    // Not enough bin boundaries to reach exact divisibility without in-bin cuts.
                    // Signal the caller to rerun with larger B (e.g., double B).
                    free(b_start); free(b_end);
                    free(final_offsets); free(final_counts);
                    free(bucketed);
                    free(bin_counts); free(bin_offsets);
                    free(local_bins); free(lb_store);
                    return -2; // REDO with bigger B (e.g., set B*=2 before re-entering)
                }

                // We can reach the target using ONLY bin boundaries.
                // Emit new bucket arrays by splitting some multi-bin buckets at internal bin boundaries.
                uint32_t target_nb = num_buckets + need;
                uint64_t *noff = (uint64_t *)malloc((size_t)target_nb * sizeof(uint64_t));
                uint32_t *ncnt = (uint32_t *)malloc((size_t)target_nb * sizeof(uint32_t));

                uint32_t w = 0;
                uint32_t need_left = need;

                for (uint32_t i = 0; i < num_buckets; i++) {
                    uint64_t off = final_offsets[i];
                    uint64_t end = off + (uint64_t)final_counts[i];
                    uint32_t sb = b_start[i], eb = b_end[i];
                    uint32_t nbins = (eb > sb) ? (eb - sb) : 0u;

                    if (need_left == 0 || nbins <= 1u) {
                        // keep as-is
                        noff[w] = off; ncnt[w] = final_counts[i]; w++;
                        continue;
                    }

                    // We may split this bucket at up to (nbins-1) inner bin boundaries, but no more than need_left.
                    // Walk bins inside [sb, eb) and cut after some of them until need_left is 0.
                    uint64_t piece_off = off;
                    uint32_t piece_cnt = 0;

                    for (uint32_t bb = sb; bb < eb; bb++) {
                        uint64_t boff = bin_offsets[bb];
                        uint32_t bcnt = bin_counts[bb];
                        if (bcnt == 0) continue;

                        if (piece_cnt == 0) piece_off = boff;
                        piece_cnt += bcnt;

                        // We can cut at this bin boundary (i.e., after this bin) if we still need more buckets
                        if (bb + 1 < eb && need_left > 0) {
                            noff[w] = piece_off; ncnt[w] = piece_cnt; w++;
                            need_left--;
                            piece_cnt = 0; // start next piece at next bin
                        }
                    }
                    // Emit the tail piece
                    if (piece_cnt > 0) { noff[w] = piece_off; ncnt[w] = piece_cnt; w++; }
                }

                // Swap in adjusted arrays (order preserved, boundaries only at bin limits)
                free(final_offsets); free(final_counts);
                final_offsets = noff; final_counts = ncnt; num_buckets = w;

                free(b_start); free(b_end);
                // Now: num_buckets % total_dpus == 0
            }
        }


        // handoff
        *out_bucketed    = bucketed;
        *out_offsets     = final_offsets;
        *out_counts      = final_counts;
        *out_num_buckets = num_buckets;

        free(bin_counts); free(bin_offsets);
        free(local_bins); free(lb_store);
        return 0;
    }

    free(bin_counts); free(bin_offsets);
    free(local_bins); free(lb_store);
    free(bucketed);
    return -1;
}


int verify_across_buckets(const uint32_t *bucketed,
                                 const uint64_t *final_offsets,
                                 const uint32_t *final_counts,
                                 uint32_t num_buckets)
{
    for (uint32_t i = 0; i < num_buckets; i++) {
        uint64_t off = final_offsets[i], cnt = final_counts[i];
        for (uint32_t k = 1; k < cnt; k++) {
            if (bucketed[off + k - 1] > bucketed[off + k]) {
                fprintf(stderr, "[VERIFY] Intra-bucket inversion at bucket %u, k=%u: %u > %u\n",
                        i, k, bucketed[off + k - 1], bucketed[off + k]);
                return -1;
            }
        }
        if (i + 1 < num_buckets) {
            uint64_t off2 = final_offsets[i + 1], cnt2 = final_counts[i + 1];
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
                            const uint64_t *off,
                            const uint32_t *cnt,
                            uint32_t nb,
                            uint64_t N_total)
{
    if (nb < 2 || N_total == 0) return 0;

    // ---- Structural validation (sequential) ----
    // Check offsets monotonicity and that each bucket fits within N_total.
    for (uint32_t i = 0; i < nb; i++) {
        if (i > 0 && off[i] < off[i - 1]) {
            fprintf(stderr, "[PRE] offsets not nondecreasing at i=%u: %" PRIu64 " < %" PRIu64 "\n",
                    i, off[i], off[i - 1]);
            return -2;
        }
        __uint128_t end128 = (__uint128_t)off[i] + (__uint128_t)cnt[i];
        if (end128 > (__uint128_t)N_total) {
            fprintf(stderr, "[PRE] bucket %u end out of range: off=%" PRIu64 " cnt=%u (end=%" PRIu64 "), N_total=%" PRIu64 "\n",
                    i, off[i], cnt[i], (uint64_t)end128, N_total);
            return -2;
        }
    }

    for (uint32_t i = 0; i + 1 < nb; i++) {
        if (off[i] + (uint64_t)cnt[i] > off[i + 1]) {
            fprintf(stderr, "[PRE] bucket %u overlaps next: end=%" PRIu64 " > next off=%" PRIu64 "\n",
                    i, off[i] + (uint64_t)cnt[i], off[i + 1]);
            return -2;
        }
    }

    // ---- Parallel cross-boundary inversion check ----
    int bad_i = INT_MAX;
    const int n_pairs = (int)nb - 1;

    #pragma omp parallel for schedule(static) reduction(min:bad_i)
    for (int i = 0; i < n_pairs; i++) {
        uint64_t off0 = off[i];
        uint32_t c0   = cnt[i];
        uint64_t off1 = off[i + 1];
        uint32_t c1   = cnt[i + 1];

        if (c0 == 0 || c1 == 0) continue;

        uint64_t end0 = off0 + (uint64_t)c0;
        uint64_t end1 = off1 + (uint64_t)c1;

        // Defensive caps (structure check already guarantees these)
        if (end0 > N_total || end1 > N_total) continue;

        uint32_t max0 = 0;
        for (uint64_t idx = off0; idx < end0; idx++) {
            uint32_t v = bucketed[(size_t)idx];
            if (v > max0) max0 = v;
        }

        uint32_t min1 = UINT32_MAX;
        for (uint64_t idx = off1; idx < end1; idx++) {
            uint32_t v = bucketed[(size_t)idx];
            if (v < min1) min1 = v;
        }

        if (max0 > min1 && i < bad_i) bad_i = i;
    }

    if (bad_i == INT_MAX) return 0;

    // ---- Detailed repro (clamped, no OOB even with bad metadata) ----
    uint32_t i = (uint32_t)bad_i;
    uint64_t off0 = off[i],     end0 = off0 + (uint64_t)cnt[i];
    uint64_t off1 = off[i + 1], end1 = off1 + (uint64_t)cnt[i + 1];

    if (end0 > N_total) end0 = N_total;
    if (end1 > N_total) end1 = N_total;

    uint32_t last0  = (off0 < end0) ? bucketed[(size_t)(end0 - 1)] : 0;
    uint32_t first1 = (off1 < end1) ? bucketed[(size_t)off1]       : 0;

    fprintf(stderr,
            "[PRE] Cross-bucket inversion between %u and %u: %u > %u\n",
            i, i + 1, last0, first1);

    uint64_t len0 = (end0 > off0) ? (end0 - off0) : 0;
    uint64_t tail = (len0 > 8 ? 8 : len0);
    fprintf(stderr, "     tail of %u:", i);
    for (uint64_t z = 0; z < tail; z++) {
        uint64_t idx = end0 - tail + z;
        fprintf(stderr, " %u", bucketed[(size_t)idx]);
    }

    uint64_t len1 = (end1 > off1) ? (end1 - off1) : 0;
    uint64_t head = (len1 > 8 ? 8 : len1);
    fprintf(stderr, "\n     head of %u:", i + 1);
    for (uint64_t z = 0; z < head; z++) {
        uint64_t idx = off1 + z;
        fprintf(stderr, " %u", bucketed[(size_t)idx]);
    }
    fprintf(stderr, "\n");

    return -1;
}
