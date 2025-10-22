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
    if (cap == 0u)
        return -2;

    omp_set_num_threads(NUM_THREADS);

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

    // Start with a reasonable B; refine until max_bin <= cap.
    uint32_t B = 1u << 12;                  // 4096 to start
    if (B > N) B = (N > 0 ? N : 1u);        // never more bins than elements
    if (B < 1024u) B = 1024u;               // floor for stability

    uint32_t *bin_counts = NULL;
    uint32_t *bin_offsets = NULL;

    uint32_t *bucketed = NULL;
    uint32_t *final_offsets = NULL;
    uint32_t *final_counts  = NULL;
    uint32_t  num_buckets   = 0;

    // Expand B until no bin exceeds "cap".
    for (int attempt = 0; attempt < 16; attempt++) {
        free(bin_counts);  bin_counts  = NULL;
        free(bin_offsets); bin_offsets = NULL;

        bin_counts  = (uint32_t *)calloc((size_t)B, sizeof(uint32_t));
        bin_offsets = (uint32_t *)malloc((size_t)B * sizeof(uint32_t));
        if (!bin_counts || !bin_offsets) { free(bin_counts); free(bin_offsets); return -4; }

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
            // Need finer bins. Double B, but don't exceed N too much.
            uint64_t newB = (uint64_t)B << 1;
            if (newB > (uint64_t)N) newB = (uint64_t)N ? (uint64_t)N : (uint64_t)B + 1u;
            if (newB == B) newB = B + (B >> 1);
            if (newB == B) break; // cannot refine further
            B = (uint32_t)newB;
            continue;
        }

        // --- prefix
        uint32_t total = 0;
        for (uint32_t b = 0; b < B; b++) { bin_offsets[b] = total; total += bin_counts[b]; }
        if (total != N) { free(bin_counts); free(bin_offsets); return -6; }

        // Pack
        free(bucketed); bucketed = NULL;
        bucketed = (uint32_t *)malloc((size_t)N * sizeof(uint32_t));
        if (!bucketed) { free(bin_counts); free(bin_offsets); return -7; }

        // rebuild local hist to compute per-thread offsets
        T = omp_get_max_threads();
        uint32_t **local_bins2 = (uint32_t **)malloc((size_t)T * sizeof(uint32_t *));
        if (!local_bins2) { free(bucketed); free(bin_counts); free(bin_offsets); return -8; }
        uint32_t *lb_store2 = (uint32_t *)calloc((size_t)T * (size_t)B, sizeof(uint32_t));
        if (!lb_store2) { free(local_bins2); free(bucketed); free(bin_counts); free(bin_offsets); return -8; }
        for (int t = 0; t < T; t++) local_bins2[t] = lb_store2 + (size_t)t * (size_t)B;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            uint32_t *lb = local_bins2[tid];

            #pragma omp for schedule(static)
            for (uint32_t i = 0; i < N; i++) {
                uint64_t rel = (uint64_t)input[i] - (uint64_t)vmin;
                uint64_t b   = (uint64_t)((long double)B * (long double)rel / (long double)range);
                if (b >= (uint64_t)B) b = (uint64_t)B - 1ull;
                lb[(uint32_t)b] += 1u;
            }
        }

        // per-thread starting offsets per bin
        uint32_t **thread_prefix = (uint32_t **)malloc((size_t)T * sizeof(uint32_t *));
        if (!thread_prefix) { free(lb_store2); free(local_bins2); free(bucketed); free(bin_counts); free(bin_offsets); return -8; }
        uint32_t *tp_store = (uint32_t *)malloc((size_t)T * (size_t)B * sizeof(uint32_t));
        if (!tp_store) { free(thread_prefix); free(lb_store2); free(local_bins2); free(bucketed); free(bin_counts); free(bin_offsets); return -8; }
        for (int t = 0; t < T; t++) thread_prefix[t] = tp_store + (size_t)t * (size_t)B;

        #pragma omp parallel for schedule(static)
        for (uint32_t b = 0; b < B; b++) {
            uint32_t run = bin_offsets[b];
            for (int t = 0; t < T; t++) {
                thread_prefix[t][b] = run;
                run += local_bins2[t][b];
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
                uint64_t b   = (uint64_t)((long double)B * (long double)rel / (long double)range);
                if (b >= (uint64_t)B) b = (uint64_t)B - 1ull;
                uint32_t pos = cur[(uint32_t)b]++;
                bucketed[pos] = input[i];
            }
            if (cursor) free(cursor);
        }

        free(tp_store);
        free(thread_prefix);
        free(lb_store2);
        free(local_bins2);

        // --- Group bins into buckets (≤ cap). Keep grouping until all N placed.
        free(final_offsets); final_offsets = NULL;
        free(final_counts);  final_counts  = NULL;

        // Initial guess for number of buckets: ceil(N / cap) (+ small headroom)
        uint64_t est_buckets = (cap ? ((uint64_t)N + cap - 1) / cap : 1);
        uint32_t cap_buckets = (uint32_t)(est_buckets + 8u);
        if (cap_buckets == 0) cap_buckets = 8u;

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

        // Sanity: partition coverage
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

        /* ------------ ensure at least total_dpus buckets (BIN-BOUNDARY SPLITS) ------------ */
        if (total_dpus > 0 && num_buckets < total_dpus && N > 0) {
            const uint32_t target = total_dpus;

            // Map each existing bucket [off, off+cnt) to the contiguous bin range [sb, eb)
            uint32_t *b_start = (uint32_t *)malloc((size_t)num_buckets * sizeof(uint32_t));
            uint32_t *b_end   = (uint32_t *)malloc((size_t)num_buckets * sizeof(uint32_t));
            if (!b_start || !b_end) {
                free(b_start); free(b_end);
                free(final_offsets); free(final_counts);
                free(bucketed); free(bin_counts); free(bin_offsets);
                return -12;
            }

            for (uint32_t i = 0; i < num_buckets; i++) {
                uint32_t off = final_offsets[i];
                uint32_t end = final_offsets[i] + final_counts[i];

                // Find sb such that bin_offsets[sb] == off
                // (bin_offsets is non-decreasing; equals on non-empty bins)
                uint32_t sb = 0, eb = 0;
                int found_sb = 0, found_eb = 0;

                for (uint32_t b = 0; b < B; b++) {
                    if (bin_counts[b] && bin_offsets[b] == off) { sb = b; found_sb = 1; break; }
                }
                if (!found_sb) {
                    // Fallback (shouldn't happen): locate the first bin whose offset >= off
                    for (uint32_t b = 0; b < B; b++) { if (bin_offsets[b] >= off) { sb = b; found_sb = 1; break; } }
                }
                // Find eb as the first bin AFTER the last bin used by this bucket
                for (uint32_t b = sb; b < B; b++) {
                    if (bin_counts[b] == 0) continue;
                    uint32_t bend = bin_offsets[b] + bin_counts[b];
                    if (bend == end) { eb = b + 1; found_eb = 1; break; }
                }
                if (!found_eb) {
                    // Fallback: advance until we pass 'end'
                    for (uint32_t b = sb; b < B; b++) {
                        uint32_t bend = bin_offsets[b] + bin_counts[b];
                        if (bend >= end) { eb = b + 1; found_eb = 1; break; }
                    }
                }
                b_start[i] = sb;
                b_end[i]   = eb; // exclusive
            }

            // Decide desired parts per bucket, proportional to size, but ≤ number of bins in that bucket.
            uint32_t *parts = (uint32_t *)malloc((size_t)num_buckets * sizeof(uint32_t));
            long double *rem = (long double *)malloc((size_t)num_buckets * sizeof(long double));
            if (!parts || !rem) {
                free(parts); free(rem); free(b_start); free(b_end);
                free(final_offsets); free(final_counts);
                free(bucketed); free(bin_counts); free(bin_offsets);
                return -16;
            }

            uint32_t base_sum = 0;
            for (uint32_t i = 0; i < num_buckets; i++) {
                uint32_t cnt   = final_counts[i];
                uint32_t nbins = (b_end[i] > b_start[i] ? (b_end[i] - b_start[i]) : 0);
                long double exact = ((long double)cnt * (long double)target) / (long double)N;
                uint32_t p = (uint32_t)floorl(exact);
                if (p == 0u && cnt > 0) p = 1u;
                if (p > nbins) p = nbins;           // cannot split more than bin count
                parts[i] = p;
                rem[i]   = exact - (long double)p;
                base_sum += p;
            }

            // Adjust to match 'target' if possible.
            if (base_sum < target) {
                uint32_t deficit = target - base_sum;
                for (uint32_t k = 0; k < deficit; k++) {
                    int best = -1; long double best_rem = -1.0L;
                    for (uint32_t i = 0; i < num_buckets; i++) {
                        uint32_t nbins = (b_end[i] > b_start[i] ? (b_end[i] - b_start[i]) : 0);
                        if (final_counts[i] == 0 || parts[i] >= nbins) continue;
                        if (rem[i] > best_rem) { best_rem = rem[i]; best = (int)i; }
                    }
                    if (best < 0) break; // no more bin-boundary splits available
                    parts[best] += 1u;
                    rem[best] = 0.0L;
                    base_sum += 1u;
                    if (base_sum == target) break;
                }
            } else if (base_sum > target) {
                uint32_t surplus = base_sum - target;
                for (uint32_t k = 0; k < surplus; k++) {
                    int best = -1; long double best_rem = 10.0L;
                    for (uint32_t i = 0; i < num_buckets; i++) {
                        if (final_counts[i] == 0) continue;
                        if (parts[i] > 1u && rem[i] < best_rem) { best_rem = rem[i]; best = (int)i; }
                    }
                    if (best < 0) break;
                    parts[best] -= 1u;
                }
            }

            // Build refined buckets by assigning WHOLE BINS to each part, load-balancing by remaining.
            uint32_t max_new = 0;
            for (uint32_t i = 0; i < num_buckets; i++) max_new += (final_counts[i] ? parts[i] : 0u);

            uint32_t *new_offsets = (uint32_t *)malloc((size_t)max_new * sizeof(uint32_t));
            uint32_t *new_counts  = (uint32_t *)malloc((size_t)max_new * sizeof(uint32_t));
            if (!new_offsets || !new_counts) {
                free(new_offsets); free(new_counts);
                free(parts); free(rem); free(b_start); free(b_end);
                free(final_offsets); free(final_counts);
                free(bucketed); free(bin_counts); free(bin_offsets);
                return -17;
            }

            uint32_t write = 0;
            for (uint32_t i = 0; i < num_buckets; i++) {
                uint32_t p = parts[i];
                if (p == 0u || final_counts[i] == 0) continue;

                uint32_t sb = b_start[i], eb = b_end[i];
                uint32_t remain = final_counts[i];

                uint32_t cur_off = 0, cur_cnt = 0;
                uint32_t piece_target = (remain + p - 1u) / p; // balanced first target

                // Start at first bin of this bucket
                if (sb < B) cur_off = bin_offsets[sb];
                cur_cnt = 0;

                for (uint32_t b = sb; b < eb; b++) {
                    if (bin_counts[b] == 0) continue; // skip empty bins inside range
                    uint32_t boff = bin_offsets[b];
                    uint32_t bcnt = bin_counts[b];

                    if (cur_cnt == 0) cur_off = boff;
                    cur_cnt += bcnt;
                    remain -= bcnt;

                    // Close piece if we met target or we're at the last used bin
                    if (cur_cnt >= piece_target || b + 1 == eb) {
                        new_offsets[write] = cur_off;
                        new_counts[write]  = cur_cnt;
                        write++;

                        p--;
                        if (p == 0u) break;

                        cur_cnt = 0;
                        if (b + 1 < eb) cur_off = bin_offsets[b + 1];
                        piece_target = (remain + p - 1u) / p;
                    }
                }
            }

            free(final_offsets); free(final_counts);
            final_offsets = new_offsets;
            final_counts  = new_counts;
            num_buckets   = write;

            free(parts); free(rem); free(b_start); free(b_end);
        }
        /* ------------ END (bin-boundary) ------------- */


        // Hand off
        *out_bucketed    = bucketed;
        *out_offsets     = final_offsets;
        *out_counts      = final_counts;
        *out_num_buckets = num_buckets;

        free(bin_counts);
        free(bin_offsets);
        return 0;
    }

    // If we fall through attempts without success
    free(bin_counts); free(bin_offsets);
    fprintf(stderr, "[bucketing_build] Could not make all bins ≤ cap after refinements\n");
    return -15;
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
    if (nb < 2) {
        return 0;
    }

    int bad_i = INT_MAX;
    const int n_pairs = (int)nb - 1;

    #pragma omp parallel for schedule(static) reduction(min:bad_i)
    for (int i = 0; i < n_pairs; i++) {
        uint32_t off0 = off[i];
        uint32_t c0   = cnt[i];
        uint32_t off1 = off[i + 1];
        uint32_t c1   = cnt[i + 1];

        if (c0 == 0 || c1 == 0) {
            continue;
        }

        uint32_t last0  = bucketed[off0 + c0 - 1];
        uint32_t first1 = bucketed[off1];

        if (last0 > first1) {
            if (i < bad_i) {
                bad_i = i;
            }
        }
    }

    if (bad_i == INT_MAX) {
        return 0;
    }

    uint32_t i = (uint32_t)bad_i;
    uint32_t off0 = off[i];
    uint32_t c0   = cnt[i];
    uint32_t off1 = off[i + 1];
    uint32_t c1   = cnt[i + 1];

    uint32_t last0  = (c0 ? bucketed[off0 + c0 - 1] : 0);
    uint32_t first1 = (c1 ? bucketed[off1] : 0);

    fprintf(stderr,
            "[PRE] Cross-bucket inversion between %u and %u: %u > %u\n",
            i, i + 1, last0, first1);

    uint32_t w0s = (c0 >= 8 ? c0 - 8 : 0);
    fprintf(stderr, "     tail of %u:", i);
    for (uint32_t z = 0; z < (c0 - w0s); z++) {
        fprintf(stderr, " %u", bucketed[off0 + w0s + z]);
    }

    fprintf(stderr, "\n     head of %u:", i + 1);
    for (uint32_t z = 0; z < (c1 < 8 ? c1 : 8); z++) {
        fprintf(stderr, " %u", bucketed[off1 + z]);
    }
    fprintf(stderr, "\n");

    return -1;
}
