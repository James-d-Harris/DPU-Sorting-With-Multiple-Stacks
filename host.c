#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#ifndef NR_DPUS
#define NR_DPUS 2546
#endif
#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif
#ifndef MAX_ELEMS_PER_DPU
#define MAX_ELEMS_PER_DPU 16384u
#endif

#ifndef DPU_EXE
#define DPU_EXE "./quicksort_dpu"
#endif

// Number of independent stacks to drive in parallel.
// Must divide TOTAL_RANKS exactly.
#ifndef NUM_SETS
#define NUM_SETS 8
#endif

#ifndef TOTAL_RANKS
#define TOTAL_RANKS 40
#endif

#ifndef MRAM_SYM
#define MRAM_SYM "MRAM_BASE"
#endif

#define SYM_MRAM_ARR "MRAM_ARR"
#define SYM_STATS    "STATS"

#ifndef COARSE_BITS
#define COARSE_BITS 18u
#endif
#define COARSE_BINS (1u << COARSE_BITS)

#ifndef SUB_BITS
#define SUB_BITS 12u
#endif
#define SUB_BINS (1u << SUB_BITS)


struct dpu_stats {
    uint32_t n_elems;
    uint32_t nr_tasklets;
    uint32_t cycles_sort;     // duration of sort region
    uint32_t cycles_total;    // timestamp at end (since perfcounter_config)
    uint32_t cycles_start;    // timestamp at start of sort region
};

typedef struct __attribute__((packed, aligned(8))) {
    uint32_t v;
    uint32_t pad;
} elem_t;

// ---- Fast bucket index helpers ----
static inline uint32_t fast_bucket_fp(uint32_t x, uint32_t min_val, uint32_t nbins, double inv_range)
{
    double d = ((double)(x - min_val)) * inv_range;   // scale into [0, nbins)
    uint64_t bi = (uint64_t)d;                        // truncate
    if (bi >= (uint64_t)nbins) bi = (uint64_t)nbins - 1ull;
    return (uint32_t)bi;
}

// For mapping heavy coarse bins using a refined sub-histogram
typedef struct {
    uint32_t coarse_bin; // which coarse bin this mapping applies to
    uint32_t first_bucket; // first global bucket index assigned to this coarse bin
    uint32_t num_buckets; // how many buckets this coarse bin consumes
    uint32_t subbins;  // SUB_BINS
    uint32_t *subbin_to_bucket; // length = subbins, maps subbin -> absolute bucket id
    uint64_t low_u64, high_u64; // numeric range [low, high) of the coarse bin
    double inv_sub_range; // subbins / (high - low)
} heavy_map_t;

static inline void coarse_bounds_u64(uint32_t bin, uint32_t min_val, uint64_t range,
                                     uint32_t nbins, uint64_t *lo, uint64_t *hi)
{
    uint64_t w  = range;
    uint64_t lo_u64 = (uint64_t)min_val + ((uint64_t)bin       * w) / (uint64_t)nbins;
    uint64_t hi_u64 = (uint64_t)min_val + ((uint64_t)(bin + 1) * w) / (uint64_t)nbins;
    if (hi_u64 <= lo_u64) hi_u64 = lo_u64 + 1ull;
    *lo = lo_u64; *hi = hi_u64;
}

static inline uint64_t ns_diff(const struct timespec a, const struct timespec b) {
    return (uint64_t)(b.tv_sec - a.tv_sec) * 1000000000ull + (uint64_t)(b.tv_nsec - a.tv_nsec);
}

static inline uint64_t ts_to_ns(struct timespec t) {
    return ((uint64_t)t.tv_sec * 1000000000ull) + (uint64_t)t.tv_nsec;
}

// Sharding helpers
static void shard_even(uint32_t total, uint32_t parts, uint32_t idx, uint32_t *start_out, uint32_t *count_out) {
    uint32_t base = total / parts;
    uint32_t rem = total % parts;

    uint32_t count = base;
    if (idx < rem) {
        count = base + 1;
    }

    uint32_t start = idx * base + (idx < rem ? idx : rem);

    *start_out = start;
    *count_out = count;
}

static void fill_dpu_shards(uint32_t N_set, uint32_t nb_dpus, uint32_t *starts, uint32_t *counts) {
    uint32_t i = 0;
    for (i=0; i < nb_dpus; i++) {
        uint32_t s = 0;
        uint32_t c = 0;

        shard_even(N_set, nb_dpus, i, &s, &c);

        starts[i] = s;
        counts[i] = c;
    }
}


// ---- K-way merge for many sorted runs ----
struct heap_node {
    uint32_t value;
    uint32_t run_idx;
};

static void heap_swap(struct heap_node *a, struct heap_node *b) {
    struct heap_node tmp = *a;
    *a = *b;
    *b = tmp;
}

static void heap_sift_down(struct heap_node *heap, uint32_t heap_size, uint32_t i) {
    while (1) {
        uint32_t left  = 2u * i + 1u;
        uint32_t right = 2u * i + 2u;
        uint32_t smallest = i;

        if (left < heap_size) {
            if (heap[left].value < heap[smallest].value) {
                smallest = left;
            }
        }

        if (right < heap_size) {
            if (heap[right].value < heap[smallest].value) {
                smallest = right;
            }
        }

        if (smallest == i) {
            break;
        }

        heap_swap(&heap[i], &heap[smallest]);
        i = smallest;
    }
}

/**
 * Merge nb_runs sorted subarrays from base_ptr into a single sorted array in place.
 * Each run i has [starts[i], starts[i] + counts[i]) relative to base_ptr.
 * Returns 0 on success, -1 on OOM.
 */
static int merge_sorted_runs(uint32_t *base_ptr, uint32_t N_set, const uint32_t *starts, const uint32_t *counts, uint32_t nb_runs) {
    if (N_set <= 1u) {
        return 0;
    }

    uint32_t *tmp = (uint32_t *)malloc((size_t)N_set * sizeof(uint32_t));
    if (tmp == NULL) {
        fprintf(stderr, "[merge] OOM: tmp buffer of %u elems\n", N_set);
        return -1;
    }

    uint32_t *pos = (uint32_t *)calloc(nb_runs, sizeof(uint32_t));
    if (pos == NULL) {
        fprintf(stderr, "[merge] OOM: pos array of %u runs\n", nb_runs);
        free(tmp);
        return -1;
    }

    struct heap_node *heap = (struct heap_node *)malloc((size_t)nb_runs * sizeof(struct heap_node));
    if (heap == NULL) {
        fprintf(stderr, "[merge] OOM: heap of %u nodes\n", nb_runs);
        free(tmp);
        free(pos);
        return -1;
    }

    uint32_t heap_size = 0u;
    uint32_t i = 0u;
    for (i = 0u; i < nb_runs; i++) {
        if (counts[i] > 0u) {
            struct heap_node node;
            node.run_idx = i;
            node.value   = base_ptr[starts[i]];
            heap[heap_size] = node;
            heap_size += 1u;
        }
    }

    if (heap_size > 1u) {
        int32_t start = (int32_t)(heap_size / 2u) - 1;
        while (start >= 0) {
            heap_sift_down(heap, heap_size, (uint32_t)start);
            start -= 1;
        }
    }

    uint32_t out_idx = 0u;
    while (heap_size > 0u) {
        struct heap_node min = heap[0];
        tmp[out_idx] = min.value;
        out_idx += 1u;

        uint32_t r = min.run_idx;
        pos[r] += 1u;

        if (pos[r] < counts[r]) {
            uint32_t next_val = base_ptr[starts[r] + pos[r]];
            heap[0].value   = next_val;
            heap[0].run_idx = r;
            heap_sift_down(heap, heap_size, 0u);
        } else {
            heap_size -= 1u;
            if (heap_size > 0u) {
                heap[0] = heap[heap_size];
                heap_sift_down(heap, heap_size, 0u);
            }
        }
    }

    if (out_idx == N_set) {
        uint32_t j = 0u;
        for (j = 0u; j < N_set; j++) {
            base_ptr[j] = tmp[j];
        }
    } else {
        fprintf(stderr, "[merge] output size mismatch: wrote %u of %u\n", out_idx, N_set);
    }

    free(heap);
    free(pos);
    free(tmp);
    return 0;
}





// ---------- Sanity helpers ----------
static inline int is_sorted_u32(const uint32_t *a, uint32_t n) {
    if (n <= 1u) {
        return 1;
    }
    for (uint32_t i = 1; i < n; i++) {
        if (a[i-1] > a[i]) {
            return 0;
        }
    }
    return 1;
}

/* Verify that final_offsets/final_counts partition [0..N) with no gaps/overlaps,
   and total == N. Also ensure monotone offsets and counts > 0 (unless N==0). */
static int check_bucket_layout(const uint32_t *final_offsets,
                               const uint32_t *final_counts,
                               uint32_t num_buckets,
                               uint32_t N,
                               int verbose) {
    if (num_buckets == 0u) {
        if (N == 0u) {
            return 0;
        } else {
            if (verbose) fprintf(stderr, "[layout] zero buckets but N=%u\n", N);
            return -1;
        }
    }

    uint64_t total = 0;
    for (uint32_t i = 0; i < num_buckets; i++) {
        uint32_t off = final_offsets[i];
        uint32_t cnt = final_counts[i];

        if (i == 0) {
            if (off != 0u) {
                if (verbose) fprintf(stderr, "[layout] first offset %u != 0\n", off);
                return -1;
            }
        } else {
            uint32_t prev_end = final_offsets[i-1] + final_counts[i-1];
            if (off != prev_end) {
                if (verbose) fprintf(stderr, "[layout] gap/overlap at bucket %u: off=%u prev_end=%u\n",
                                     i, off, prev_end);
                return -1;
            }
        }

        if (off > N || (uint64_t)off + (uint64_t)cnt > (uint64_t)N) {
            if (verbose) fprintf(stderr, "[layout] out-of-range bucket %u: off=%u cnt=%u N=%u\n",
                                 i, off, cnt, N);
            return -1;
        }

        if (cnt == 0u) {
            if (verbose) fprintf(stderr, "[layout] empty bucket %u\n", i);
            return -1;
        }

        total += (uint64_t)cnt;
    }

    if (total != (uint64_t)N) {
        if (verbose) fprintf(stderr, "[layout] total counts=%" PRIu64 " != N=%u\n", total, N);
        return -1;
    }

    return 0;
}

static inline int is_sorted_elemv(const elem_t *a, uint32_t n) {
    if (n <= 1u) {
        return 1;
    }
    for (uint32_t i = 1; i < n; i++) {
        if (a[i - 1].v > a[i].v) {
            return 0;
        }
    }
    return 1;
}



int dpu_sort(uint32_t set_id, uint32_t *base_ptr, uint32_t N_set, uint64_t *alloc_start_ns, uint64_t *d2h_end_ns) {
    struct timespec t_alloc_s, t_alloc_e;
    struct timespec t_load_s, t_load_e;
    struct timespec t_h2d_s, t_h2d_e;
    struct timespec t_exec_s, t_exec_e;
    struct timespec t_d2h_s, t_d2h_e;


    clock_gettime(CLOCK_MONOTONIC, &t_alloc_s);
    if (alloc_start_ns != NULL) {
        *alloc_start_ns = ts_to_ns(t_alloc_s);
    }

    struct dpu_set_t set;
    dpu_error_t derr = dpu_alloc_ranks(TOTAL_RANKS / NUM_SETS, NULL, &set);
    if(derr != DPU_OK) {
        fprintf(stderr, "[Set %u] alloc_ranks failed: %s\n", set_id, dpu_error_to_string(derr));
        return 1;
    }

    uint32_t nb_dpus = 0;
    DPU_ASSERT(dpu_get_nr_dpus(set, &nb_dpus));
    if ( nb_dpus == 0 ) {
        fprintf(stderr, "No DPUs allocated.\n");
        return 1;
    }
    printf("%u DPUs allocated for DPU set %u.\n", nb_dpus, set_id);

    clock_gettime(CLOCK_MONOTONIC, &t_alloc_e);

    clock_gettime(CLOCK_MONOTONIC, &t_load_s);
    DPU_ASSERT(dpu_load(set, DPU_EXE, NULL));
    clock_gettime(CLOCK_MONOTONIC, &t_load_e);

    // Compute per-DPU shards
    uint32_t *starts = (uint32_t *)malloc(nb_dpus * sizeof(uint32_t));
    uint32_t *counts = (uint32_t *)malloc(nb_dpus * sizeof(uint32_t));
    if (starts == NULL || counts == NULL) {
        fprintf(stderr, "[Set %u] OOM allocating shard arrays.\n", set_id);
        if (starts) free(starts);
        if (counts) free(counts);
        DPU_ASSERT(dpu_free(set));
        return 1;
    }

    fill_dpu_shards(N_set, nb_dpus, starts, counts);

    uint32_t i = 0;
    for (i = 0; i < nb_dpus; i++) {
        if (counts[i] > MAX_ELEMS_PER_DPU) {
            fprintf(stderr, "[Set %u] Shard %u too large for MRAM (%u > %u)\n", set_id, i, counts[i], MAX_ELEMS_PER_DPU);
            free(starts);
            free(counts);
            DPU_ASSERT(dpu_free(set));
            return 1;
        }
    }

    // Compute max shard for uniform push size
    uint32_t max_cnt = 0;
    for (i = 0; i < nb_dpus; i++) {
        if (counts[i] > max_cnt) {
            max_cnt = counts[i];
        }
    }
    size_t max_bytes = (size_t)max_cnt * sizeof(elem_t);

    // Stage per-DPU H2D buffers and STATS blocks
    elem_t **h2d_bufs = (elem_t **)calloc(nb_dpus, sizeof(elem_t *));
    struct dpu_stats *h2d_stats = (struct dpu_stats *)calloc(nb_dpus, sizeof(struct dpu_stats));
    if (h2d_bufs == NULL || h2d_stats == NULL) {
        fprintf(stderr, "[Set %u] OOM: staging arrays.\n", set_id);
        free(starts);
        free(counts);
        if (h2d_bufs) free(h2d_bufs);
        if (h2d_stats) free(h2d_stats);
        DPU_ASSERT(dpu_free(set));
        return 1;
    }

    uint32_t d = 0;
    for (d = 0; d < nb_dpus; d++) {
        if (max_bytes > 0) {
            h2d_bufs[d] = (elem_t *)aligned_alloc(8, max_bytes);
            if (h2d_bufs[d] == NULL) {
                fprintf(stderr, "[Set %u] OOM: h2d_bufs[%u]\n", set_id, d);
                // cleanup below
                break;
            }

            // Pack the shard for this DPU; any tail beyond counts[d] is left as don't-care
            uint32_t cnt = counts[d];

            uint32_t k = 0;
            for (k = 0; k < cnt; k++) {
                h2d_bufs[d][k].v   = base_ptr[starts[d] + k];
                h2d_bufs[d][k].pad = 0u;
            }
            for (; k < max_cnt; k++) {
                // Not strictly necessary; zeroing keeps things deterministic
                h2d_bufs[d][k].v   = 0u;
                h2d_bufs[d][k].pad = 0u;
            }
        }

        // Fill STATS as "args": set n_elems
        h2d_stats[d].n_elems     = counts[d];
        h2d_stats[d].nr_tasklets = NR_TASKLETS;
        h2d_stats[d].cycles_sort = 0u;
        h2d_stats[d].cycles_total= 0u;
    }
    if (d != nb_dpus && max_bytes > 0) {
        // allocation failure path
        for (uint32_t t = 0; t < d; t++) {
            if (h2d_bufs[t]) free(h2d_bufs[t]);
        }
        free(h2d_bufs);
        free(h2d_stats);
        free(starts);
        free(counts);
        DPU_ASSERT(dpu_free(set));
        return 1;
    }


    clock_gettime(CLOCK_MONOTONIC, &t_h2d_s);

    if (max_bytes > 0) {
        struct dpu_set_t dpu;
        uint32_t idx = 0;

        DPU_FOREACH(set, dpu) {
            if (idx >= nb_dpus) {
                break;
            }

            DPU_ASSERT(dpu_prepare_xfer(dpu, h2d_bufs[idx]));
            idx += 1;
        }

        DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, SYM_MRAM_ARR, 0, max_bytes, DPU_XFER_DEFAULT));
    }

    {
        struct dpu_set_t dpu;
        uint32_t idx = 0;

        DPU_FOREACH(set, dpu) {
            if (idx >= nb_dpus) {
                break;
            }

            DPU_ASSERT(dpu_prepare_xfer(dpu, &h2d_stats[idx]));
            idx += 1;
        }

        DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, SYM_STATS, 0, sizeof(struct dpu_stats), DPU_XFER_DEFAULT));
    }

    clock_gettime(CLOCK_MONOTONIC, &t_h2d_e);


    clock_gettime(CLOCK_MONOTONIC, &t_exec_s);
    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
    clock_gettime(CLOCK_MONOTONIC, &t_exec_e);


    clock_gettime(CLOCK_MONOTONIC, &t_d2h_s);

    elem_t **d2h_bufs = NULL;
    if (max_bytes > 0) {
        d2h_bufs = (elem_t **)calloc(nb_dpus, sizeof(elem_t *));
        if (d2h_bufs == NULL) {
            fprintf(stderr, "[Set %u] OOM: d2h_bufs.\n", set_id);
            // cleanup below
        } else {
            for (d = 0; d < nb_dpus; d++) {
                d2h_bufs[d] = (elem_t *)aligned_alloc(8, max_bytes);
                if (d2h_bufs[d] == NULL) {
                    fprintf(stderr, "[Set %u] OOM: d2h_bufs[%u]\n", set_id, d);
                    break;
                }
            }
            if (d != nb_dpus) {
                for (uint32_t t = 0; t < d; t++) {
                    if (d2h_bufs[t]) free(d2h_bufs[t]);
                }
                free(d2h_bufs);
                d2h_bufs = NULL;
            }
        }

        if (d2h_bufs != NULL) {
            struct dpu_set_t dpu;
            uint32_t idx = 0;

            DPU_FOREACH(set, dpu) {
                if (idx >= nb_dpus) {
                    break;
                }

                DPU_ASSERT(dpu_prepare_xfer(dpu, d2h_bufs[idx]));
                idx += 1;
            }

            DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, SYM_MRAM_ARR, 0, max_bytes,  DPU_XFER_DEFAULT));
        }
    }

    struct dpu_stats *stats_out = (struct dpu_stats *)calloc(nb_dpus, sizeof(struct dpu_stats));
    if (stats_out == NULL) {
        fprintf(stderr, "[Set %u] OOM: stats_out.\n", set_id);
        // continue; not fatal for sorting result
    } else {
        struct dpu_set_t dpu;
        uint32_t idx = 0;

        DPU_FOREACH(set, dpu) {
            if (idx >= nb_dpus) {
                break;
            }

            DPU_ASSERT(dpu_prepare_xfer(dpu, &stats_out[idx]));
            idx += 1;
        }

        DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, SYM_STATS, 0, sizeof(struct dpu_stats), DPU_XFER_DEFAULT));
    }

    // Unpack only the first counts[d] elements back into the original array
    if (d2h_bufs != NULL) {
        for (d = 0; d < nb_dpus; d++) {
            uint32_t cnt = counts[d];

            uint32_t k = 0;
            for (k = 0; k < cnt; k++) {
                base_ptr[starts[d] + k] = d2h_bufs[d][k].v;
            }
        }
    }

    // Merge all per-DPU runs into one sorted set slice
    {
        int mrc = merge_sorted_runs(base_ptr, N_set, starts, counts, nb_dpus);
        if (mrc != 0) {
            fprintf(stderr, "[Set %u] Merge failed (rc=%d)\n", set_id, mrc);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_d2h_e);
    if (d2h_end_ns != NULL) {
        *d2h_end_ns = ts_to_ns(t_d2h_e);
    }

    // Cleanup
    if (h2d_bufs != NULL) {
        for (d = 0; d < nb_dpus; d++) {
            if (h2d_bufs[d] != NULL) {
                free(h2d_bufs[d]);
            }
        }
        free(h2d_bufs);
    }
    if (stats_out != NULL) {
        uint64_t min_c = UINT64_MAX;
        uint64_t max_c = 0;
        double   sum_c = 0.0;

        for (d = 0; d < nb_dpus; d++) {
            uint64_t c = stats_out[d].cycles_sort;

            if (c < min_c) {
                min_c = c;
            }
            if (c > max_c) {
                max_c = c;
            }
            sum_c += (double)c;
        }

        double avg_c = (nb_dpus > 0) ? (sum_c / (double)nb_dpus) : 0.0;
        printf("[Set %u] cycles_sort min=%" PRIu64 " avg=%.1f max=%" PRIu64 "\n",
               set_id, min_c, avg_c, max_c);

        free(stats_out);
    }

    free(starts);
    free(counts);
    free(h2d_stats);
    DPU_ASSERT(dpu_free(set));

    
    // Report per-set timings
    double ms_alloc = (double)ns_diff(t_alloc_s, t_alloc_e) / 1e6;
    double ms_load  = (double)ns_diff(t_load_s,  t_load_e)  / 1e6;
    double ms_h2d   = (double)ns_diff(t_h2d_s,   t_h2d_e)   / 1e6;
    double ms_exec  = (double)ns_diff(t_exec_s,  t_exec_e)  / 1e6;
    double ms_d2h   = (double)ns_diff(t_d2h_s,   t_d2h_e)   / 1e6;

    printf("[Set %u] ms alloc=%.3f, load=%.3f, H2D=%.3f, exec=%.3f, D2H=%.3f\n", set_id, ms_alloc, ms_load, ms_h2d, ms_exec, ms_d2h);

    return 0;
}

// Bucketed DPU run: one (or more) whole buckets per DPU, no merge needed.
int dpu_sort_bucketed(uint32_t set_id,
                  uint32_t *bucketed,
                  const uint32_t *bucket_offsets,
                  const uint32_t *bucket_counts,
                  uint32_t num_buckets,
                  uint32_t *next_bucket_cursor,
                  uint64_t *alloc_start_ns,
                  uint64_t *d2h_end_ns) {
    struct timespec t_alloc_s, t_alloc_e;
    struct timespec t_load_s,  t_load_e;
    struct timespec t_h2d_s,   t_h2d_e;
    struct timespec t_exec_s,  t_exec_e;
    struct timespec t_d2h_s,   t_d2h_e;

    // --- allocate ranks for this set ---
    clock_gettime(CLOCK_MONOTONIC, &t_alloc_s);
    if (alloc_start_ns) {
        *alloc_start_ns = ts_to_ns(t_alloc_s);
    }

    struct dpu_set_t set;
    dpu_error_t derr = dpu_alloc_ranks(TOTAL_RANKS / NUM_SETS, NULL, &set);
    if (derr != DPU_OK) {
        fprintf(stderr, "[Set %u] alloc_ranks failed: %s\n", set_id, dpu_error_to_string(derr));
        return 1;
    }

    uint32_t nb_dpus = 0;
    DPU_ASSERT(dpu_get_nr_dpus(set, &nb_dpus));
    if (nb_dpus == 0) {
        DPU_ASSERT(dpu_free(set));
        return 0; // nothing to do
    }

    // claim a contiguous range of buckets [b0, bN)
    uint32_t b0 = 0;
    uint32_t my_nd = 0;

    #pragma omp critical
    {
        uint32_t cur = *next_bucket_cursor;

        if (cur >= num_buckets) {
            b0    = num_buckets;
            my_nd = 0;                 // nothing to do
        } else {
            uint32_t remain = num_buckets - cur;
            uint32_t take   = (remain < nb_dpus) ? remain : nb_dpus;

            b0    = cur;
            my_nd = take;

            *next_bucket_cursor = cur + take;  // advance by what we actually took
        }
    }

    if (my_nd == 0) {
        // No buckets left; free and exit early
        DPU_ASSERT(dpu_free(set));
        return 0;
    }

    uint32_t bN = b0 + my_nd;  // exclusive end


    if (b0 >= num_buckets) {
        // No buckets left; free and exit
        DPU_ASSERT(dpu_free(set));
        return 0;
    }
    if (bN > num_buckets) {
        bN = num_buckets; // clamp if we over-claimed
    }

    clock_gettime(CLOCK_MONOTONIC, &t_alloc_e);

    // --- load program ---
    clock_gettime(CLOCK_MONOTONIC, &t_load_s);
    DPU_ASSERT(dpu_load(set, DPU_EXE, NULL));
    clock_gettime(CLOCK_MONOTONIC, &t_load_e);

    // --- compute max bucket size among [b0, bN) and guard MRAM ---
    uint32_t max_cnt = 0;
    for (uint32_t i = 0; i < my_nd; i++) {
        uint32_t cnt = bucket_counts[b0 + i];
        if (cnt > MAX_ELEMS_PER_DPU) {
            fprintf(stderr, "[Set %u] bucket %u too large (%u > %u)\n",
                    set_id, b0 + i, cnt, MAX_ELEMS_PER_DPU);
            DPU_ASSERT(dpu_free(set));
            return 1;
        }
        if (cnt > max_cnt) {
            max_cnt = cnt;
        }
    }
    size_t max_bytes = (size_t)max_cnt * sizeof(elem_t);

    // --- stage H2D buffers + STATS blocks (only for used DPUs) ---
    elem_t **h2d_bufs = (elem_t **)calloc(my_nd, sizeof(elem_t *));
    struct dpu_stats *h2d_stats = (struct dpu_stats *)calloc(my_nd, sizeof(struct dpu_stats));
    if (!h2d_bufs || !h2d_stats) {
        fprintf(stderr, "[Set %u] OOM staging\n", set_id);
        if (h2d_bufs)  free(h2d_bufs);
        if (h2d_stats) free(h2d_stats);
        DPU_ASSERT(dpu_free(set));
        return 1;
    }

    for (uint32_t d = 0; d < my_nd; d++) {
        if (max_bytes > 0) {
            h2d_bufs[d] = (elem_t *)aligned_alloc(8, max_bytes);
            if (!h2d_bufs[d]) {
                fprintf(stderr, "[Set %u] OOM: h2d_bufs[%u]\n", set_id, d);
                for (uint32_t t = 0; t < d; t++) free(h2d_bufs[t]);
                free(h2d_bufs); free(h2d_stats);
                DPU_ASSERT(dpu_free(set));
                return 1;
            }

            const uint32_t gb   = b0 + d;                      // global bucket index
            const uint32_t off  = bucket_offsets[gb];          // start in `bucketed`
            const uint32_t cnt  = bucket_counts[gb];

            uint32_t k = 0;
            for (; k < cnt; k++) {
                h2d_bufs[d][k].v   = bucketed[off + k];
                h2d_bufs[d][k].pad = 0u;
            }
            for (; k < max_cnt; k++) {
                h2d_bufs[d][k].v   = 0u;
                h2d_bufs[d][k].pad = 0u;
            }
        }

        h2d_stats[d].n_elems      = bucket_counts[b0 + d];
        h2d_stats[d].nr_tasklets  = NR_TASKLETS;
        h2d_stats[d].cycles_sort  = 0;
        h2d_stats[d].cycles_total = 0;
        h2d_stats[d].cycles_start = 0;
    }

    // --- H2D transfers ---
    clock_gettime(CLOCK_MONOTONIC, &t_h2d_s);

    if (max_bytes > 0) {
        struct dpu_set_t dpu;
        uint32_t idx = 0, used = 0;
        DPU_FOREACH(set, dpu) {
            if (idx >= my_nd) break;  // only load the DPUs we actually have buckets for
            DPU_ASSERT(dpu_prepare_xfer(dpu, h2d_bufs[idx]));
            idx++;
        }
        used = my_nd;
        if (used > 0) {
            DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, SYM_MRAM_ARR, 0, max_bytes, DPU_XFER_DEFAULT));
        }
    }
    {
        struct dpu_set_t dpu;
        uint32_t idx = 0;
        DPU_FOREACH(set, dpu) {
            if (idx >= my_nd) break;
            DPU_ASSERT(dpu_prepare_xfer(dpu, &h2d_stats[idx]));
            idx++;
        }
        if (my_nd > 0) {
            DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, SYM_STATS, 0, sizeof(struct dpu_stats), DPU_XFER_DEFAULT));
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_h2d_e);

    // --- launch ---
    clock_gettime(CLOCK_MONOTONIC, &t_exec_s);
    if (my_nd > 0) {
        DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
    }
    clock_gettime(CLOCK_MONOTONIC, &t_exec_e);

    // --- D2H transfers ---
    clock_gettime(CLOCK_MONOTONIC, &t_d2h_s);

    elem_t **d2h_bufs = NULL;
    if (max_bytes > 0 && my_nd > 0) {
        d2h_bufs = (elem_t **)calloc(my_nd, sizeof(elem_t *));
        if (!d2h_bufs) {
            fprintf(stderr, "[Set %u] OOM: d2h_bufs\n", set_id);
        } else {
            for (uint32_t d = 0; d < my_nd; d++) {
                d2h_bufs[d] = (elem_t *)aligned_alloc(8, max_bytes);
                if (!d2h_bufs[d]) {
                    fprintf(stderr, "[Set %u] OOM: d2h_bufs[%u]\n", set_id, d);
                    for (uint32_t t = 0; t < d; t++) free(d2h_bufs[t]);
                    free(d2h_bufs);
                    d2h_bufs = NULL;
                    break;
                }
            }
        }
        if (d2h_bufs) {
            struct dpu_set_t dpu;
            uint32_t idx = 0;
            DPU_FOREACH(set, dpu) {
                if (idx >= my_nd) break;
                DPU_ASSERT(dpu_prepare_xfer(dpu, d2h_bufs[idx]));
                idx++;
            }
            DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, SYM_MRAM_ARR, 0, max_bytes, DPU_XFER_DEFAULT));
        }
    }

    // pull back STATS
    struct dpu_stats *stats_out = NULL;
    if (my_nd > 0) {
        stats_out = (struct dpu_stats *)calloc(my_nd, sizeof(struct dpu_stats));
        if (stats_out) {
            struct dpu_set_t dpu;
            uint32_t idx = 0;
            DPU_FOREACH(set, dpu) {
                if (idx >= my_nd) break;
                DPU_ASSERT(dpu_prepare_xfer(dpu, &stats_out[idx]));
                idx++;
            }
            DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, SYM_STATS, 0, sizeof(struct dpu_stats), DPU_XFER_DEFAULT));

            uint64_t min_c = UINT64_MAX, max_c = 0; double sum_c = 0.0;
            for (uint32_t d = 0; d < my_nd; d++) {
                uint64_t c = stats_out[d].cycles_sort;
                if (c < min_c) min_c = c;
                if (c > max_c) max_c = c;
                sum_c += (double)c;
            }
            double avg_c = my_nd ? (sum_c / (double)my_nd) : 0.0;
            printf("[Set %u] cycles_sort min=%" PRIu64 " avg=%.1f max=%" PRIu64 "\n",
                   set_id, min_c, avg_c, max_c);
        }
    }

    // Write results back to their original positions (bucket regions are already contiguous)
    // if (d2h_bufs) {
    //     for (uint32_t d = 0; d < my_nd; d++) {
    //         const uint32_t gb  = b0 + d;
    //         const uint32_t off = bucket_offsets[gb];
    //         const uint32_t cnt = bucket_counts[gb];
    //         for (uint32_t k = 0; k < cnt; k++) {
    //             bucketed[off + k] = d2h_bufs[d][k].v;
    //         }
    //     }
    // }
    if (d2h_bufs) {
        for (uint32_t d = 0; d < my_nd; d++) {
            const uint32_t gb  = b0 + d;
            const uint32_t off = bucket_offsets[gb];
            const uint32_t cnt = bucket_counts[gb];

            if (!is_sorted_elemv((const elem_t *)d2h_bufs[d], cnt)) {
                // Find first inversion for debugging
                uint32_t bad_i = 0;
                for (bad_i = 1; bad_i < cnt; bad_i++) {
                    if (d2h_bufs[d][bad_i - 1].v > d2h_bufs[d][bad_i].v) {
                        break;
                    }
                }

                fprintf(stderr,
                        "[UNSORTED] set=%u bucket=%u off=%u cnt=%u at i=%u: %u > %u\n",
                        set_id, gb, off, cnt, bad_i,
                        d2h_bufs[d][bad_i - 1].v, d2h_bufs[d][bad_i].v);

                uint32_t w0 = (bad_i >= 4 ? bad_i - 4 : 0);
                uint32_t w1 = (bad_i + 4 < cnt ? bad_i + 4 : cnt - 1);
                fprintf(stderr, "  window [%u..%u]:", w0, w1);
                for (uint32_t z = w0; z <= w1; z++) {
                    fprintf(stderr, " %u", d2h_bufs[d][z].v);
                }
                fprintf(stderr, "\n");

                DPU_ASSERT(dpu_free(set));
                return 1;
            }

            // Copy back now that it's verified sorted
            for (uint32_t k = 0; k < cnt; k++) {
                bucketed[off + k] = d2h_bufs[d][k].v;
            }
        }
    }


    clock_gettime(CLOCK_MONOTONIC, &t_d2h_e);
    if (d2h_end_ns) {
        *d2h_end_ns = ts_to_ns(t_d2h_e);
    }

    // --- cleanup ---
    if (h2d_bufs) {
        for (uint32_t d = 0; d < my_nd; d++) {
            if (h2d_bufs[d]) free(h2d_bufs[d]);
        }
        free(h2d_bufs);
    }
    if (d2h_bufs) {
        for (uint32_t d = 0; d < my_nd; d++) {
            if (d2h_bufs[d]) free(d2h_bufs[d]);
        }
        free(d2h_bufs);
    }
    if (h2d_stats) free(h2d_stats);
    if (stats_out) free(stats_out);

    DPU_ASSERT(dpu_free(set));

    // Per-set timing summary
    double ms_alloc = (double)ns_diff(t_alloc_s, t_alloc_e) / 1e6;
    double ms_load  = (double)ns_diff(t_load_s,  t_load_e)  / 1e6;
    double ms_h2d   = (double)ns_diff(t_h2d_s,   t_h2d_e)   / 1e6;
    double ms_exec  = (double)ns_diff(t_exec_s,  t_exec_e)  / 1e6;
    double ms_d2h   = (double)ns_diff(t_d2h_s,   t_d2h_e)   / 1e6;

    printf("[Set %u] ms alloc=%.3f, load=%.3f, H2D=%.3f, exec=%.3f, D2H=%.3f\n",
           set_id, ms_alloc, ms_load, ms_h2d, ms_exec, ms_d2h);

    return 0;
}


int main(int argc, char **argv) {

    if (TOTAL_RANKS % NUM_SETS != 0) {
        fprintf(stderr, "%d sets is not a factor of %d ranks!\n\n", NUM_SETS, TOTAL_RANKS);
        return 1;
    }

    // Discover total DPUs available in the system (independent of set ordering)
    uint32_t total_dpus = 0;
    for (int s = 0; s < NUM_SETS; s++) {
        struct dpu_set_t tmp;
        DPU_ASSERT(dpu_alloc_ranks(TOTAL_RANKS / NUM_SETS, NULL, &tmp));
        uint32_t n = 0;
        DPU_ASSERT(dpu_get_nr_dpus(tmp, &n));
        total_dpus += n;
        DPU_ASSERT(dpu_free(tmp));
    }
    printf("Total DPUs available: %u\n", total_dpus);

    // Total number of elems 
    uint32_t N = 16384 * total_dpus;

    printf("Sorting N=%u across %u DPUs (NR_TASKLETS=%u), NUM_SETS=%d, TOTAL_RANKS=%d\n", N, total_dpus, (unsigned)NR_TASKLETS, NUM_SETS, TOTAL_RANKS);

    // Create the random set of numbers to be sorted
    uint32_t *input = (uint32_t *)malloc(sizeof(uint32_t) * N);
    if (!input) { 
        fprintf(stderr, "OOM\n"); 
        return 1; 
    }

    srand(0xC0FFEEu);
    for (uint32_t i = 0; i < N; i++) input[i] = (uint32_t)rand();

    // Timer for entire program, one for including overhead
    struct timespec t_all_s, t_all_e;

    struct timespec t_correction_s, t_correction_e;


    int set_errors = 0;

    uint64_t global_alloc_min_ns = UINT64_MAX;
    uint64_t global_d2h_max_ns   = 0;

    // ----- START BUCKET -----
    
    clock_gettime(CLOCK_MONOTONIC, &t_all_s);
    clock_gettime(CLOCK_MONOTONIC, &t_correction_s);

    // Global min/max
    uint32_t min_val = UINT32_MAX;
    uint32_t max_val = 0;

    #pragma omp parallel for num_threads(NUM_SETS) reduction(min:min_val) reduction(max:max_val) schedule(static)
    for (uint32_t i = 0; i < N; i++) {
        uint32_t v = input[i];
        if (v < min_val) {
            min_val = v;
        }
        if (v > max_val) {
            max_val = v;
        }
    }

    uint64_t range = (uint64_t)max_val - (uint64_t)min_val + 1ull;
    if (range == 0ull) {
        range = 1ull;
    }

    // 2) Choose many coarse bins (keeps per-bin counts small)
    const uint32_t CAP = MAX_ELEMS_PER_DPU;
    const uint32_t M   = (total_dpus == 0 ? 1u : total_dpus);

    uint32_t B = 4u * M;
    if (B < 65536u) {
        B = 65536u;
    }
    if (B > 262144u) {
        B = 262144u;
    }

    double inv_range_bins = (double)B / (double)range;

    // 3) Per-thread local histograms: local_bins[T][B]
    const uint32_t T = (uint32_t)NUM_SETS;
    size_t bins_row_bytes = (size_t)B * sizeof(uint32_t);

    uint32_t **local_bins = (uint32_t **)malloc((size_t)T * sizeof(uint32_t *));
    if (local_bins == NULL) {
        fprintf(stderr, "OOM: local_bins ptrs\n");
        free(input);
        return 1;
    }

    uint32_t *lc_storage = (uint32_t *)calloc((size_t)T * (size_t)B, sizeof(uint32_t));
    if (lc_storage == NULL) {
        fprintf(stderr, "OOM: lc_storage\n");
        free(local_bins);
        free(input);
        return 1;
    }

    for (uint32_t t = 0; t < T; t++) {
        local_bins[t] = lc_storage + (size_t)t * (size_t)B;
    }

    // 4) Build local histograms 
    #pragma omp parallel num_threads(NUM_SETS)
    {
        const uint32_t t_id = (uint32_t)omp_get_thread_num();
        uint32_t s = 0;
        uint32_t c = 0;

        shard_even(N, T, t_id, &s, &c);

        uint32_t *lb = local_bins[t_id];

        #pragma omp simd
        for (uint32_t i = 0; i < c; i++) {
            uint32_t v = input[s + i];
            uint32_t b = fast_bucket_fp(v, min_val, B, inv_range_bins);
            lb[b] += 1u;
        }
    }

    // 5) Reduce to global bin_counts and compute bin_offsets 
    uint32_t *bin_counts  = (uint32_t *)calloc((size_t)B, sizeof(uint32_t));
    uint32_t *bin_offsets = (uint32_t *)malloc((size_t)B * sizeof(uint32_t));
    if (bin_counts == NULL || bin_offsets == NULL) {
        fprintf(stderr, "OOM: bin arrays\n");
        free(bin_offsets);
        free(bin_counts);
        free(lc_storage);
        free(local_bins);
        free(input);
        return 1;
    }

    for (uint32_t b = 0; b < B; b++) {
        uint32_t sum = 0u;
        for (uint32_t t = 0; t < T; t++) {
            sum += local_bins[t][b];
        }
        bin_counts[b] = sum;
    }

    uint32_t total = 0u;
    for (uint32_t b = 0; b < B; b++) {
        bin_offsets[b] = total;
        total += bin_counts[b];
    }

    if (total != N) {
        fprintf(stderr, "[warn] histogram total %u != N %u\n", total, N);
    }

    // 6) Per-thread per-bin starting offsets 
    uint32_t **thread_prefix_bins = (uint32_t **)malloc((size_t)T * sizeof(uint32_t *));
    if (thread_prefix_bins == NULL) {
        fprintf(stderr, "OOM: thread_prefix_bins\n");
        free(bin_offsets);
        free(bin_counts);
        free(lc_storage);
        free(local_bins);
        free(input);
        return 1;
    }

    for (uint32_t t = 0; t < T; t++) {
        thread_prefix_bins[t] = (uint32_t *)malloc(bins_row_bytes);
        if (thread_prefix_bins[t] == NULL) {
            fprintf(stderr, "OOM: thread_prefix_bins[%u]\n", t);
            for (uint32_t u = 0; u < t; u++) {
                free(thread_prefix_bins[u]);
            }
            free(thread_prefix_bins);
            free(bin_offsets);
            free(bin_counts);
            free(lc_storage);
            free(local_bins);
            free(input);
            return 1;
        }
    }

    for (uint32_t b = 0; b < B; b++) {
        uint32_t run = bin_offsets[b];
        for (uint32_t t = 0; t < T; t++) {
            thread_prefix_bins[t][b] = run;
            run += local_bins[t][b];
        }
    }

    // 7) Pack by bin into `bucketed` (stable within-bucket order) 
    uint32_t *bucketed = (uint32_t *)malloc((size_t)N * sizeof(uint32_t));
    if (bucketed == NULL) {
        fprintf(stderr, "OOM: bucketed\n");
        for (uint32_t t = 0; t < T; t++) {
            free(thread_prefix_bins[t]);
        }
        free(thread_prefix_bins);
        free(bin_offsets);
        free(bin_counts);
        free(lc_storage);
        free(local_bins);
        free(input);
        return 1;
    }

    #pragma omp parallel num_threads(NUM_SETS)
    {
        const uint32_t t_id = (uint32_t)omp_get_thread_num();
        uint32_t s = 0;
        uint32_t c = 0;

        shard_even(N, T, t_id, &s, &c);

        uint32_t *cursor = (uint32_t *)malloc(bins_row_bytes);
        if (cursor == NULL) {
            cursor = thread_prefix_bins[t_id];
        } else {
            memcpy(cursor, thread_prefix_bins[t_id], bins_row_bytes);
        }

        for (uint32_t i = 0; i < c; i++) {
            uint32_t v = input[s + i];
            uint32_t b = fast_bucket_fp(v, min_val, B, inv_range_bins);
            uint32_t pos = cursor[b]++;
            bucketed[pos] = v;
        }

        if (cursor != thread_prefix_bins[t_id]) {
            free(cursor);
        }
    }

    // 8) Group consecutive bins into final buckets with count ≤ CAP
    //    Use growable arrays to avoid overruns that corrupt the heap. 
    uint32_t cap_buckets = (uint32_t)((N + CAP - 1u) / CAP) + 8u;
    if (cap_buckets < 1024u) {
        cap_buckets = 1024u;
    }

    uint32_t *final_counts  = (uint32_t *)malloc((size_t)cap_buckets * sizeof(uint32_t));
    uint32_t *final_offsets = (uint32_t *)malloc((size_t)cap_buckets * sizeof(uint32_t));
    if (final_counts == NULL || final_offsets == NULL) {
        fprintf(stderr, "OOM: final bucket arrays\n");
        free(final_offsets);
        free(final_counts);
        for (uint32_t t = 0; t < T; t++) {
            free(thread_prefix_bins[t]);
        }
        free(thread_prefix_bins);
        free(bin_offsets);
        free(bin_counts);
        free(lc_storage);
        free(local_bins);
        free(bucketed);
        free(input);
        return 1;
    }

    #define ENSURE_BUCKET_CAPACITY()                                                     \
        do {                                                                             \
            if (num_buckets >= cap_buckets) {                                            \
                uint32_t new_cap = cap_buckets * 2u;                                     \
                uint32_t *nc = (uint32_t *)realloc(final_counts,  (size_t)new_cap * sizeof(uint32_t)); \
                uint32_t *no = (uint32_t *)realloc(final_offsets, (size_t)new_cap * sizeof(uint32_t)); \
                if (nc == NULL || no == NULL) {                                          \
                    fprintf(stderr, "OOM: growing bucket arrays (%u -> %u)\n", cap_buckets, new_cap); \
                    free(no);                                                            \
                    free(nc);                                                            \
                    for (uint32_t t = 0; t < T; t++) {                                   \
                        free(thread_prefix_bins[t]);                                      \
                    }                                                                     \
                    free(thread_prefix_bins);                                            \
                    free(bin_offsets);                                                   \
                    free(bin_counts);                                                    \
                    free(lc_storage);                                                    \
                    free(local_bins);                                                    \
                    free(final_offsets);                                                 \
                    free(final_counts);                                                  \
                    free(bucketed);                                                      \
                    free(input);                                                         \
                    return 1;                                                            \
                }                                                                         \
                final_counts  = nc;                                                       \
                final_offsets = no;                                                       \
                cap_buckets   = new_cap;                                                  \
            }                                                                             \
        } while (0)

    uint32_t num_buckets = 0u;
    uint32_t acc = 0u;
    uint32_t cur_start = (B > 0u) ? bin_offsets[0] : 0u;

    for (uint32_t b = 0; b < B; b++) {
        uint32_t cnt = bin_counts[b];

        if (cnt > CAP) {
            fprintf(stderr, "[error] Bin %u has %u > CAP=%u. Increase B.\n", b, cnt, CAP);

            for (uint32_t t = 0; t < T; t++) {
                free(thread_prefix_bins[t]);
            }
            free(thread_prefix_bins);
            free(bin_offsets);
            free(bin_counts);
            free(lc_storage);
            free(local_bins);
            free(final_offsets);
            free(final_counts);
            free(bucketed);
            free(input);
            return 1;
        }

        if (acc > 0u) {
            size_t tentative = (size_t)acc + (size_t)cnt;
            if (tentative > (size_t)CAP) {
                ENSURE_BUCKET_CAPACITY();
                final_offsets[num_buckets] = cur_start;
                final_counts[num_buckets]  = acc;
                num_buckets += 1u;

                cur_start = bin_offsets[b];
                acc = 0u;
            }
        }

        acc += cnt;
    }

    if (acc > 0u || (B == 0u && N == 0u)) {
        ENSURE_BUCKET_CAPACITY();
        final_offsets[num_buckets] = cur_start;
        final_counts[num_buckets]  = acc;
        num_buckets += 1u;
    }

    // Sanity: bucket layout must perfectly partition [0..N) 
    if (check_bucket_layout(final_offsets, final_counts, num_buckets, N, 1) != 0) {
        fprintf(stderr, "[fatal] invalid bucket layout. Aborting.\n");
        for (uint32_t t = 0; t < T; t++) free(thread_prefix_bins[t]);
        free(thread_prefix_bins);
        free(bin_offsets); free(bin_counts);
        free(lc_storage);  free(local_bins);
        free(final_offsets); free(final_counts);
        free(bucketed); free(input);
        return 1;
    }


    // Tidy bin temp memory before DPU work 
    for (uint32_t t = 0; t < T; t++) {
        free(thread_prefix_bins[t]);
    }
    free(thread_prefix_bins);
    free(bin_offsets);
    free(bin_counts);
    free(lc_storage);
    free(local_bins);

    clock_gettime(CLOCK_MONOTONIC, &t_correction_e);

    // 1Global bucket cursor across sets 
    uint32_t next_bucket = 0;

    // Launch bucketed DPUs; each bucket ≤ CAP by construction 
    #pragma omp parallel for reduction(+:set_errors) schedule(static)
    for (int s = 0; s < NUM_SETS; s++) {
        uint64_t alloc_ns = 0;
        uint64_t d2h_ns   = 0;

        int rc = dpu_sort_bucketed((uint32_t)s,
                                bucketed,
                                final_offsets,
                                final_counts,
                                num_buckets,
                                &next_bucket,
                                &alloc_ns,
                                &d2h_ns);
        if (rc != 0) {
            set_errors += 1;
        }

        #pragma omp critical
        {
            if (alloc_ns < global_alloc_min_ns) {
                global_alloc_min_ns = alloc_ns;
            }
            if (d2h_ns > global_d2h_max_ns) {
                global_d2h_max_ns = d2h_ns;
            }
        }
    }

    // Cross-bucket boundary check on host 
    int xbad = 0;
    for (uint32_t i = 1; i < num_buckets; i++) {
        uint32_t prev_off = final_offsets[i-1];
        uint32_t prev_cnt = final_counts[i-1];
        uint32_t this_off = final_offsets[i];
        uint32_t this_cnt = final_counts[i];

        if (prev_cnt == 0u || this_cnt == 0u) {
            fprintf(stderr, "[sanity] empty bucket at boundary i=%u\n", i);
            xbad = 1;
            break;
        }

        uint32_t prev_last = bucketed[prev_off + prev_cnt - 1];
        uint32_t this_first = bucketed[this_off];

        if (prev_last > this_first) {
            fprintf(stderr, "[CROSS] boundary inversion between buckets %u and %u: %u > %u\n",
                    i - 1, i, prev_last, this_first);
            xbad = 1;
            break;
        }
    }

    if (xbad) {
        fprintf(stderr, "[fatal] cross-bucket ordering failed; not copying back to input.\n");
        free(final_counts);
        free(final_offsets);
        free(bucketed);
        free(input);
        return 1;
    }


    // Copy back & free 
    memcpy(input, bucketed, (size_t)N * sizeof(uint32_t));

    // sanity: sum of final_counts equals N 
    uint64_t check_total = 0ull;
    for (uint32_t i = 0; i < num_buckets; i++) {
        check_total += (uint64_t)final_counts[i];
    }
    if (check_total != (uint64_t)N) {
        fprintf(stderr, "[sanity] bucketed total %" PRIu64 " != N %u\n", check_total, N);
    }

    free(final_counts);
    free(final_offsets);
    free(bucketed);
    
    // ----- END BUCKET -----



    // START MERGE
    /*
    clock_gettime(CLOCK_MONOTONIC, &t_all_s);
    clock_gettime(CLOCK_MONOTONIC, &t_correction_s);
    #pragma omp parallel for reduction(+:set_errors) schedule(static)
    for (int s = 0; s < NUM_SETS; s++) {
        uint32_t set_start = 0, set_count = 0;
        shard_even(N, NUM_SETS, (uint32_t)s, &set_start, &set_count);

        uint32_t *set_base = &input[set_start];

        uint64_t alloc_ns = 0, d2h_ns = 0;
        int rc = dpu_sort((uint32_t)s, set_base, set_count, &alloc_ns, &d2h_ns);
        if (rc != 0) {
            set_errors += 1;
        }

        #pragma omp critical
        {
            if (alloc_ns < global_alloc_min_ns) {
                global_alloc_min_ns = alloc_ns;
            }
            if (d2h_ns > global_d2h_max_ns) {
                global_d2h_max_ns = d2h_ns;
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t_correction_e);
    */
    // END MERGE

    clock_gettime(CLOCK_MONOTONIC, &t_all_e);


    if (set_errors > 0) {
        fprintf(stderr, "One or more sets failed (%d errors).\n", set_errors);
        free(input);
        return 1;
    }

    int bad = 0;
    for (int s = 0; s < NUM_SETS; s++) {
        uint32_t set_start = 0;
        uint32_t set_count = 0;

        shard_even(N, NUM_SETS, (uint32_t)s, &set_start, &set_count);

        uint32_t j = 1;
        for (j = 1; j < set_count; j++) {
            uint32_t a = input[set_start + j - 1];
            uint32_t b = input[set_start + j];

            if (a > b) {
                bad = 1;
                break;
            }
        }
    }

    double ms_total = (double)ns_diff(t_all_s, t_all_e) / 1e6;
    double ms_correction = (double)ns_diff(t_correction_s, t_correction_e) / 1e6;

    printf("\n--- Host Summary ---\n");
    printf("Total wall time (ms): %.3f\n", ms_total);
    printf("Bucket/Merge time (ms): %.3f\n", ms_correction);
    printf("Per-set execution ran %s.\n", (set_errors == 0) ? "successfully" : "with errors");
    printf("Intra-set sorted check: %s\n", (bad == 0) ? "OK" : "FAILED");

    if (global_d2h_max_ns >= global_alloc_min_ns) {
        uint64_t window_ns = global_d2h_max_ns - global_alloc_min_ns;
        printf("[Global] DPU host window (alloc → last D2H): %.3f ms\n", window_ns / 1e6);
    } else {
        // very unlikely unless clocks moved backwards
        printf("[Global] DPU host window unavailable (timestamps out of order)\n");
    }


    free(input);
    return (bad || set_errors) ? 1 : 0;
}
