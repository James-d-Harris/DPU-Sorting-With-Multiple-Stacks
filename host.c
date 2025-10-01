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
#define MAX_ELEMS_PER_DPU 16000
#endif

#ifndef DPU_EXE
#define DPU_EXE "./quicksort_dpu"
#endif
// Must be a factor of 40, with a maximum of 40 (# ranks)
#ifndef NUM_SETS
#define NUM_SETS 2

#endif

#ifndef TOTAL_RANKS
#define TOTAL_RANKS 40
#endif

#ifndef MRAM_SYM
#define MRAM_SYM "MRAM_BASE"
#endif

#define SYM_MRAM_ARR "MRAM_ARR"
#define SYM_STATS "STATS"

struct dpu_stats {
    uint32_t n_elems;
    uint32_t nr_tasklets;
    uint32_t cycles_sort;
    uint32_t cycles_total;
};

typedef struct __attribute__((packed, aligned(8))) {
    uint32_t v;
    uint32_t pad;
} elem_t;

static inline uint64_t ns_diff(const struct timespec a, const struct timespec b) {
    return (uint64_t)(b.tv_sec - a.tv_sec) * 1000000000ull
            + (uint64_t)(b.tv_nsec - a.tv_nsec);
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
    for (uint32_t i = 0; i < nb_dpus; i++) {
        uint32_t s = 0, c = 0;
        shard_even(N_set, nb_dpus, i, &s, &c);
        starts[i] = s;
        counts[i] = c;
    }
}

int dpu_sort(uint32_t set_id, uint32_t *base_ptr, uint32_t N_set) {
    struct timespec t_alloc_s, t_alloc_e, t_load_s, t_load_e, t_h2d_s, t_h2d_e;
    struct timespec t_exec_s, t_exec_e, t_d2h_s, t_d2h_e;

    clock_gettime(CLOCK_MONOTONIC, &t_alloc_s);
    struct dpu_set_t set;
    dpu_error_t derr = dpu_alloc_ranks(TOTAL_RANKS / NUM_SETS, NULL, &set);
    if (derr != DPU_OK) {
        fprintf(stderr, "[Set %u] alloc_ranks failed: %s\n", set_id, dpu_error_to_string(derr));
        return 1;
    }

    uint32_t nb_dpus = 0;
    DPU_ASSERT(dpu_get_nr_dpus(set, &nb_dpus));
    if (nb_dpus == 0) {
        fprintf(stderr, "No DPUs allocated.\n");
        return 1;
    }

    printf("%u DPUs allocated for DPU set %u.\n", nb_dpus, set_id);
    clock_gettime(CLOCK_MONOTONIC, &t_alloc_e);

    clock_gettime(CLOCK_MONOTONIC, &t_load_s);
    DPU_ASSERT(dpu_load(set, DPU_EXE, NULL));
    clock_gettime(CLOCK_MONOTONIC, &t_load_e);

    uint32_t *starts = malloc(nb_dpus * sizeof(uint32_t));
    uint32_t *counts = malloc(nb_dpus * sizeof(uint32_t));
    if (!starts || !counts) {
        fprintf(stderr, "[Set %u] OOM allocating shard arrays.\n", set_id);
        free(starts); free(counts);
        DPU_ASSERT(dpu_free(set));
        return 1;
    }

    fill_dpu_shards(N_set, nb_dpus, starts, counts);

    for (uint32_t i = 0; i < nb_dpus; i++) {
        if (counts[i] > MAX_ELEMS_PER_DPU) {
            fprintf(stderr, "[Set %u] Shard %u too large for MRAM (%u > %u)\n", set_id, i, counts[i], MAX_ELEMS_PER_DPU);
            free(starts); free(counts);
            DPU_ASSERT(dpu_free(set));
            return 1;
        }
    }

    uint32_t max_cnt = 0;
    for (uint32_t i = 0; i < nb_dpus; i++) {
        if (counts[i] > max_cnt) {
            max_cnt = counts[i];
        }
    }
    size_t max_bytes = (size_t)max_cnt * sizeof(elem_t);

    elem_t **h2d_bufs = calloc(nb_dpus, sizeof(elem_t *));
    struct dpu_stats *h2d_stats = calloc(nb_dpus, sizeof(struct dpu_stats));
    if (!h2d_bufs || !h2d_stats) {
        fprintf(stderr, "[Set %u] OOM: staging arrays.\n", set_id);
        free(starts); free(counts);
        free(h2d_bufs); free(h2d_stats);
        DPU_ASSERT(dpu_free(set));
        return 1;
    }

    uint32_t d = 0;
    for (; d < nb_dpus; d++) {
        if (max_bytes > 0) {
            h2d_bufs[d] = aligned_alloc(8, max_bytes);
            if (!h2d_bufs[d]) {
                fprintf(stderr, "[Set %u] OOM: h2d_bufs[%u]\n", set_id, d);
                break;
            }

            for (uint32_t k = 0; k < counts[d]; k++) {
                h2d_bufs[d][k].v = base_ptr[starts[d] + k];
                h2d_bufs[d][k].pad = 0u;
            }
            for (uint32_t k = counts[d]; k < max_cnt; k++) {
                h2d_bufs[d][k].v = 0u;
                h2d_bufs[d][k].pad = 0u;
            }
        }

        h2d_stats[d].n_elems = counts[d];
        h2d_stats[d].nr_tasklets = NR_TASKLETS;
        h2d_stats[d].cycles_sort = 0u;
        h2d_stats[d].cycles_total = 0u;
    }

    if (d != nb_dpus && max_bytes > 0) {
        for (uint32_t t = 0; t < d; t++) {
            free(h2d_bufs[t]);
        }
        free(h2d_bufs); free(h2d_stats);
        free(starts); free(counts);
        DPU_ASSERT(dpu_free(set));
        return 1;
    }

    if (max_bytes > 0) {
        struct dpu_set_t dpu;
        uint32_t idx = 0;
        DPU_FOREACH(set, dpu) {
            if (idx >= nb_dpus) break;
            DPU_ASSERT(dpu_prepare_xfer(dpu, h2d_bufs[idx]));
            idx++;
        }
        DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, SYM_MRAM_ARR, 0, max_bytes, DPU_XFER_DEFAULT));
    }

    {
        struct dpu_set_t dpu;
        uint32_t idx = 0;
        DPU_FOREACH(set, dpu) {
            if (idx >= nb_dpus) break;
            DPU_ASSERT(dpu_prepare_xfer(dpu, &h2d_stats[idx]));
            idx++;
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
        d2h_bufs = calloc(nb_dpus, sizeof(elem_t *));
        if (!d2h_bufs) {
            fprintf(stderr, "[Set %u] OOM: d2h_bufs.\n", set_id);
        } else {
            for (d = 0; d < nb_dpus; d++) {
                d2h_bufs[d] = aligned_alloc(8, max_bytes);
                if (!d2h_bufs[d]) {
                    fprintf(stderr, "[Set %u] OOM: d2h_bufs[%u]\n", set_id, d);
                    break;
                }
            }
            if (d != nb_dpus) {
                for (uint32_t t = 0; t < d; t++) {
                    free(d2h_bufs[t]);
                }
                free(d2h_bufs);
                d2h_bufs = NULL;
            }
        }

        if (d2h_bufs) {
            struct dpu_set_t dpu;
            uint32_t idx = 0;
            DPU_FOREACH(set, dpu) {
                if (idx >= nb_dpus) break;
                DPU_ASSERT(dpu_prepare_xfer(dpu, d2h_bufs[idx]));
                idx++;
            }
            DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, SYM_MRAM_ARR, 0, max_bytes, DPU_XFER_DEFAULT));
        }
    }

    struct dpu_stats *stats_out = calloc(nb_dpus, sizeof(struct dpu_stats));
    if (stats_out) {
        struct dpu_set_t dpu;
        uint32_t idx = 0;
        DPU_FOREACH(set, dpu) {
            if (idx >= nb_dpus) break;
            DPU_ASSERT(dpu_prepare_xfer(dpu, &stats_out[idx]));
            idx++;
        }
        DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, SYM_STATS, 0, sizeof(struct dpu_stats), DPU_XFER_DEFAULT));
    }

    if (d2h_bufs) {
        for (d = 0; d < nb_dpus; d++) {
            for (uint32_t k = 0; k < counts[d]; k++) {
                base_ptr[starts[d] + k] = d2h_bufs[d][k].v;
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t_d2h_e);

    if (h2d_bufs) {
        for (d = 0; d < nb_dpus; d++) {
            free(h2d_bufs[d]);
        }
        free(h2d_bufs);
    }

    if (d2h_bufs) {
        for (d = 0; d < nb_dpus; d++) {
            free(d2h_bufs[d]);
        }
        free(d2h_bufs);
    }

    if (stats_out) {
        uint64_t min_c = UINT64_MAX, max_c = 0;
        double sum_c = 0.0;
        for (d = 0; d < nb_dpus; d++) {
            uint64_t c = stats_out[d].cycles_sort;
            if (c < min_c) min_c = c;
            if (c > max_c) max_c = c;
            sum_c += (double)c;
        }
        double avg_c = (nb_dpus > 0) ? (sum_c / nb_dpus) : 0.0;
        printf("[Set %u] cycles_sort min=%" PRIu64 " avg=%.1f max=%" PRIu64 "\n", set_id, min_c, avg_c, max_c);
        free(stats_out);
    }

    free(starts); free(counts); free(h2d_stats);
    DPU_ASSERT(dpu_free(set));

    double ms_alloc = ns_diff(t_alloc_s, t_alloc_e) / 1e6;
    double ms_load = ns_diff(t_load_s, t_load_e) / 1e6;
    double ms_h2d = ns_diff(t_h2d_s, t_h2d_e) / 1e6;
    double ms_exec = ns_diff(t_exec_s, t_exec_e) / 1e6;
    double ms_d2h = ns_diff(t_d2h_s, t_d2h_e) / 1e6;
    printf("[Set %u] ms alloc=%.3f, load=%.3f, H2D=%.3f, exec=%.3f, D2H=%.3f\n", set_id, ms_alloc, ms_load, ms_h2d, ms_exec, ms_d2h);

    return 0;
}

int main(int argc, char **argv) {

    if (TOTAL_RANKS % NUM_SETS != 0) {
        fprintf(stderr, "%d sets is not a factor of %d ranks!\n\n", NUM_SETS, TOTAL_RANKS);
        return 1;
    }

    // Total number of elems 
    uint32_t N = (argc >= 2) ? (uint32_t)strtoul(argv[1], NULL, 10) : (1u << 20);
    printf("Sorting N=%u across %u DPUs (NR_TASKLETS=%u), NUM_SETS=%d, TOTAL_RANKS=%d\n", N, NR_DPUS, NR_TASKLETS, NUM_SETS, TOTAL_RANKS);

    // Create the random set of numbers to be sorted
    uint32_t *input = (uint32_t *)malloc(sizeof(uint32_t) * N);
    if (!input) {
        fprintf(stderr, "OOM\n");
        return 1;
    }

    srand(0xC0FFEEu);
    for (uint32_t i = 0; i < N; i++) input[i] = (uint32_t)rand();

    struct timespec t_all_s, t_all_e;
    clock_gettime(CLOCK_MONOTONIC, &t_all_s);

    int set_errors = 0;
    #pragma omp parallel for reduction(+:set_errors) schedule(static)
    for (int s = 0; s < NUM_SETS; s++) {
        uint32_t set_start = 0, set_count = 0;
        shard_even(N, NUM_SETS, s, &set_start, &set_count);
        int rc = dpu_sort(s, &input[set_start], set_count);
        if (rc != 0) set_errors++;
    }

    clock_gettime(CLOCK_MONOTONIC, &t_all_e);

    if (set_errors > 0) {
        fprintf(stderr, "One or more sets failed (%d errors).\n", set_errors);
        free(input);
        return 1;
    }

    int bad = 0;
    for (int s = 0; s < NUM_SETS; s++) {
        uint32_t set_start = 0, set_count = 0;
        shard_even(N, NUM_SETS, s, &set_start, &set_count);
        for (uint32_t j = 1; j < set_count; j++) {
            if (input[set_start + j - 1] > input[set_start + j]) {
                bad = 1;
                break;
            }
        }
    }

    double ms_total = ns_diff(t_all_s, t_all_e) / 1e6;
    printf("\n--- Host Summary ---\n");
    printf("Total wall time (ms): %.3f\n", ms_total);
    printf("Per-set execution ran %s.\n", (set_errors == 0) ? "successfully" : "with errors");
    printf("Intra-set sorted check: %s\n", (bad == 0) ? "OK" : "FAILED");

    free(input);
    return (bad || set_errors) ? 1 : 0;
}