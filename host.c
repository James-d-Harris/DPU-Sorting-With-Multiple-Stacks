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
#define NUM_SETS 2

#define TOTAL_RANKS 40


struct dpu_stats {
    uint32_t n_elems;
    uint32_t nr_tasklets;
    uint32_t cycles_sort;
    uint32_t cycles_total;
};

static inline uint64_t ns_diff(const struct timespec a, const struct timespec b) {
    return (uint64_t)(b.tv_sec - a.tv_sec) * 1000000000ull
            + (uint64_t)(b.tv_nsec - a.tv_nsec);
}


int dpu_sort(uint32_t *input, uint32_t num_elems, uint32_t set_num) {
    struct timespec t_h2d_s, t_h2d_e, t_exec_s, t_exec_e, t_d2h_s, t_d2h_e;
    clock_gettime(CLOCK_MONOTONIC, &t_exec_s);
    struct dpu_set_t set, dpu;

    dpu_error_t err = dpu_alloc_ranks(TOTAL_RANKS / NUM_SETS, NULL, &set);
    if(err != DPU_OK) {
        fprintf(stderr, "alloc_ranks failed: %s\n", dpu_error_to_string(err));
        return 1;
    }

    uint32_t num_dpu = 0;
    DPU_ASSERT(dpu_get_nr_dpus(set, &num_dpu));
    if ( num_dpu == 0 ) {
        fprintf(stderr, "No DPUs allocated.\n");
        return 1;
    }
    printf("%u DPUs allocated for DPU set %u.\n", num_dpu, set_num);

    DPU_ASSERT(dpu_load(set, DPU_EXE, NULL));

    clock_gettime(CLOCK_MONOTONIC, &t_h2d_s);

    uint32_t idx = 0;
    DPU_FOREACH(set, dpu) {
        if (idx >= num_dpu) break;

        idx++;
    }
    clock_gettime(CLOCK_MONOTONIC, &t_h2d_e);



    clock_gettime(CLOCK_MONOTONIC, &t_d2h_s);

    idx = 0;
    DPU_FOREACH(set, dpu) {
        if (idx >= num_dpu) break;

        idx++;
    }
    clock_gettime(CLOCK_MONOTONIC, &t_d2h_e);

    clock_gettime(CLOCK_MONOTONIC, &t_exec_e);

    return 0;
}

int main(int argc, char **argv) {

    if (TOTAL_RANKS % NUM_SETS != 0) {
        fprintf(stderr, "%d sets is not a factor of %d ranks!\n\n", NUM_SETS, TOTAL_RANKS);
        return 1;
    }

    // Total number of elems 
    uint32_t N = (argc >= 2) ? (uint32_t)strtoul(argv[1], NULL, 10) : (1u << 20);
    // number of DPUs to be used. Not really utilized outside of shards
    uint32_t NB_DPUS = NR_DPUS;

    printf("Sorting N=%u across %u DPUs (NR_TASKLETS=%u)\n", N, NB_DPUS, (unsigned)NR_TASKLETS);

    // Create the random set of numbers to be sorted
    uint32_t *input = (uint32_t *)malloc(sizeof(uint32_t) * N);
    if (!input) { fprintf(stderr, "OOM\n"); return 1; }
    srand(0xC0FFEEu);
    for (uint32_t i = 0; i < N; i++) input[i] = (uint32_t)rand();

    // Timer for entire program, one for including overhead
    struct timespec t_after_rand, t_completed, t_all_sets_s, t_all_sets_e;
    clock_gettime(CLOCK_MONOTONIC, &t_after_rand);

    // ----- Shard across DPUs -----
    uint32_t *shard_starts = (uint32_t *)malloc(sizeof(uint32_t) * NB_DPUS);
    uint32_t *shard_counts = (uint32_t *)malloc(sizeof(uint32_t) * NB_DPUS);
    if (!shard_starts || !shard_counts) { fprintf(stderr, "OOM\n"); return 1; }

    uint32_t base = 0;
    for (uint32_t i = 0; i < NB_DPUS; i++) {
        uint32_t cnt = N / NB_DPUS + (i < (N % NB_DPUS) ? 1 : 0);
        shard_starts[i] = base;
        shard_counts[i] = cnt;
        base += cnt;
        if (cnt > MAX_ELEMS_PER_DPU) {
            fprintf(stderr, "Shard %u too large for MRAM (%u > %u)\n", i, cnt, MAX_ELEMS_PER_DPU);
            return 1;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_all_sets_s);

    #pragma omp parallel for 
    for (uint32_t i = 0; i < NUM_SETS; i++) {
        // Split input depending on set size
        uint32_t offset = i * (N / NUM_SETS);
        int err = dpu_sort(&input[offset], offset, i);

        if (err > 0) {
            printf("SET #%u encounterd and error", i);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_all_sets_e);



    clock_gettime(CLOCK_MONOTONIC, &t_completed);

    uint64_t ns_total_time = ns_diff(t_after_rand, t_completed);
    uint64_t ns_all_sets = ns_diff(t_all_sets_s, t_all_sets_e);


    printf("\n--- DPU Timing (ms) ---\n");
    printf("Total Execute Time.  : %.3f\n", ns_total_time / 1e6);
    printf("Longest Set Time.    : %.3f\n", ns_all_sets / 1e6);


    free(input);

    free(shard_starts);
    free(shard_counts);
}