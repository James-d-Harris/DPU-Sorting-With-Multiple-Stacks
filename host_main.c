#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <time.h>
#include <string.h>
#include <omp.h>

#include "bucketing.h"
#include "dpu_exec.h"

// ---------------- Build-time knobs ----------------
#ifndef DPU_EXE
#define DPU_EXE "./quicksort_dpu"
#endif

// UPDATE IN MAKEFILE
#ifndef MAX_ELEMS_PER_DPU
#define MAX_ELEMS_PER_DPU 8192000u
#endif

// UPDATE IN MAKEFILE
#ifndef TOTAL_RANKS
#define TOTAL_RANKS 40u
#endif

// UPDATE IN MAKEFILE
#ifndef NUM_SETS
#define NUM_SETS 1u
#endif

// How many elements to sort per dpu
// #define ELEMS_PER_DPU 800000u

static const char *format_with_commas(uint64_t n) {
    static char buf[32];
    char tmp[32];
    snprintf(tmp, sizeof(tmp), "%llu", (unsigned long long)n);

    int len = strlen(tmp);
    int commas = (len - 1) / 3;
    int out_len = len + commas;

    buf[out_len] = '\0';
    int j = out_len - 1;

    for (int i = len - 1, count = 0; i >= 0; i--) {
        buf[j--] = tmp[i];
        count++;
        if (count == 3 && i > 0) {
            buf[j--] = ',';
            count = 0;
        }
    }
    return buf;
}

static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// xorshift64* generator (fast, decent quality for benchmarking)
static inline uint64_t xorshift64s(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

int main(void)
{
    // Make stdout/stderr unbuffered so we see logs immediately even if it hangs.
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    fprintf(stderr, "[host] entered main\n");

    fprintf(stderr, "[host] after now_ms; preparing DPUs...\n");


    // Prepare DPUs and program (done entirely in dpu_exec)
    dpu_exec_ctx_t *ctx = NULL;
    uint32_t nb_dpus = 0;
    double ms_alloc = 0.0, ms_load = 0.0;

    fprintf(stderr, "[host] calling dpu_prepare_sets...\n");
    int pret = dpu_prepare_sets(TOTAL_RANKS,
                                NUM_SETS,
                                DPU_EXE,
                                &ctx,
                                &nb_dpus,
                                &ms_alloc,
                                &ms_load);
    if (pret != 0) {
        fprintf(stderr, "dpu_prepare_sets failed (%d)\n", pret);
        return 1;
    }
    fprintf(stderr, "[host] dpu_prepare_sets returned %d\n", pret);

    const uint64_t N = (uint64_t)nb_dpus * (uint64_t)ELEMS_PER_DPU;

    fprintf(stderr, "[host] sorting %s elements\n", format_with_commas(N));

    uint32_t *h_input = (uint32_t *)malloc((size_t)N * sizeof(uint32_t));
    if (!h_input) {
        fprintf(stderr, "OOM for input N=%" PRIu64 "\n", N);
        dpu_release(ctx);
        return 1;
    }

    // fill input
    uint64_t base_seed = (uint64_t)time(NULL);   // use a fixed constant for reproducible runs
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint64_t state = splitmix64(base_seed ^ (uint64_t)tid);

        #pragma omp for schedule(static)
        for (uint64_t i = 0; i < N; i++) {
            uint64_t r = xorshift64s(&state);
            h_input[i] = (uint32_t)r;
        }
    }


    double t0_total = now_ms();

    // how many DPUs total (opaque getter — no direct field access)
    uint32_t total_dpus = 0;
    if (dpu_total_dpus(ctx, &total_dpus) != 0) {
        fprintf(stderr, "dpu_total_dpus failed\n");
        free(h_input);
        dpu_release(ctx);
        return 1;
    }

    // build buckets to match (or <=) available DPUs
    uint32_t *bucketed      = NULL;
    uint64_t *final_offsets = NULL;
    uint32_t *final_counts  = NULL;
    uint32_t  num_buckets   = 0;

    printf("Starting bucketing\n");
    double t_bkt0 = now_ms();
    int bret2 = bucketing_build(
        h_input, N, total_dpus, MAX_ELEMS_PER_DPU,
        &bucketed, &final_offsets, &final_counts, &num_buckets
    );
    double t_bkt1 = now_ms();
    double ms_bucket = t_bkt1 - t_bkt0;
    printf("Done bucketing\n");

    if (bret2 != 0) {
        fprintf(stderr, "bucketing_build failed (%d)\n", bret2);
        free(h_input);
        dpu_release(ctx);
        return 2;
    }

    // // Verifies the buckets were set up properly
    // // Unneeded sanity check
    // int pre_ok = verify_buckets_host_pre(bucketed, final_offsets, final_counts, num_buckets);
    // printf("[Pre] host bucket order: %s\n", pre_ok == 0 ? "OK" : "FAILED");

    // run DPUs with deterministic mapping (bucket i -> global DPU position i)
    double ms_h2d = 0.0, ms_exec = 0.0, ms_d2h = 0.0;

    printf("Starting dpu runs\n");
    int rret = dpu_run_and_collect_parallel_bucketed(
        ctx,
        NUM_SETS,
        bucketed,         // DPU writes back sorted buckets in place
        N,
        final_offsets,
        final_counts,
        num_buckets,      // can be <= total_dpus (extras idle)
        &ms_h2d, &ms_exec, &ms_d2h
    );
    if (rret != 0) {
        fprintf(stderr, "dpu_run_and_collect_parallel_bucketed failed (%d)\n", rret);
        free(final_counts); free(final_offsets); free(bucketed); free(h_input);
        dpu_release(ctx);
        return 3;
    }

    printf("\nNumber of Elements: %s", format_with_commas(N));

    // --- timings printout
    printf("\n--- Host Timings (ms) ---\n");
    printf("DPU alloc (ranks):    %s ms\n", format_with_commas((uint64_t)ms_alloc));
    printf("Bucketing/packing:    %s ms\n", format_with_commas((uint64_t)ms_bucket));
    printf("Host→DPU span:        %s ms\n", format_with_commas((uint64_t)ms_h2d));
    printf("Execute span:         %s ms\n", format_with_commas((uint64_t)ms_exec));
    printf("DPU→Host span:        %s ms\n", format_with_commas((uint64_t)ms_d2h));
    printf("TOTAL wall time:      %s ms\n", format_with_commas((uint64_t)(now_ms() - t0_total)));



    // verification in key order (buckets are in global key order)
    int v = verify_across_buckets(bucketed, final_offsets, final_counts, num_buckets);
    printf("[Verify] buckets (key order): %s\n", (v == 0) ? "OK" : "FAILED");

    // cleanup
    free(final_counts);
    free(final_offsets);
    free(bucketed);
    free(h_input);
    dpu_release(ctx);
    return (rret == 0 && v == 0) ? 0 : 2;

}
