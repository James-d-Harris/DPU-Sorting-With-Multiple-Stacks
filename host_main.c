#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <time.h>
#include <string.h>

#include "bucketing.h"
#include "dpu_exec.h"

// ---------------- Build-time knobs----------------
#ifndef DPU_EXE
#define DPU_EXE "./quicksort_dpu"
#endif

// UPDATE IN MAKEFILE
#ifndef MAX_ELEMS_PER_DPU
#define MAX_ELEMS_PER_DPU 16384u
#endif

// UPDATE IN MAKEFILE
#ifndef TOTAL_RANKS
#define TOTAL_RANKS 40u
#endif

// UPDATE IN MAKEFILE
#ifndef NUM_SETS
#define NUM_SETS 1u
#endif

static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

int main(void)
{
    // Make stdout/stderr unbuffered so we see logs immediately even if it hangs.
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    fprintf(stderr, "[host] entered main\n");

    double t0_total = now_ms();
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

    const uint32_t N = nb_dpus * 16000;

    fprintf(stderr, "[host] sorting %u elements\n", N);

    uint32_t *h_input = (uint32_t *)malloc((size_t)N * sizeof(uint32_t));
    if (!h_input) {
        fprintf(stderr, "OOM for input N=%" PRIu32 "\n", N);
        dpu_release(ctx);
        return 1;
    }

    // fill input
    double t_gen0 = now_ms();
    for (uint32_t i = 0; i < N; i++) {
        uint32_t x = i * 2654435761u;
        h_input[i] = (x ^ (x >> 16)) + 12345u;
    }
    double t_gen1 = now_ms();

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
    uint32_t *final_offsets = NULL;
    uint32_t *final_counts  = NULL;
    uint32_t  num_buckets   = 0;

    double t_bkt0 = now_ms();
    int bret2 = bucketing_build(
        h_input, N, total_dpus, MAX_ELEMS_PER_DPU,
        &bucketed, &final_offsets, &final_counts, &num_buckets
    );
    double t_bkt1 = now_ms();
    double ms_bucket = t_bkt1 - t_bkt0;

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

    // --- timings printout (keep your format)
    printf("\n--- Host Timings (ms) ---\n");
    printf("Input generation:     %.3f ms\n", t_gen1 - t_gen0);
    printf("DPU alloc (ranks):    %.3f ms\n", ms_alloc);
    // printf("DPU load (program):   %.3f ms\n", ms_load); // Consistent and negligable time
    printf("Bucketing/packing:    %.3f ms\n", ms_bucket);
    printf("Host→DPU span:        %.3f ms\n", ms_h2d);
    printf("Execute span:         %.3f ms\n", ms_exec);
    printf("DPU→Host span:        %.3f ms\n", ms_d2h);
    printf("TOTAL wall time:      %.3f ms\n", now_ms() - t0_total);


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
