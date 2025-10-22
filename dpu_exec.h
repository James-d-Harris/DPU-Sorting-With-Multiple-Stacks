#ifndef DPU_EXEC_H
#define DPU_EXEC_H

#include <stdint.h>

struct dpu_stats {
    uint32_t n_elems;
    uint32_t nr_tasklets;
    uint32_t cycles_sort;
    uint32_t cycles_total;
    uint32_t cycles_start;
};

typedef struct dpu_exec_ctx dpu_exec_ctx_t;

// NEW: allocate one set per host set using dpu_alloc_ranks(total_ranks/num_sets)
int dpu_prepare_sets(uint32_t total_ranks,
                     uint32_t num_sets,
                     const char *dpu_exe,
                     dpu_exec_ctx_t **out_ctx,
                     uint32_t *out_nb_dpus_total,
                     double *out_ms_alloc,
                     double *out_ms_load);

int dpu_run_and_collect_parallel_bucketed(
    dpu_exec_ctx_t *ctx,
    uint32_t num_sets,
    uint32_t *bucketed,
    uint32_t N,
    const uint32_t *final_offsets,
    const uint32_t *final_counts,
    uint32_t num_buckets,
    double *out_ms_h2d,
    double *out_ms_exec,
    double *out_ms_d2h);


void dpu_release(dpu_exec_ctx_t *ctx);

// Add this prototype
int dpu_total_dpus(const dpu_exec_ctx_t *ctx, uint32_t *out_total_dpus);


#endif
