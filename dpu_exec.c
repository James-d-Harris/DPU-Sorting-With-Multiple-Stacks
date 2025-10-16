#define _POSIX_C_SOURCE 200809L
#include "dpu_exec.h"

#include <dpu.h>
#include <dpu_log.h>

#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include <omp.h>

#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

#ifndef DPU_INPUT_SYM
#define DPU_INPUT_SYM "MRAM_ARR"
#endif
#ifndef DPU_STATS_SYM
#define DPU_STATS_SYM "STATS"
#endif

// Host-side mirror of elem_t in DPU (8B aligned/packed)
typedef struct __attribute__((packed, aligned(8))) {
    uint32_t v;
    uint32_t pad;
} elem_t;

static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static double span_ms(const double *starts, const double *ends, uint32_t num_sets) {
    double min_s = 1e300, max_e = -1e300; uint32_t seen = 0;
    for (uint32_t s = 0; s < num_sets; s++) {
        if (starts[s] == 0.0 && ends[s] == 0.0) continue;
        if (starts[s] < min_s) min_s = starts[s];
        if (ends[s]   > max_e) max_e = ends[s];
        seen++;
    }
    if (seen == 0 || max_e < min_s) return 0.0;
    return max_e - min_s;
}




struct dpu_exec_ctx {
    uint32_t num_sets;
    struct dpu_set_t *sets;     // array length num_sets (each from dpu_alloc_ranks)
    struct dpu_set_t *dpus;     // flat array of all DPU handles (ordered by set)
    uint32_t *set_begin;        // per-set begin index into dpus[]
    uint32_t *set_end;          // per-set end index (exclusive)
    uint32_t nb_dpus_total;
};

static void summarize_set(uint32_t set_id,
                          const struct dpu_stats *stats,
                          const uint32_t *counts,
                          uint32_t begin,
                          uint32_t end)
{
    uint32_t min_c = UINT32_MAX;
    uint32_t max_c = 0;
    double   sum_c = 0.0;
    uint64_t sum_n = 0ull;

    for (uint32_t i = begin; i < end; i++) {
        uint32_t c = stats[i].cycles_sort;
        if (c < min_c) { min_c = c; }
        if (c > max_c) { max_c = c; }
        sum_c += (double)c;
        sum_n += (uint64_t)counts[i];
    }

    double avg_c = 0.0;
    uint32_t ndpus = (end > begin) ? (end - begin) : 0;
    if (ndpus > 0) {
        avg_c = sum_c / (double)ndpus;
    }

    printf("[Set %u] cycles_sort min=%" PRIu32 " avg=%.1f max=%" PRIu32
           " ; elems=%" PRIu64 " ; dpus=%" PRIu32 "\n",
           set_id, min_c, avg_c, max_c, sum_n, ndpus);
}

int dpu_prepare_sets(uint32_t total_ranks,
                     uint32_t num_sets,
                     const char *dpu_exe,
                     dpu_exec_ctx_t **out_ctx,
                     uint32_t *out_nb_dpus_total,
                     double *out_ms_alloc,
                     double *out_ms_load)
{
    fprintf(stderr, "[exec] dpu_prepare_sets(total_ranks=%u, num_sets=%u)\n",
            (unsigned)total_ranks, (unsigned)num_sets);

    if (!out_ctx || !out_nb_dpus_total) return -1;
    if (num_sets == 0) {
        fprintf(stderr, "num_sets must be > 0\n");
        return -2;
    }
    if ((total_ranks % num_sets) != 0) {
        fprintf(stderr, "TOTAL_RANKS (%u) must be divisible by NUM_SETS (%u)\n",
                (unsigned)total_ranks, (unsigned)num_sets);
        return -3;
    }

    dpu_exec_ctx_t *ctx = (dpu_exec_ctx_t *)calloc(1, sizeof(*ctx));
    if (!ctx) return -4;

    ctx->num_sets  = num_sets;
    ctx->sets      = (struct dpu_set_t *)calloc(num_sets, sizeof(struct dpu_set_t));
    ctx->set_begin = (uint32_t *)calloc(num_sets, sizeof(uint32_t));
    ctx->set_end   = (uint32_t *)calloc(num_sets, sizeof(uint32_t));
    if (!ctx->sets || !ctx->set_begin || !ctx->set_end) {
        free(ctx->sets); free(ctx->set_begin); free(ctx->set_end); free(ctx);
        return -5;
    }

    const uint32_t ranks_per_set = total_ranks / num_sets;
    fprintf(stderr, "[exec] ranks_per_set=%u; beginning per-set allocation\n",
            (unsigned)ranks_per_set);

    // Per-set bookkeeping for timing & errors
    uint32_t *counts     = (uint32_t *)calloc(num_sets, sizeof(uint32_t));
    int      *alloc_err  = (int *)calloc(num_sets, sizeof(int));
    int      *load_err   = (int *)calloc(num_sets, sizeof(int));
    double   *alloc_t0   = (double *)calloc(num_sets, sizeof(double));
    double   *alloc_t1   = (double *)calloc(num_sets, sizeof(double));
    double   *load_t0    = (double *)calloc(num_sets, sizeof(double));
    double   *load_t1    = (double *)calloc(num_sets, sizeof(double));
    if (!counts || !alloc_err || !load_err || !alloc_t0 || !alloc_t1 || !load_t0 || !load_t1) {
        free(counts); free(alloc_err); free(load_err);
        free(alloc_t0); free(alloc_t1); free(load_t0); free(load_t1);
        free(ctx->sets); free(ctx->set_begin); free(ctx->set_end); free(ctx);
        return -5;
    }

    // ---------- Parallel allocation + count ----------
    #pragma omp parallel for schedule(static) num_threads(num_sets)
    for (uint32_t s = 0; s < num_sets; s++) {
        alloc_t0[s] = now_ms();

        struct dpu_set_t set;
        fprintf(stderr, "[exec] set %u: calling dpu_alloc_ranks(%u)\n",
                s, (unsigned)ranks_per_set);
        dpu_error_t derr = dpu_alloc_ranks(ranks_per_set, NULL, &set);
        if (derr != DPU_OK) {
            alloc_err[s] = 1;
            fprintf(stderr, "dpu_alloc_ranks failed for set %u: %s\n",
                    s, dpu_error_to_string(derr));
        } else {
            ctx->sets[s] = set;
            fprintf(stderr, "[exec] set %u: dpu_alloc_ranks returned %s\n",
                    s, dpu_error_to_string(derr));

            uint32_t local_count = 0;
            dpu_error_t ge = dpu_get_nr_dpus(set, &local_count);
            if (ge != DPU_OK) {
                alloc_err[s] = 1;
                fprintf(stderr, "dpu_get_nr_dpus failed for set %u: %s\n",
                        s, dpu_error_to_string(ge));
            } else {
                counts[s] = local_count;
            }
        }

        alloc_t1[s] = now_ms();
    }

    // Check any allocation failure and compute total DPUs
    uint32_t nb_total = 0;
    int any_alloc_fail = 0;
    for (uint32_t s = 0; s < num_sets; s++) {
        if (alloc_err[s]) any_alloc_fail = 1;
        nb_total += counts[s];
        if (counts[s] > 0) {
            fprintf(stderr, "[exec] set %u: discovered %u DPUs in this set; total so far=%u\n",
                    s, counts[s], nb_total);
        }
    }
    if (any_alloc_fail || nb_total == 0) {
        fprintf(stderr, "%s\n", nb_total == 0
                ? "No DPUs discovered across all sets."
                : "One or more sets failed during allocation.");
        // cleanup sets that did succeed
        for (uint32_t s = 0; s < num_sets; s++) {
            if (!alloc_err[s] && counts[s] > 0) {
                dpu_free(ctx->sets[s]);
            }
        }
        free(counts); free(alloc_err); free(load_err);
        free(alloc_t0); free(alloc_t1); free(load_t0); free(load_t1);
        free(ctx->sets); free(ctx->set_begin); free(ctx->set_end); free(ctx);
        return -6;
    }

    // Report alloc wall-span (earliest start -> latest end)
    if (out_ms_alloc) {
        double min_s = 1e300, max_e = -1e300;
        for (uint32_t s = 0; s < num_sets; s++) {
            if (alloc_t0[s] < min_s) min_s = alloc_t0[s];
            if (alloc_t1[s] > max_e) max_e = alloc_t1[s];
        }
        *out_ms_alloc = (max_e > min_s) ? (max_e - min_s) : 0.0;
    }

    // ---------- Parallel program load ----------
    #pragma omp parallel for schedule(static) num_threads(num_sets)
    for (uint32_t s = 0; s < num_sets; s++) {
        load_t0[s] = now_ms();
        if (counts[s] == 0) { load_t1[s] = load_t0[s]; continue; }

        dpu_error_t e = dpu_load(ctx->sets[s], dpu_exe, NULL);
        if (e != DPU_OK) {
            load_err[s] = 1;
            fprintf(stderr, "dpu_load failed for set %u: %s\n", s, dpu_error_to_string(e));
        }
        load_t1[s] = now_ms();
    }

    int any_load_fail = 0;
    for (uint32_t s = 0; s < num_sets; s++) {
        if (load_err[s]) { any_load_fail = 1; break; }
    }
    if (any_load_fail) {
        for (uint32_t s = 0; s < num_sets; s++) {
            if (!alloc_err[s] && counts[s] > 0) {
                dpu_free(ctx->sets[s]);
            }
        }
        free(counts); free(alloc_err); free(load_err);
        free(alloc_t0); free(alloc_t1); free(load_t0); free(load_t1);
        free(ctx->sets); free(ctx->set_begin); free(ctx->set_end); free(ctx);
        return -8;
    }

    // Report load wall-span
    if (out_ms_load) {
        double min_s = 1e300, max_e = -1e300;
        for (uint32_t s = 0; s < num_sets; s++) {
            if (load_t0[s] < min_s) min_s = load_t0[s];
            if (load_t1[s] > max_e) max_e = load_t1[s];
        }
        *out_ms_load = (max_e > min_s) ? (max_e - min_s) : 0.0;
    }

    // ---------- Build flat DPU list + set boundaries (serial) ----------
    ctx->dpus = (struct dpu_set_t *)calloc(nb_total, sizeof(struct dpu_set_t));
    if (!ctx->dpus) {
        for (uint32_t s = 0; s < num_sets; s++) {
            dpu_free(ctx->sets[s]);
        }
        free(counts); free(alloc_err); free(load_err);
        free(alloc_t0); free(alloc_t1); free(load_t0); free(load_t1);
        free(ctx->sets); free(ctx->set_begin); free(ctx->set_end); free(ctx);
        return -9;
    }

    uint32_t off = 0;
    for (uint32_t s = 0; s < num_sets; s++) {
        ctx->set_begin[s] = off;

        struct dpu_set_t r, d;
        DPU_RANK_FOREACH(ctx->sets[s], r) {
            DPU_FOREACH(r, d) {
                ctx->dpus[off++] = d;
            }
        }

        ctx->set_end[s] = off;
    }

    ctx->nb_dpus_total = nb_total;
    *out_ctx = ctx;
    *out_nb_dpus_total = nb_total;

    // tidy temporaries
    free(counts); free(alloc_err); free(load_err);
    free(alloc_t0); free(alloc_t1); free(load_t0); free(load_t1);

    return 0;
}

int dpu_run_and_collect(dpu_exec_ctx_t *ctx,
                        uint32_t num_sets,
                        uint32_t max_elems_per_dpu,
                        uint32_t **per_dpu,
                        const uint32_t *counts,
                        struct dpu_stats *stats,
                        double *out_ms_h2d,
                        double *out_ms_exec,
                        double *out_ms_d2h)
{
    if (ctx == NULL || per_dpu == NULL || counts == NULL || stats == NULL) {
        return -1;
    }
    if (num_sets != ctx->num_sets) {
        fprintf(stderr, "num_sets mismatch: caller=%u ctx=%u\n", num_sets, ctx->num_sets);
        return -2;
    }

    // Use boundaries computed in dpu_prepare_sets(...)
    const uint32_t *set_begin = ctx->set_begin;
    const uint32_t *set_end   = ctx->set_end;

    double t_h2d_total  = 0.0;
    double t_exec_total = 0.0;
    double t_d2h_total  = 0.0;

    for (uint32_t s = 0; s < num_sets; s++) {
        uint32_t begin = set_begin[s];
        uint32_t end   = set_end[s];

        // ----- H2D -----
        double t0_h2d = now_ms();

        for (uint32_t i = begin; i < end; i++) {
            uint32_t n = counts[i];

            struct dpu_stats in_stats = {0};
            in_stats.n_elems     = n;
            in_stats.nr_tasklets = NR_TASKLETS;

            dpu_error_t e_stats = dpu_copy_to(ctx->dpus[i], DPU_STATS_SYM, 0,
                                              &in_stats, sizeof(in_stats));
            if (e_stats != DPU_OK) {
                fprintf(stderr, "[Set %u, DPU %u] copy_to(%s) failed: %s\n",
                        s, i, DPU_STATS_SYM, dpu_error_to_string(e_stats));
            }

            if (n > 0u) {
                elem_t *buf = (elem_t *)malloc((size_t)n * sizeof(elem_t));
                if (buf == NULL) {
                    fprintf(stderr, "[Set %u, DPU %u] OOM elem buffer (n=%u)\n", s, i, n);
                } else {
                    for (uint32_t k = 0; k < n; k++) {
                        buf[k].v   = per_dpu[i][k];
                        buf[k].pad = 0u;
                    }
                    dpu_error_t e_in = dpu_copy_to(ctx->dpus[i], DPU_INPUT_SYM, 0,
                                                   buf, (size_t)n * sizeof(elem_t));
                    if (e_in != DPU_OK) {
                        fprintf(stderr, "[Set %u, DPU %u] copy_to(%s) failed: %s\n",
                                s, i, DPU_INPUT_SYM, dpu_error_to_string(e_in));
                    }
                    free(buf);
                }
            }
        }

        double t1_h2d = now_ms();
        t_h2d_total += (t1_h2d - t0_h2d);

        // ----- Execute -----
        double t0_exec = now_ms();
        for (uint32_t i = begin; i < end; i++) {
            dpu_error_t e = dpu_launch(ctx->dpus[i], DPU_SYNCHRONOUS);
            if (e != DPU_OK) {
                fprintf(stderr, "[Set %u, DPU %u] dpu_launch failed: %s\n",
                        s, i, dpu_error_to_string(e));
            }
        }
        double t1_exec = now_ms();
        t_exec_total += (t1_exec - t0_exec);

        // ----- D2H -----
        double t0_d2h = now_ms();
        for (uint32_t i = begin; i < end; i++) {
            dpu_error_t e = dpu_copy_from(ctx->dpus[i], DPU_STATS_SYM, 0,
                                          &stats[i], sizeof(struct dpu_stats));
            if (e != DPU_OK) {
                fprintf(stderr, "[Set %u, DPU %u] copy_from(%s) failed: %s\n",
                        s, i, DPU_STATS_SYM, dpu_error_to_string(e));
            }
        }
        double t1_d2h = now_ms();
        t_d2h_total += (t1_d2h - t0_d2h);

        summarize_set(s, stats, counts, begin, end);
    }

    if (out_ms_h2d)  { *out_ms_h2d  = t_h2d_total; }
    if (out_ms_exec) { *out_ms_exec = t_exec_total; }
    if (out_ms_d2h)  { *out_ms_d2h  = t_d2h_total; }

    return 0;
}

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
    double *out_ms_d2h)
{
    if (!ctx || !bucketed || !final_offsets || !final_counts) {
        fprintf(stderr, "[exec] bucketed: bad args\n"); return -1;
    }
    if (num_sets != ctx->num_sets) {
        fprintf(stderr, "[exec] bucketed: num_sets mismatch\n"); return -2;
    }

    // total available DPUs across sets
    uint32_t total_dpus = 0;
    for (uint32_t s = 0; s < num_sets; s++) {
        total_dpus += (ctx->set_end[s] - ctx->set_begin[s]);
    }
    if (num_buckets > total_dpus) {
        fprintf(stderr, "[exec] bucketed: num_buckets (%u) > total_dpus (%u)\n",
                num_buckets, total_dpus);
        return -3;
    }

    // Per-set time spans
    double *h2d_start = (double *)calloc(num_sets, sizeof(double));
    double *h2d_end   = (double *)calloc(num_sets, sizeof(double));
    double *exe_start = (double *)calloc(num_sets, sizeof(double));
    double *exe_end   = (double *)calloc(num_sets, sizeof(double));
    double *d2h_start = (double *)calloc(num_sets, sizeof(double));
    double *d2h_end   = (double *)calloc(num_sets, sizeof(double));
    if (!h2d_start || !h2d_end || !exe_start || !exe_end || !d2h_start || !d2h_end) {
        free(h2d_start); free(h2d_end); free(exe_start); free(exe_end); free(d2h_start); free(d2h_end);
        fprintf(stderr, "[exec] bucketed: OOM spans\n"); return -4;
    }

    // Per-set error flags (so we don’t return from inside OMP loop)
    int *set_err = (int *)calloc(num_sets, sizeof(int));
    if (!set_err) { fprintf(stderr, "[exec] bucketed: OOM set_err\n"); return -5; }

    #pragma omp parallel for schedule(static) num_threads(num_sets)
    for (uint32_t s = 0; s < num_sets; s++) {
        const uint32_t begin = ctx->set_begin[s];
        const uint32_t end   = ctx->set_end[s];
        const uint32_t ndpus = (end > begin) ? (end - begin) : 0;

        // Base global bucket index = sum of DPUs in all previous sets
        uint32_t set_base = 0;
        for (uint32_t t = 0; t < s; t++) set_base += (ctx->set_end[t] - ctx->set_begin[t]);

        // Find how many buckets remain for this set
        uint32_t remaining = (num_buckets > set_base) ? (num_buckets - set_base) : 0;
        uint32_t ndpus_used = (remaining < ndpus) ? remaining : ndpus;
        if (ndpus_used == 0) {
            // No buckets mapped to this set (idle)
            continue;
        }

        // Max count and MRAM guard
        uint32_t max_cnt = 0;
        for (uint32_t j = 0; j < ndpus_used; j++) {
            uint32_t bidx = set_base + j;
            uint32_t cnt  = final_counts[bidx];
            if (cnt > max_cnt) max_cnt = cnt;
            if (cnt > MAX_ELEMS_PER_DPU) {
                fprintf(stderr, "[exec][set %u] bucket %u too large (%u > %u)\n",
                        s, bidx, cnt, MAX_ELEMS_PER_DPU);
                set_err[s] = 1;
            }
        }
        if (set_err[s]) continue;

        const size_t max_bytes = (size_t)max_cnt * sizeof(elem_t);

        // Stage buffers
        elem_t **h2d_bufs = (elem_t **)calloc(ndpus_used, sizeof(elem_t *));
        struct dpu_stats *h2d_stats = (struct dpu_stats *)calloc(ndpus_used, sizeof(struct dpu_stats));
        if (!h2d_bufs || !h2d_stats) {
            if (h2d_bufs) free(h2d_bufs);
            if (h2d_stats) free(h2d_stats);
            fprintf(stderr, "[exec][set %u] OOM staging\n", s);
            set_err[s] = 1; continue;
        }

        for (uint32_t j = 0; j < ndpus_used; j++) {
            uint32_t bidx = set_base + j;
            uint32_t off  = final_offsets[bidx];
            uint32_t cnt  = final_counts[bidx];

            if (max_bytes > 0) {
                h2d_bufs[j] = (elem_t *)aligned_alloc(8, max_bytes);
                if (!h2d_bufs[j]) {
                    fprintf(stderr, "[exec][set %u] OOM: h2d_bufs[%u]\n", s, j);
                    for (uint32_t t = 0; t < j; t++) free(h2d_bufs[t]);
                    free(h2d_bufs); free(h2d_stats);
                    h2d_bufs = NULL; h2d_stats = NULL;
                    set_err[s] = 1; break;
                }
                // pack
                uint32_t k = 0;
                for (; k < cnt; k++) { h2d_bufs[j][k].v = bucketed[off + k]; h2d_bufs[j][k].pad = 0u; }
                for (; k < max_cnt; k++) { h2d_bufs[j][k].v = 0u; h2d_bufs[j][k].pad = 0u; }
            }

            h2d_stats[j].n_elems      = cnt;
            h2d_stats[j].nr_tasklets  = NR_TASKLETS;
            h2d_stats[j].cycles_sort  = 0u;
            h2d_stats[j].cycles_total = 0u;
            h2d_stats[j].cycles_start = 0u;
        }
        if (set_err[s]) continue;

        // H2D
        double t0_h2d = now_ms();

        if (max_bytes > 0) {
            struct dpu_set_t dpu; uint32_t j = 0, prepared = 0;
            DPU_FOREACH(ctx->sets[s], dpu) {
                if (j >= ndpus_used) break;
                DPU_ASSERT(dpu_prepare_xfer(dpu, h2d_bufs[j]));
                j++; prepared++;
            }
            if (prepared > 0) {
                DPU_ASSERT(dpu_push_xfer(ctx->sets[s], DPU_XFER_TO_DPU, DPU_INPUT_SYM, 0, max_bytes, DPU_XFER_DEFAULT));
            }
        }
        {
            struct dpu_set_t dpu; uint32_t j = 0, prepared = 0;
            DPU_FOREACH(ctx->sets[s], dpu) {
                if (j >= ndpus_used) break;
                DPU_ASSERT(dpu_prepare_xfer(dpu, &h2d_stats[j]));
                j++; prepared++;
            }
            if (prepared > 0) {
                DPU_ASSERT(dpu_push_xfer(ctx->sets[s], DPU_XFER_TO_DPU, DPU_STATS_SYM, 0, sizeof(struct dpu_stats), DPU_XFER_DEFAULT));
            }
        }

        double t1_h2d = now_ms();
        h2d_start[s] = t0_h2d; h2d_end[s] = t1_h2d;

        // EXEC
        double t0_exec = now_ms();
        DPU_ASSERT(dpu_launch(ctx->sets[s], DPU_SYNCHRONOUS));
        double t1_exec = now_ms();
        exe_start[s] = t0_exec; exe_end[s] = t1_exec;

        // D2H
        double t0_d2h = now_ms();

        if (max_bytes > 0) {
            elem_t **d2h_bufs = (elem_t **)calloc(ndpus_used, sizeof(elem_t *));
            if (!d2h_bufs) {
                fprintf(stderr, "[exec][set %u] OOM: d2h_bufs\n", s);
            } else {
                for (uint32_t j = 0; j < ndpus_used; j++) {
                    d2h_bufs[j] = (elem_t *)aligned_alloc(8, max_bytes);
                    if (!d2h_bufs[j]) {
                        for (uint32_t t = 0; t < j; t++) free(d2h_bufs[t]);
                        free(d2h_bufs); d2h_bufs = NULL; break;
                    }
                }
                if (d2h_bufs) {
                    struct dpu_set_t dpu; uint32_t j = 0, prepared = 0;
                    DPU_FOREACH(ctx->sets[s], dpu) {
                        if (j >= ndpus_used) break;
                        DPU_ASSERT(dpu_prepare_xfer(dpu, d2h_bufs[j]));
                        j++; prepared++;
                    }
                    if (prepared > 0) {
                        DPU_ASSERT(dpu_push_xfer(ctx->sets[s], DPU_XFER_FROM_DPU, DPU_INPUT_SYM, 0, max_bytes, DPU_XFER_DEFAULT));
                    }
                    // write back in-place to bucketed array
                    for (uint32_t j2 = 0; j2 < ndpus_used; j2++) {
                        uint32_t bidx = set_base + j2;
                        uint32_t off  = final_offsets[bidx];
                        uint32_t cnt  = final_counts[bidx];
                        for (uint32_t k = 0; k < cnt; k++) {
                            bucketed[off + k] = d2h_bufs[j2][k].v;
                        }
                    }
                    for (uint32_t j3 = 0; j3 < ndpus_used; j3++) free(d2h_bufs[j3]);
                    free(d2h_bufs);
                }
            }
        }

        // pull STATS
        {
            struct dpu_stats *stats_tmp = (struct dpu_stats *)calloc(ndpus_used, sizeof(*stats_tmp));
            if (stats_tmp) {
                struct dpu_set_t dpu; uint32_t j = 0, prepared = 0;
                DPU_FOREACH(ctx->sets[s], dpu) {
                    if (j >= ndpus_used) break;
                    DPU_ASSERT(dpu_prepare_xfer(dpu, &stats_tmp[j]));
                    j++; prepared++;
                }
                if (prepared > 0) {
                    DPU_ASSERT(dpu_push_xfer(ctx->sets[s], DPU_XFER_FROM_DPU, DPU_STATS_SYM, 0, sizeof(struct dpu_stats), DPU_XFER_DEFAULT));
                }
                uint64_t min_c = UINT64_MAX, max_c = 0; double sum_c = 0.0;
                for (uint32_t j2 = 0; j2 < ndpus_used; j2++) {
                    uint64_t c = stats_tmp[j2].cycles_sort;
                    if (c < min_c) min_c = c;
                    if (c > max_c) max_c = c;
                    sum_c += (double)c;
                }
                double avg_c = ndpus_used ? (sum_c / (double)ndpus_used) : 0.0;
                printf("[Set %u] cycles_sort min=%" PRIu64 " avg=%.1f max=%" PRIu64 " ; dpus=%u\n",
                       s, min_c, avg_c, max_c, ndpus_used);
                free(stats_tmp);
            }
        }

        double t1_d2h = now_ms();
        d2h_start[s] = t0_d2h; d2h_end[s] = t1_d2h;

        if (h2d_bufs) { for (uint32_t j = 0; j < ndpus_used; j++) free(h2d_bufs[j]); free(h2d_bufs); }
        if (h2d_stats) free(h2d_stats);
    } // end parallel for

    // Aggregate errors
    int any_err = 0;
    for (uint32_t s = 0; s < num_sets; s++) any_err |= set_err[s];
    free(set_err);

    // Spans (overlap)
    if (out_ms_h2d)  *out_ms_h2d  = span_ms(h2d_start, h2d_end, num_sets);
    if (out_ms_exec) *out_ms_exec = span_ms(exe_start,  exe_end,  num_sets);
    if (out_ms_d2h)  *out_ms_d2h  = span_ms(d2h_start, d2h_end, num_sets);

    free(h2d_start); free(h2d_end);
    free(exe_start); free(exe_end);
    free(d2h_start); free(d2h_end);

    return any_err ? -6 : 0;
}

int dpu_total_dpus(const dpu_exec_ctx_t *ctx, uint32_t *out_total_dpus) {
    if (!ctx || !out_total_dpus) return -1;
    uint32_t total = 0;
    for (uint32_t s = 0; s < ctx->num_sets; s++) {
        total += (ctx->set_end[s] - ctx->set_begin[s]);
    }
    *out_total_dpus = total;
    return 0;
}


void dpu_release(dpu_exec_ctx_t *ctx)
{
    if (!ctx) return;

    if (ctx->dpus) {
        free(ctx->dpus);
    }
    if (ctx->set_begin) {
        free(ctx->set_begin);
    }
    if (ctx->set_end) {
        free(ctx->set_end);
    }
    if (ctx->sets) {
        for (uint32_t s = 0; s < ctx->num_sets; s++) {
            dpu_free(ctx->sets[s]);
        }
        free(ctx->sets);
    }
    free(ctx);
}

