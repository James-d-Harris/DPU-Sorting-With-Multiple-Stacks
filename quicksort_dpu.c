// dpu_quicksort.c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <perfcounter.h>

#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

// ---------- Per-DPU stats (host pulls this) ----------
struct dpu_stats {
    uint32_t n_elems;
    uint32_t nr_tasklets;
    uint32_t cycles_sort;    // cycles in quicksort region (tasklet 0)
    uint32_t cycles_total;   // cycles since perfcounter_config (tasklet 0)
};
__host struct dpu_stats STATS;

BARRIER_INIT(sync_barrier, NR_TASKLETS);

// ---------- Element & MRAM layout ----------
typedef struct __attribute__((packed, aligned(8))) {
    uint32_t v;
    uint32_t pad; // keep 8B alignment for MRAM transactions
} elem_t;

#ifndef MAX_ELEMS_PER_DPU
#define MAX_ELEMS_PER_DPU (1u << 20)
#endif

__mram_noinit elem_t MRAM_ARR[MAX_ELEMS_PER_DPU];

// ---------- Host-provided arguments ----------
struct dpu_args {
    uint32_t n_elems;   // how many valid elems in MRAM_ARR
};
__host struct dpu_args ARGS;

// ---------- MRAM helpers (8B aligned) ----------
static inline void mram_read_elem(uint32_t idx, elem_t *dst) {
    mram_read(&MRAM_ARR[idx], dst, sizeof(elem_t));
}
static inline void mram_write_elem(uint32_t idx, const elem_t *src) {
    mram_write(src, &MRAM_ARR[idx], sizeof(elem_t));
}
static inline void mram_swap(uint32_t i, uint32_t j, elem_t *tmp_i, elem_t *tmp_j) {
    if (i == j) return;
    mram_read_elem(i, tmp_i);
    mram_read_elem(j, tmp_j);
    mram_write_elem(i, tmp_j);
    mram_write_elem(j, tmp_i);
}

static inline uint32_t read_value(uint32_t idx, elem_t *tmp) {
    mram_read_elem(idx, tmp);
    return tmp->v;
}

// ---------- Iterative quicksort on MRAM ----------
typedef struct { uint32_t lo, hi; } range_t;

static inline uint32_t median3(uint32_t a, uint32_t b, uint32_t c, elem_t *t) {
    uint32_t va = read_value(a, t), vb = read_value(b, t), vc = read_value(c, t);
    if ((va < vb) ^ (va < vc)) return a;
    if ((vb < va) ^ (vb < vc)) return b;
    return c;
}

static void quicksort_mram(uint32_t start, uint32_t count) {
    if (count <= 1) return;

    // Depth ~ 2*log2(n)
    uint32_t lg = 0; for (uint32_t t = count; t > 1; t >>= 1) lg++;
    const uint32_t STACK_MAX = (lg << 1) + 8;
    const uint32_t STACK_CAP = (STACK_MAX < 128) ? STACK_MAX : 128;

    range_t *stack = (range_t *)mem_alloc(sizeof(range_t) * STACK_CAP);
    if (!stack) return; // out of WRAM heap, bail

    uint32_t sp = 0;
    elem_t a_buf, b_buf, piv_buf;

    stack[sp++] = (range_t){ .lo = start, .hi = start + count - 1 };

    while (sp) {
        range_t rg = stack[--sp];
        uint32_t lo = rg.lo, hi = rg.hi;

        while (lo < hi) {
            uint32_t mid = lo + ((hi - lo) >> 1);
            uint32_t piv_idx = median3(lo, mid, hi, &piv_buf);
            mram_swap(piv_idx, hi, &a_buf, &b_buf);
            uint32_t pivot = read_value(hi, &piv_buf);

            uint32_t i = lo;
            for (uint32_t j = lo; j < hi; j++) {
                uint32_t vj = read_value(j, &a_buf);
                if (vj <= pivot) { mram_swap(i, j, &a_buf, &b_buf); i++; }
            }
            mram_swap(i, hi, &a_buf, &b_buf);

            uint32_t left_lo = lo, left_hi = (i == 0) ? 0 : (i - 1);
            uint32_t right_lo = i + 1, right_hi = hi;

            uint32_t left_sz  = (left_hi >= left_lo) ? (left_hi - left_lo + 1) : 0;
            uint32_t right_sz = (right_hi >= right_lo) ? (right_hi - right_lo + 1) : 0;

            if (left_sz < right_sz) {
                if (right_sz > 1 && sp < STACK_CAP) stack[sp++] = (range_t){ right_lo, right_hi };
                hi = (left_sz > 0) ? left_hi : left_lo;
            } else {
                if (left_sz > 1 && sp < STACK_CAP) stack[sp++] = (range_t){ left_lo, left_hi };
                lo = right_lo;
            }
        }
    }
}

// ---------- Tasklet entry ----------
int main() {
    if (me() == 0) {
        mem_reset();
        perfcounter_config(COUNT_CYCLES, true);
    }
    barrier_wait(&sync_barrier);

    uint32_t n = ARGS.n_elems;

    uint32_t sort_start = 0, sort_end = 0;
    if (me() == 0) sort_start = perfcounter_get();
    barrier_wait(&sync_barrier);

    if (n > 1 && me() == 0) quicksort_mram(0, n);

    barrier_wait(&sync_barrier);

    if (me() == 0) {
        sort_end = perfcounter_get();
        STATS.n_elems      = n;
        STATS.nr_tasklets  = NR_TASKLETS;
        STATS.cycles_sort  = sort_end - sort_start;
        STATS.cycles_total = perfcounter_get();
    }

    barrier_wait(&sync_barrier);
    return 0;
}
