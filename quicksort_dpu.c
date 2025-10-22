// quicksort_dpu.c  — DPU-side in-place quicksort for MRAM_ARR
// Debug-safe (no vprintf). Works with any NR_TASKLETS (tasklet 0 sorts).
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <perfcounter.h>

#ifndef NR_TASKLETS
#define NR_TASKLETS 1
#endif

// ---------- Per-DPU stats (host pulls this) ----------
struct dpu_stats {
    uint32_t n_elems;
    uint32_t nr_tasklets;
    uint32_t cycles_sort;     // duration of sort region
    uint32_t cycles_total;    // timestamp at end (since perfcounter_config)
    uint32_t cycles_start;    // timestamp at start of sort region
};
__host struct dpu_stats STATS;

// Barrier object is created for any NR_TASKLETS; waits are guarded below.
BARRIER_INIT(sync_barrier, NR_TASKLETS);

// ---------- Element & MRAM layout ----------
typedef struct __attribute__((packed, aligned(8))) {
    uint32_t v;
    uint32_t pad; // keep 8B alignment for MRAM transactions
} elem_t;

#ifndef MAX_ELEMS_PER_DPU
#define MAX_ELEMS_PER_DPU 8192000u
#endif

__mram_noinit elem_t MRAM_ARR[MAX_ELEMS_PER_DPU];

// ---------- MRAM helpers ----------
static inline __mram_ptr void *mram_ptr_of(uint32_t idx) {
    return (__mram_ptr void *)&MRAM_ARR[idx];
}

static inline elem_t mram_get(uint32_t idx) {
    elem_t e;
    mram_read(mram_ptr_of(idx), &e, sizeof(elem_t));
    return e;
}

static inline void mram_set(uint32_t idx, const elem_t *e) {
    mram_write(e, mram_ptr_of(idx), sizeof(elem_t));
}

static inline void mram_swap(uint32_t i, uint32_t j) {
    if (i == j) return;
    elem_t a, b;
    mram_read(mram_ptr_of(i), &a, sizeof(elem_t));
    mram_read(mram_ptr_of(j), &b, sizeof(elem_t));
    mram_write(&b, mram_ptr_of(i), sizeof(elem_t));
    mram_write(&a, mram_ptr_of(j), sizeof(elem_t));
}

// ---------- Small-range insertion sort on MRAM [lo, hi] ----------
static void insertion_sort_mram(uint32_t lo, uint32_t hi) {
    for (uint32_t i = lo + 1; i <= hi; i++) {
        elem_t key = mram_get(i);
        uint32_t j = i;
        while (j > lo) {
            elem_t prev = mram_get(j - 1);
            if (prev.v <= key.v) break;
            mram_set(j, &prev);  // shift right
            j--;
        }
        if (j != i) mram_set(j, &key);
    }
}

// Median-of-three: makes MRAM[lo] <= MRAM[mid] <= MRAM[hi],
// moves pivot (mid) to hi-1, returns pivot value.
static uint32_t median_of_three(uint32_t lo, uint32_t hi) {
    uint32_t mid = lo + ((hi - lo) >> 1);
    elem_t a = mram_get(lo);
    elem_t b = mram_get(mid);
    elem_t c = mram_get(hi);

    if (a.v > b.v) { elem_t t = a; a = b; b = t; mram_set(lo, &a); mram_set(mid, &b); }
    else { mram_set(lo, &a); mram_set(mid, &b); }

    if (b.v > c.v) { elem_t t = b; b = c; c = t; mram_set(mid, &b); mram_set(hi, &c); }
    else { mram_set(mid, &b); mram_set(hi, &c); }

    a = mram_get(lo);
    b = mram_get(mid);
    if (a.v > b.v) { elem_t t = a; a = b; b = t; mram_set(lo, &a); mram_set(mid, &b); }

    elem_t pv = mram_get(mid);
    mram_swap(mid, hi - 1);  // stash pivot at hi-1
    return pv.v;
}

// Partition MRAM [lo, hi] around pivot value `pv` (pivot at hi-1). Returns pivot index.
static uint32_t partition_mram(uint32_t lo, uint32_t hi, uint32_t pv) {
    uint32_t i = lo;
    uint32_t j = hi - 1; // pivot at hi-1
    while (true) {
        while (i < j) {
            elem_t ei = mram_get(i);
            if (ei.v >= pv) break;
            i++;
        }
        while (j > i) {
            elem_t ej = mram_get(j - 1); // element before pivot slot
            if (ej.v <= pv) break;
            j--;
        }
        if (i >= j) break;
        mram_swap(i, j - 1);
        i++;
        j--;
    }
    mram_swap(i, hi - 1); // restore pivot into place
    return i;
}

// Iterative quicksort with insertion-sort cutoff; entirely MRAM-based.
static void quicksort_mram(uint32_t lo, uint32_t hi) {
    if (hi <= lo) return;

    const uint32_t TINY = 32; // cutoff for insertion sort

    typedef struct { uint32_t lo, hi; } range_t;
    // Depth for 32-bit keys is small; 64 entries is ample.
    range_t *stack = (range_t *)mem_alloc(sizeof(range_t) * 64);
    uint32_t sp = 0;

    stack[sp++] = (range_t){ lo, hi };

    while (sp) {
        range_t r = stack[--sp];
        uint32_t l = r.lo, h = r.hi;
        if (h <= l) continue;

        uint32_t n = h - l + 1;
        if (n <= TINY) {
            insertion_sort_mram(l, h);
            continue;
        }

        uint32_t pv = median_of_three(l, h);
        uint32_t p  = partition_mram(l, h, pv);

        // Process smaller partition first to keep stack shallow.
        uint32_t left_n  = (p > l)     ? (p - l)     : 0;
        uint32_t right_n = (p < h)     ? (h - p)     : 0;

        if (left_n > right_n) {
            if (l < p)     stack[sp++] = (range_t){ l, p - 1 };
            if (p + 1 < h) stack[sp++] = (range_t){ p + 1, h };
        } else {
            if (p + 1 < h) stack[sp++] = (range_t){ p + 1, h };
            if (l < p)     stack[sp++] = (range_t){ l, p - 1 };
        }
    }
}

// ---------- Entry ----------
int main() {
    mem_reset(); // ready WRAM allocator

    const uint32_t tid = me();
    if (tid == 0) {
        STATS.nr_tasklets = NR_TASKLETS;
    }

    // Host preloads STATS.n_elems and MRAM_ARR[0..n-1].
    // (Host may pad MRAM beyond n; we ignore those.)
    uint32_t n = STATS.n_elems;
    if (n > MAX_ELEMS_PER_DPU) n = MAX_ELEMS_PER_DPU;

    if (NR_TASKLETS > 1) barrier_wait(&sync_barrier);

    if (tid == 0) {
        perfcounter_config(COUNT_CYCLES, true);
        STATS.cycles_start = perfcounter_get();

        if (n > 1) quicksort_mram(0, n - 1);

        STATS.cycles_total = perfcounter_get();
        STATS.cycles_sort  = STATS.cycles_total - STATS.cycles_start;
    }

    if (NR_TASKLETS > 1) barrier_wait(&sync_barrier);
    return 0;
}
