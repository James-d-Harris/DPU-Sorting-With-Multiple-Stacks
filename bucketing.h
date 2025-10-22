#ifndef BUCKETING_H
#define BUCKETING_H

#include <stdint.h>

int bucketing_build(const uint32_t *input, uint64_t N,
                    uint32_t total_dpus, uint32_t cap,
                    uint32_t **out_bucketed,
                    uint64_t **out_offsets,
                    uint32_t **out_counts,
                    uint32_t *out_num_buckets);

int verify_buckets_host_pre(const uint32_t *bucketed,
                                   const uint64_t *off,
                                   const uint32_t *cnt,
                                   uint32_t nb,
                                uint64_t N_total);

int verify_across_buckets(const uint32_t *bucketed,
                                 const uint64_t *final_offsets,
                                 const uint32_t *final_counts,
                                 uint32_t num_buckets);

#endif
