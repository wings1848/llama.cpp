#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <vector>
#include <cstdint>

struct llama_model;

struct llama_swlp_state {
    // ======== Window management ========
    int window_size;
    int prefetch_depth;
    int num_layers;
    int window_start = 0;
    int fixed_gpu_layers = 0;  // layers < this are permanently GPU-resident (ngl)
    bool auto_window = false;  // window_size was auto-computed, may be refined by analyze_model

    // Incremental migration: track previous window to only migrate the delta
    int  prev_window_start = -1;
    int  prev_window_end   = -1;
    int  prev_prefetch_end = -1;
    bool migration_initialized = false;
    int  last_migrated_window = 0;  // window_size at last prepare_migration call (for graph-reuse detection)

    std::vector<bool> layer_in_gpu;
    std::vector<bool> layer_fixed_gpu;  // layers pre-loaded on GPU by model loader (never evict)
    std::vector<int>  layer_boundaries;

    std::vector<size_t> layer_sizes_bytes;
    size_t total_model_bytes = 0;
    size_t non_layer_bytes   = 0;

    // ======== Backend & migration ========
    ggml_backend_t gpu_backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    bool backend_migration_ready = false;

    // Async migration (PCIe pipelining with CUDA streams)
    bool    async_migration_enabled = false;
    std::vector<void *> migration_events; // cudaEvent_t per layer (nullptr = no pending migration)

    const llama_model * model_ptr = nullptr;

    struct layer_gpu_state {
        ggml_backend_buffer_t gpu_buffer = nullptr;
        std::vector<void*>              saved_data;
        std::vector<ggml_backend_buffer_t> saved_buffers;
        // Pinned host buffer for async H2D migration (page-locked, safe for cudaMemcpyAsync).
        // Allocated per-layer when async=true, freed after event is consumed.
        ggml_backend_buffer_t migration_host_buf = nullptr;
    };
    std::vector<layer_gpu_state> layer_gpu;

    size_t total_migrated_bytes = 0;
    int    total_migrations = 0;

    // ======== Counters ========
    int64_t copy_count   = 0;
    int64_t cache_hits   = 0;

    // ======== Adaptive tuning (EWMA) ========
    bool adaptive_enabled = false;
    int  min_window = 2;
    int  max_window = 0;

    std::vector<int64_t> layer_compute_us;
    std::vector<int64_t> layer_migrate_us;

    float ewma_layer_us   = 0.0f;
    float ewma_migrate_us = 0.0f;
    float ewma_alpha      = 0.3f;
    bool  alpha_auto      = true;
    int   adapt_interval  = 4;
    int   forward_count       = 0;
    int   window_adjust_count = 0;

    // Hysteresis: only apply window change after N consecutive same-direction suggestions
    int   hysteresis_direction = 0;  // -1=shrink, 0=neutral, 1=expand
    int   hysteresis_count     = 0;

    // Separate PP/Gen: adaptive only runs during generation (not batch prompt processing)
    bool  is_generation = false;

    // ======== Timing instrumentation ========
    int64_t last_migration_us     = 0;
    int64_t last_compute_us       = 0;
    int64_t cumulative_migrate_us = 0;
    int64_t cumulative_compute_us = 0;

    // ======== MoE expert cache ========
    int expert_cache_size     = 0;
    int expert_prefetch_depth = 0;
    bool is_moe = false;

    std::vector<std::vector<bool>>    expert_in_cache;
    std::vector<int>                  expert_count_per_layer;
    std::vector<std::vector<int64_t>> activation_counts;
    std::vector<std::vector<size_t>>  expert_sizes;

    int64_t expert_cache_hits   = 0;
    int64_t expert_cache_misses = 0;
    size_t  expert_bytes_streamed = 0;

    llama_swlp_state(int ws, int pd, int nl)
        : window_size(ws), prefetch_depth(pd), num_layers(nl)
        , layer_in_gpu(nl, false)
        , layer_fixed_gpu(nl, false)
        , layer_gpu(nl)
        , migration_events(nl, nullptr)
        , expert_in_cache(nl)
        , expert_count_per_layer(nl, 0)
        , activation_counts(nl)
        , expert_sizes(nl)
    {}

    ~llama_swlp_state() {
        for (int il = 0; il < num_layers; il++) {
            if (layer_gpu[il].gpu_buffer) {
                ggml_backend_buffer_free(layer_gpu[il].gpu_buffer);
                layer_gpu[il].gpu_buffer = nullptr;
            }
            if (layer_gpu[il].migration_host_buf) {
                ggml_backend_buffer_free(layer_gpu[il].migration_host_buf);
                layer_gpu[il].migration_host_buf = nullptr;
            }
#ifdef GGML_USE_CUDA
            if (migration_events[il]) {
                ggml_backend_cuda_migrate_event_destroy(migration_events[il]);
                migration_events[il] = nullptr;
            }
#endif
        }
    }
};
