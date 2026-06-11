#include "llama-swlp-stats.h"
#include "llama-impl.h"
#include "llama-swlp-internal.h"

void llama_swlp::print_stats() const {
    if (!enabled) return;

    size_t active_mb = estimate_active_memory_mb();
    size_t total_mb = estimate_total_memory_mb();
    float ratio = get_memory_reduction_ratio();

    LLAMA_LOG_INFO("SWLP stats:\n");
    LLAMA_LOG_INFO("  window:       %d/%d layers active\n",
        state->window_size, state->num_layers);
    LLAMA_LOG_INFO("  memory:       %zu MB active / %zu MB total (%.1f%%)\n",
        active_mb, total_mb, ratio * 100.0f);
    LLAMA_LOG_INFO("  cache:        %lld hits\n",
        (long long)state->cache_hits);
    if (!state->layer_sizes_bytes.empty()) {
        LLAMA_LOG_INFO("  per-layer:   ");
        for (int i = 0; i < state->num_layers && i < 5; i++) {
            LLAMA_LOG_INFO("%.1f ", state->layer_sizes_bytes[i] / (1024.0*1024.0));
        }
        LLAMA_LOG_INFO("... MB\n");
    }
    LLAMA_LOG_INFO("  copies:       %lld\n", (long long)state->copy_count);
    LLAMA_LOG_INFO("  migrations:   %d total, %zu MB migrated\n",
        state->total_migrations, state->total_migrated_bytes / (1024 * 1024));
    if (state->adaptive_enabled) {
        LLAMA_LOG_INFO("  adaptive:     %d adjustments, window now %d (alpha=%.2f)\n",
            state->window_adjust_count, state->window_size, state->ewma_alpha);
    }
    if (state->cumulative_migrate_us > 0 || state->cumulative_compute_us > 0) {
        LLAMA_LOG_INFO("  timing:       compute=%.1f ms, migration=%.1f ms\n",
            state->cumulative_compute_us / 1000.0, state->cumulative_migrate_us / 1000.0);
        if (state->cumulative_compute_us > 0 && state->cumulative_migrate_us > 0) {
            double ratio = (double)state->cumulative_migrate_us / state->cumulative_compute_us * 100.0;
            LLAMA_LOG_INFO("  overhead:     migration is %.1f%% of compute time\n", ratio);
        }
    }
    if (state->expert_cache_size > 0) {
        LLAMA_LOG_INFO("  expert cache: %d per layer, %lld hits, %lld misses (%.1f%% hit rate)\n",
            state->expert_cache_size,
            (long long)state->expert_cache_hits,
            (long long)state->expert_cache_misses,
            state->expert_cache_hits + state->expert_cache_misses > 0 ?
                100.0 * state->expert_cache_hits / (state->expert_cache_hits + state->expert_cache_misses) : 0.0);
        LLAMA_LOG_INFO("  expert stream: %zu MB total\n",
            state->expert_bytes_streamed / (1024 * 1024));
    }
}

float llama_swlp::get_ewma_alpha() const {
    if (!enabled || !state) return 0.3f;
    return state->ewma_alpha;
}

int llama_swlp::get_adapt_interval() const {
    if (!enabled || !state) return 0;
    return state->adapt_interval;
}

size_t llama_swlp::estimate_active_memory_mb() const {
    if (!enabled || state->layer_sizes_bytes.empty()) return 0;

    size_t active_bytes = state->non_layer_bytes; // always-resident
    for (int i = 0; i < state->num_layers; i++) {
        if (state->layer_in_gpu[i]) {
            active_bytes += state->layer_sizes_bytes[i];
        }
    }
    return active_bytes / (1024 * 1024);
}

size_t llama_swlp::estimate_total_memory_mb() const {
    if (!enabled || state->total_model_bytes == 0) return 0;
    return state->total_model_bytes / (1024 * 1024);
}

float llama_swlp::get_memory_reduction_ratio() const {
    if (!enabled || state->total_model_bytes == 0) return 1.0f;
    size_t active_bytes = state->non_layer_bytes;
    for (int i = 0; i < state->num_layers; i++) {
        if (state->layer_in_gpu[i]) {
            active_bytes += state->layer_sizes_bytes[i];
        }
    }
    return (float)active_bytes / (float)state->total_model_bytes;
}
