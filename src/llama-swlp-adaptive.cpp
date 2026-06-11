#include "llama-swlp-adaptive.h"
#include "llama-impl.h"
#include "llama-swlp-internal.h"

#include <algorithm>

void llama_swlp::auto_tune_adaptive_params() {
    if (!enabled || !state->adaptive_enabled) return;

    float model_gb = state->total_model_bytes / (1024.0f * 1024.0f * 1024.0f);

    // Set min_window based on model layer count (at least 10% of layers, min 2)
    state->min_window = std::max(2, state->num_layers / 10);
    state->max_window = state->num_layers - 1;

    // Clamp starting window if it falls below min_window
    if (state->window_size < state->min_window) {
        state->window_size = state->min_window;
    }

    // Auto-select EWMA alpha based on model size (discrete tiers)
    if (state->alpha_auto) {
        float auto_alpha;
        if      (model_gb < 1.0f)  auto_alpha = 0.12f;  // very small: suppress noise aggressively
        else if (model_gb < 3.0f)  auto_alpha = 0.18f;  // small
        else if (model_gb < 8.0f)  auto_alpha = 0.24f;  // medium
        else                        auto_alpha = 0.28f;  // large+

        state->ewma_alpha = std::clamp(auto_alpha, 0.10f, 0.30f);

        if (verbose) {
            LLAMA_LOG_INFO("SWLP: auto-tuned alpha=%.2f (model=%.1f GB)\n",
                state->ewma_alpha, model_gb);
        }
    }

    // Auto-select adapt_interval based on model size
    if (state->adapt_interval == 0) {
        if      (model_gb < 1.0f)  state->adapt_interval = 8;   // high-throughput, save overhead
        else if (model_gb < 5.0f)  state->adapt_interval = 4;   // medium
        else                        state->adapt_interval = 2;   // large: migration costly, react fast

        if (verbose) {
            LLAMA_LOG_INFO("SWLP: auto-tuned adapt_interval=%d (model=%.1f GB)\n",
                state->adapt_interval, model_gb);
        }
    }
}

void llama_swlp::record_layer_timing(int il, int64_t compute_us) {
    if (!enabled || !state->adaptive_enabled) return;
    if (il < 0 || il >= state->num_layers) return;

    state->layer_compute_us[il] = compute_us;

    // Estimate migration time for this layer
    // PCIe 3.0 x16 ~12 GB/s = 12e9 bytes/s -> time_us = bytes / (12e9) * 1e6 = bytes / 12000
    size_t layer_bytes = state->layer_sizes_bytes[il];
    constexpr double PCIE_BANDWIDTH_GBPS = 12.0;
    int64_t migrate_us = (int64_t)(layer_bytes / (PCIE_BANDWIDTH_GBPS * 1000.0));
    state->layer_migrate_us[il] = migrate_us;
}

void llama_swlp::record_compute_time(int64_t compute_us) {
    if (!enabled) return;
    state->last_compute_us = compute_us;
    state->cumulative_compute_us += compute_us;
}

void llama_swlp::record_forward_type(bool is_generation) {
    if (!enabled || !state->adaptive_enabled) return;
    state->is_generation = is_generation;
}

void llama_swlp::adapt_window() {
    if (!enabled || !state->adaptive_enabled) return;
    if (state->forward_count < 3) {
        state->forward_count++;
        return; // need warmup
    }

    // Only adapt during generation mode (skip prompt processing batches)
    if (!state->is_generation) {
        state->forward_count++;
        return;
    }

    // Decimate: only run full adaptation every adapt_interval decodes
    if (state->forward_count % state->adapt_interval != 0) {
        state->forward_count++;
        return;
    }

    // Use ACTUAL measured migration time from prepare_migration() instead of
    // the PCIe-bandwidth estimate. The measured time reflects real conditions.
    int64_t measured_migrate_us = state->last_migration_us;

    // If no significant migration happened (all layers fixed, e.g. ngl=99, or
    // window hasn't moved), adaptive has nothing to optimize — keep current window.
    // Threshold: skip if migration took less than 50us (trivial function overhead).
    if (measured_migrate_us < 50) {
        state->forward_count++;
        return;
    }

    // Compute average layer compute time within the current window
    int64_t avg_compute_us = 0;
    int count = 0;
    int w_start = state->window_start;
    int w_end = w_start + state->window_size;
    for (int il = w_start; il < w_end && il < state->num_layers; il++) {
        if (state->layer_compute_us[il] > 0) {
            avg_compute_us += state->layer_compute_us[il];
            count++;
        }
    }
    if (count == 0) {
        state->forward_count++;
        return;
    }
    avg_compute_us /= count;

    // Gen-mode EWMA: use 4x stronger smoothing than PP mode because single-token
    // decode times are noisy (microseconds vs milliseconds in PP mode).
    float alpha = state->ewma_alpha * 0.25f;
    if (state->ewma_layer_us == 0.0f) {
        state->ewma_layer_us   = (float)avg_compute_us;
        state->ewma_migrate_us = (float)measured_migrate_us;
    } else {
        state->ewma_layer_us   = alpha * avg_compute_us + (1.0f - alpha) * state->ewma_layer_us;
        state->ewma_migrate_us = alpha * measured_migrate_us + (1.0f - alpha) * state->ewma_migrate_us;
    }

    // Decision: can the window's total compute time cover a single migration?
    //   ratio = (W * compute_per_layer) / migrate_time
    //   ratio < 0.5  -> migration dominates, need larger window
    //   ratio > 4.0  -> wasting VRAM, can shrink
    float ratio = (state->window_size * state->ewma_layer_us) / (state->ewma_migrate_us + 1.0f);
    int direction = 0;

    if (ratio < 0.5f) {
        direction = 1;  // expand
    } else if (ratio > 4.0f && state->window_size > state->min_window) {
        direction = -1; // shrink
    }

    // Hysteresis: require 8 consecutive same-direction suggestions to
    // prevent jitter from noisy single-token decode timings.
    static constexpr int GEN_HYSTERESIS = 8;
    if (direction != 0 && direction == state->hysteresis_direction) {
        state->hysteresis_count++;
    } else {
        state->hysteresis_direction = direction;
        state->hysteresis_count = (direction != 0) ? 1 : 0;
    }

    int new_window = state->window_size;
    if (state->hysteresis_count >= GEN_HYSTERESIS) {
        if (direction > 0) {
            new_window = state->window_size + 1;
        } else if (direction < 0) {
            new_window = state->window_size - 1;
        }
        state->hysteresis_count = 0;
    }

    new_window = std::clamp(new_window, state->min_window, state->max_window);

    if (new_window != state->window_size) {
        state->window_size = new_window;
        state->window_adjust_count++;

        if (verbose) {
            float active_mb = (float)estimate_active_memory_mb() / 1024.0f;
            LLAMA_LOG_INFO("SWLP adaptive: window %d (ratio=%.2f, active=%.1f GB, compute=%.0fus, migrate=%.0fus)\n",
                state->window_size, ratio, active_mb, (double)state->ewma_layer_us, (double)state->ewma_migrate_us);
        }
    }

    state->forward_count++;
}


