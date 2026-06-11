#include "llama-swlp.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-swlp-tensors.h"
#include "llama-swlp-internal.h"

#ifdef GGML_USE_CUDA
#include "llama-swlp-cuda.h"
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cstring>

llama_swlp::llama_swlp(const llama_swlp_params & params, int num_layers) {
    // window_size = -1: auto-compute recommended window
    // window_size =  0: SWLP disabled
    // window_size >= 1: explicit window
    if (params.window_size == 0 || params.window_size >= num_layers) {
        enabled = false;
        return;
    }

    bool auto_window = (params.window_size < 0);
    int effective_window = (auto_window && num_layers > 2)
        ? std::max(2, std::min(num_layers - 1, num_layers * 40 / 100))
        : params.window_size;

    if (effective_window <= 0 || effective_window >= num_layers) {
        enabled = false;
        return;
    }

    enabled = true;
    expert_topk_tensors.resize(num_layers, nullptr);
    state = std::make_unique<llama_swlp_state>(
        effective_window, params.prefetch_depth, num_layers);
    state->auto_window = auto_window;
    verbose = params.verbose;

    state->expert_cache_size = params.expert_cache_size;
    state->expert_prefetch_depth = params.expert_prefetch ? params.prefetch_depth : 0;

    if (params.adaptive) {
        state->adaptive_enabled = true;
        state->min_window = 2;
        state->max_window = num_layers - 1;
        state->layer_compute_us.resize(num_layers, 0);
        state->layer_migrate_us.resize(num_layers, 0);

        state->ewma_alpha     = std::clamp(params.ewma_alpha, 0.1f, 1.0f);
        state->alpha_auto     = params.alpha_auto;
        state->adapt_interval = params.adapt_interval > 0 ? params.adapt_interval : 0;

        if (verbose) {
            LLAMA_LOG_INFO("SWLP: adaptive window enabled (min=%d, max=%d, start=%d, alpha=%.2f, alpha_auto=%s)\n",
                state->min_window, state->max_window, state->window_size,
                state->ewma_alpha, state->alpha_auto ? "yes" : "no");
        }
    }
}

llama_swlp::~llama_swlp() = default;

bool llama_swlp::has_prefetch() const {
    return enabled && state->prefetch_depth > 0;
}

bool llama_swlp::need_layer_move(int il) const {
    if (!enabled) return false;
    if (il < 0 || il >= state->num_layers) return false;
    // Fixed GPU layers never need to move
    if (il < state->fixed_gpu_layers || state->layer_fixed_gpu[il]) return false;
    return !state->layer_in_gpu[il];
}

void llama_swlp::prepare_layer(int il) {
    if (!enabled) return;
    if (il < 0 || il >= state->num_layers) return;
    if (!state->backend_migration_ready) return;

    // Fixed GPU layers are permanently resident; just count as cache hit
    if (il < state->fixed_gpu_layers || state->layer_fixed_gpu[il]) {
        state->cache_hits++;
        return;
    }

    bool old_gpu_state = state->layer_in_gpu[il];
    GGML_UNUSED(old_gpu_state);  // used when GGML_USE_CUDA is defined

    if (!state->layer_in_gpu[il]) {
        // Don't set layer_in_gpu here — prepare_migration() is the sole authority.
        // prepare_layer is called from the graph build callback, which runs BEFORE
        // prepare_migration.  Marking layer_in_gpu early would make prepare_migration
        // Phase 2 skip the actual tensor data load.
        // No cache_hit counted here — the layer is not in GPU yet.
    } else {
        state->cache_hits++;
    }

    if (il < state->window_start || il >= state->window_start + state->window_size) {
        // After initial migration, the window is already positioned correctly.
        // No slide needed when the first window layer is accessed.
        if (state->migration_initialized && il == state->window_start) {
            return;
        }

        int old_start = state->window_start;
        state->window_start = std::max(state->fixed_gpu_layers, il - state->window_size + 1);

        // Ensure window doesn't overflow past the last layer
        int max_start = state->num_layers - state->window_size;
        if (state->window_start > max_start) {
            state->window_start = max_start;
        }
        // Ensure window_start is at least fixed_gpu_layers
        if (state->window_start < state->fixed_gpu_layers) {
            state->window_start = state->fixed_gpu_layers;
        }

        for (int i = old_start; i < state->window_start && i < state->num_layers; i++) {
            // Never evict fixed GPU layers
            if (i < state->fixed_gpu_layers || state->layer_fixed_gpu[i]) continue;
            // Don't clear layer_in_gpu here — prepare_migration() is the sole authority
            // for state transitions and data migration. Only invalidate CUDA graphs
            // proactively so stale captured kernels aren't replayed.
#ifdef GGML_USE_CUDA
            if (state->layer_in_gpu[i]) {
                invalidate_layer_graph(i);
            }
#endif
        }

        if (verbose) {
            LLAMA_LOG_INFO("SWLP: window moved to [%d, %d) for layer %d\n",
                state->window_start, state->window_start + state->window_size, il);
        }
    }

#ifdef GGML_USE_CUDA
    if (old_gpu_state != state->layer_in_gpu[il]) {
        invalidate_layer_graph(il);
    }
#endif
}

int llama_swlp::get_window_start() const {
    return enabled ? state->window_start : 0;
}

int llama_swlp::get_window_size() const {
    return enabled ? state->window_size : 0;
}

bool llama_swlp::is_layer_gpu(int il) const {
    if (!enabled) return true;
    if (il < 0 || il >= state->num_layers) return false;
    // Fixed GPU layers are always on GPU
    if (il < state->fixed_gpu_layers || state->layer_fixed_gpu[il]) return true;
    return state->layer_in_gpu[il];
}

void llama_swlp::analyze_model(const llama_model & model) {
    if (!enabled) return;

    state->layer_sizes_bytes.resize(state->num_layers, 0);
    state->total_model_bytes = 0;
    state->non_layer_bytes = 0;

    auto add_tensor = [&](ggml_tensor * t, int layer_idx) {
        if (!t) return;
        size_t nbytes = ggml_nbytes(t);
        if (layer_idx >= 0 && layer_idx < state->num_layers) {
            state->layer_sizes_bytes[layer_idx] += nbytes;
        } else {
            state->non_layer_bytes += nbytes;
        }
        state->total_model_bytes += nbytes;
    };

    for_each_tensor_in_model_non_layer(model, [&](ggml_tensor * t) { add_tensor(t, -1); });

    for (int il = 0; il < state->num_layers; il++) {
        for_each_tensor_in_layer(model.layers[il], [&](ggml_tensor * t) { add_tensor(t, il); });
    }

    for (int il = 0; il < state->num_layers; il++) {
        const auto & layer = model.layers[il];

        ggml_tensor * expert_tensor = layer.ffn_gate_exps
            ? layer.ffn_gate_exps
            : layer.ffn_gate_up_exps;
        if (expert_tensor) {
            int n_experts = (int)expert_tensor->ne[2];

            state->expert_count_per_layer[il] = n_experts;
            state->expert_in_cache[il].resize(n_experts, false);
            state->activation_counts[il].resize(n_experts, 0);
            state->expert_sizes[il].resize(n_experts, 0);

            size_t expert_total = 0;
            if (layer.ffn_gate_exps)    expert_total += ggml_nbytes(layer.ffn_gate_exps);
            if (layer.ffn_gate_up_exps) expert_total += ggml_nbytes(layer.ffn_gate_up_exps);
            if (layer.ffn_down_exps)    expert_total += ggml_nbytes(layer.ffn_down_exps);
            if (layer.ffn_up_exps)      expert_total += ggml_nbytes(layer.ffn_up_exps);

            size_t per_expert = expert_total / n_experts;
            for (int e = 0; e < n_experts; e++) {
                state->expert_sizes[il][e] = per_expert;
            }

            state->is_moe = true;

            if (verbose) {
                LLAMA_LOG_INFO("SWLP: layer %d has %d experts (%.1f MB each)\n",
                    il, n_experts, per_expert / (1024.0*1024.0));
            }
        } else {
            state->expert_count_per_layer[il] = 0;
        }
    }

    // Refine auto-computed window for MoE models (larger per-layer = smaller window)
    if (state->auto_window && state->is_moe) {
        int num_experts = 0;
        for (int il = 0; il < state->num_layers; il++) {
            num_experts = std::max(num_experts, state->expert_count_per_layer[il]);
        }
        // MoE layers are roughly num_experts× larger; reduce window proportionally
        // but not below min_window (2 or 10% of layers)
        if (num_experts >= 8) {
            int moe_window = std::max(2, state->num_layers * 15 / 100);
            state->window_size = std::min(state->window_size, moe_window);
            if (state->adaptive_enabled) {
                state->min_window = 2;
            }
            if (verbose) {
                LLAMA_LOG_INFO("SWLP: MoE model (%d experts), auto window refined to %d\n",
                    num_experts, state->window_size);
            }
        }
    }

    if (verbose) {
        LLAMA_LOG_INFO("SWLP: analyzed model: %zu MB total, %zu MB non-layer, %d layers\n",
            state->total_model_bytes / (1024*1024),
            state->non_layer_bytes / (1024*1024),
            state->num_layers);
    }
}

void llama_swlp::set_backends(ggml_backend_t gpu, ggml_backend_t cpu) {
    state->gpu_backend = gpu;
    state->cpu_backend = cpu;
    state->backend_migration_ready = (gpu != nullptr && cpu != nullptr);

    if (!state->backend_migration_ready && verbose) {
        LLAMA_LOG_WARN("SWLP: no GPU backend available, layer migration disabled\n");
    }

    if (verbose && state->backend_migration_ready) {
        LLAMA_LOG_INFO("SWLP: backend migration ready (GPU backend available)\n");
    }

    // Detect layers already on GPU (pre-loaded by the model loader via ngl).
    // Mark them as fixed: never evict, never re-migrate — avoids double VRAM allocation.
    if (state->backend_migration_ready && state->model_ptr) {
        ggml_backend_buffer_type_t gpu_buft =
            ggml_backend_get_default_buffer_type(state->gpu_backend);

        for (int il = 0; il < state->num_layers; il++) {
            const auto & layer = state->model_ptr->layers[il];
            bool on_gpu = false;
            for_each_tensor_in_layer(layer, [&](ggml_tensor * t) {
                if (!on_gpu && t && t->buffer) {
                    if (ggml_backend_buffer_get_type(t->buffer) == gpu_buft) {
                        on_gpu = true;
                    }
                }
            });
            if (on_gpu) {
                state->layer_fixed_gpu[il] = true;
                state->layer_in_gpu[il] = true;
            }
        }

        if (verbose) {
            int fixed_count = 0;
            for (int il = 0; il < state->num_layers; il++) {
                if (state->layer_fixed_gpu[il]) fixed_count++;
            }
            LLAMA_LOG_INFO("SWLP: %d/%d layers already GPU-resident (model loader), marked as fixed\n",
                fixed_count, state->num_layers);
        }
    }
}

void llama_swlp::set_model(const llama_model * model) {
    state->model_ptr = model;
}

void llama_swlp::set_fixed_gpu_layers(int n_layers) {
    if (!enabled) return;

    // Model loader offloads the LAST n_layers (including output layer) to GPU.
    // The GPU-resident repeating layers are [num_layers - n_layers + 1, num_layers).
    int gpu_start = std::max(0, state->num_layers - n_layers + 1);

    for (int il = gpu_start; il < state->num_layers; il++) {
        state->layer_fixed_gpu[il] = true;
        state->layer_in_gpu[il] = true;
    }

    // Keep fixed_gpu_layers at 0: no prefix layers are GPU-fixed in mixed mode.
    // Existing il < fixed_gpu_layers checks are no-ops; layer_fixed_gpu is authoritative.
    state->fixed_gpu_layers = 0;
}

void llama_swlp::for_each_layer_tensor(int il, const llama_model & model,
        std::function<void(ggml_tensor*)> fn) const {
    if (il < 0 || il >= (int)model.layers.size()) return;
    for_each_tensor_in_layer(model.layers[il], fn);
}

void llama_swlp::annotate_graph(ggml_cgraph * gf, int n_layers) {
    if (!enabled) return;
    if (!gf) return;

    int n_nodes = ggml_graph_n_nodes(gf);

    // Find layer boundary nodes by looking for ffn_norm in node names
    // Each graph node named "ffn_norm-<il>" marks the end of layer il
    std::vector<int> boundaries;
    boundaries.reserve(n_layers);

    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (!node || !node->name || node->name[0] == '\0') continue;

        const char * name = node->name;
        // Use prefix match ("ffn_norm-") to avoid false positives on
        // MoE tensors like "ffn_norm_exps" that contain the substring.
        static const char * marker = "ffn_norm-";
        if (strncmp(name, marker, strlen(marker)) == 0) {
            boundaries.push_back(i);
        }
    }

    state->layer_boundaries = boundaries;

    if (verbose) {
        LLAMA_LOG_INFO("SWLP: annotated graph with %zu layer boundaries (total nodes: %d)\n",
            boundaries.size(), n_nodes);
    }
}

void llama_swlp::annotate_expert_topk(int il, ggml_tensor * topk_tensor) {
    if (!enabled) return;
    if (il < 0 || il >= state->num_layers) return;
    if (topk_tensor) {
        expert_topk_tensors[il] = topk_tensor;
    }
}

void llama_swlp::record_expert_activations() {
    // FIXME: ggml_backend_tensor_get on intermediate graph tensors (ffn_moe_topk)
    // hangs on large MoE models (Phi-mini-MoE). The tensor buffers may not be
    // readable post-compute. Need to switch to a compute-time callback or read
    // from output tensors instead. See git history for the disabled implementation.
}

void llama_swlp::on_split_begin(int split_idx) {
    if (!enabled) return;
    // Reserved for future timing/profiling hooks.
    // Migration is performed in prepare_migration() before alloc_graph.
    GGML_UNUSED(split_idx);
}


void llama_swlp::set_async_migration(bool enable) {
    if (!enabled) return;
    state->async_migration_enabled = enable;
    if (verbose) {
        LLAMA_LOG_INFO("SWLP: async migration %s\n", enable ? "enabled" : "disabled");
    }
}

bool llama_swlp::is_async_migration() const {
    return enabled && state->async_migration_enabled;
}

int64_t llama_swlp::get_last_migration_us() const {
    return enabled ? state->last_migration_us : 0;
}

int64_t llama_swlp::get_last_compute_us() const {
    return enabled ? state->last_compute_us : 0;
}

bool llama_swlp::has_pending_window_change() const {
    if (!enabled) return false;
    return state->window_size != state->last_migrated_window;
}

void llama_swlp::apply_pending_window() {
    if (!has_pending_window_change()) return;
    prepare_migration();
}

#ifdef GGML_USE_CUDA

void llama_swlp::enable_cuda_graphs() {
    if (!enabled) return;
    if (cuda_graphs) return;

    cuda_graphs = std::make_unique<llama_swlp_cuda_graphs>(state->num_layers);

    if (verbose) {
        LLAMA_LOG_INFO("SWLP: CUDA per-layer graphs enabled (%d layers)\n",
            state->num_layers);
    }
}

bool llama_swlp::begin_capture_layer(int il, void * stream) {
    if (!enabled || !cuda_graphs) return false;
    return cuda_graphs->begin_capture(il, stream);
}

bool llama_swlp::end_capture_layer(int il, void * stream) {
    if (!enabled || !cuda_graphs) return false;
    return cuda_graphs->end_capture(il, stream);
}

bool llama_swlp::replay_layer_graph(int il, void * stream) const {
    if (!enabled || !cuda_graphs) return false;
    return cuda_graphs->replay_layer_graph(il, stream);
}

void llama_swlp::invalidate_layer_graph(int il) {
    if (!enabled || !cuda_graphs) return;
    cuda_graphs->invalidate_graph(il);

    if (verbose) {
        LLAMA_LOG_DEBUG("SWLP: CUDA graph for layer %d invalidated\n", il);
    }
}

bool llama_swlp::is_layer_graph_valid(int il) const {
    if (!enabled || !cuda_graphs) return true;
    return cuda_graphs->is_graph_valid(il);
}

void llama_swlp::print_cuda_graph_stats() const {
    if (cuda_graphs) {
        cuda_graphs->print_stats();
    }
}

#endif // GGML_USE_CUDA
