#include "llama-swlp.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-swlp-tensors.h"
#include "llama-swlp-internal.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

// === Static helpers: layer migration between CPU and GPU ===

static bool migrate_layer_to_gpu(
    int il,
    const llama_model & model,
    ggml_backend_t gpu_backend,
    std::vector<void*> & saved_tensor_data,
    std::vector<ggml_backend_buffer_t> & saved_tensor_buffers,
    ggml_backend_buffer_t & out_gpu_buffer,
    bool verbose,
    bool async,
    void ** out_migration_event,
    ggml_backend_buffer_t & out_host_buffer
) {
    if (il < 0 || il >= (int)model.layers.size()) return false;
    if (!gpu_backend) return false;

    // Compute total bytes needed for this layer on GPU.
    // Use the GPU buffer type, and include inter-tensor alignment padding.
    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_backend);
    size_t gpu_alignment = ggml_backend_buft_get_alignment(gpu_buft);

    size_t total_bytes = 0;
    size_t total_host_bytes = 0;  // sum of ggml_nbytes (no padding needed on host side)
    int tensor_count = 0;
    for_each_tensor_in_layer(model.layers[il], [&](ggml_tensor * t) {
        if (t && t->buffer) {
            total_bytes = GGML_PAD(total_bytes, gpu_alignment);
            total_bytes += ggml_backend_buft_get_alloc_size(gpu_buft, t);
            total_host_bytes += ggml_nbytes(t);
            tensor_count++;
        }
    });

    if (total_bytes == 0 || tensor_count == 0) return false;

    // Allocate a single GPU buffer for all tensors in this layer
    out_gpu_buffer = ggml_backend_buft_alloc_buffer(gpu_buft, total_bytes);
    if (!out_gpu_buffer) {
        LLAMA_LOG_WARN("SWLP: failed to allocate %.1f MB GPU buffer for layer %d\n",
            total_bytes / (1024.0*1024.0), il);
        return false;
    }

    void * gpu_base = ggml_backend_buffer_get_base(out_gpu_buffer);
    size_t offset = 0;
    size_t alignment = ggml_backend_buft_get_alignment(gpu_buft);

    saved_tensor_data.clear();
    saved_tensor_buffers.clear();

    // For async migration: allocate a pinned (page-locked) host buffer once and
    // stage all tensor data into it. cudaMemcpyAsync requires page-locked source.
    // The buffer is freed after the migration event is consumed on the compute stream.
    void * host_base = nullptr;
    size_t host_offset = 0;
#ifdef GGML_USE_CUDA
    if (async) {
        ggml_backend_buffer_type_t host_buft = ggml_backend_cuda_host_buffer_type();
        out_host_buffer = ggml_backend_buft_alloc_buffer(host_buft, total_host_bytes);
        if (out_host_buffer) {
            host_base = ggml_backend_buffer_get_base(out_host_buffer);
        }
        // If allocation fails, fall through to synchronous path below
    }
#endif

    size_t copied_bytes = 0;
    int copied_count = 0;

    for_each_tensor_in_layer(model.layers[il], [&](ggml_tensor * t) {
        if (!t || !t->buffer) return;

        size_t nbytes = ggml_nbytes(t);
        size_t alloc_size = ggml_backend_buft_get_alloc_size(gpu_buft, t);

        // Align offset
        offset = GGML_PAD(offset, alignment);

        // Save original CPU-side pointers
        saved_tensor_data.push_back(t->data);
        saved_tensor_buffers.push_back(t->buffer);

        // Read data from CPU into staging memory BEFORE swapping the buffer.
        // After swap, ggml_backend_tensor_get would target the (empty) GPU buffer.
#ifdef GGML_USE_CUDA
        if (async && host_base) {
            // Stage into pinned host buffer (page-locked, safe for cudaMemcpyAsync)
            ggml_backend_tensor_get(t, (char*)host_base + host_offset, 0, nbytes);
        } else
#endif
        {
            // Synchronous: use stack buffer
            std::vector<uint8_t> host_buf(nbytes);
            ggml_backend_tensor_get(t, host_buf.data(), 0, nbytes);

            // Reassign tensor to GPU buffer
            t->buffer = out_gpu_buffer;
            t->data = (char*)gpu_base + offset;
            ggml_backend_buffer_init_tensor(out_gpu_buffer, t);

            // Write data to GPU
            ggml_backend_tensor_set(t, host_buf.data(), 0, nbytes);

            copied_bytes += nbytes;
            copied_count++;
            offset += alloc_size;
            return;
        }

        // Async path: swap buffer AFTER reading from CPU
        t->buffer = out_gpu_buffer;
        t->data = (char*)gpu_base + offset;
        ggml_backend_buffer_init_tensor(out_gpu_buffer, t);

        // Async H2D: pinned buffer -> GPU on migration stream (returns immediately)
#ifdef GGML_USE_CUDA
        ggml_backend_cuda_set_tensor_migrate_async(gpu_backend, t,
            (char*)host_base + host_offset, 0, nbytes);
        host_offset += nbytes;
#endif

        copied_bytes += nbytes;
        copied_count++;
        offset += alloc_size;
    });

    // Record migration event for async pipelining
#ifdef GGML_USE_CUDA
    if (async && out_migration_event && copied_count > 0) {
        if (*out_migration_event == nullptr) {
            *out_migration_event = ggml_backend_cuda_migrate_event_create(gpu_backend);
        }
        ggml_backend_cuda_migrate_event_record(gpu_backend, *out_migration_event);
    }
#endif

    if (verbose && copied_count > 0) {
        LLAMA_LOG_INFO("SWLP: layer %d -> GPU: %d tensors (%.1f MB)%s\n",
            il, copied_count, copied_bytes / (1024.0 * 1024.0),
            async ? " [async]" : "");
    }

    return true;
}

// Restore a layer's tensors back to their original CPU buffers.
// Frees the GPU buffer.
static void restore_layer_to_cpu(
    int il,
    const llama_model & model,
    const std::vector<void*> & saved_tensor_data,
    const std::vector<ggml_backend_buffer_t> & saved_tensor_buffers,
    ggml_backend_buffer_t gpu_buffer,
    bool verbose
) {
    if (il < 0 || il >= (int)model.layers.size()) return;

    size_t idx = 0;
    size_t restored_bytes = 0;
    int restored_count = 0;

    for_each_tensor_in_layer(model.layers[il], [&](ggml_tensor * t) {
        if (!t || !t->buffer) return;
        if (idx >= saved_tensor_data.size()) return;

        size_t nbytes = ggml_nbytes(t);

        // Read data from GPU buffer before restoring (in case it was modified)
        std::vector<uint8_t> host_buf(nbytes);
        ggml_backend_tensor_get(t, host_buf.data(), 0, nbytes);

        // Restore original CPU buffer and data pointer
        t->buffer = saved_tensor_buffers[idx];
        t->data = saved_tensor_data[idx];
        ggml_backend_buffer_init_tensor(t->buffer, t);

        // Write any modified data back to CPU buffer
        ggml_backend_tensor_set(t, host_buf.data(), 0, nbytes);

        restored_bytes += nbytes;
        restored_count++;
        idx++;
    });

    // Free the GPU buffer
    if (gpu_buffer) {
        ggml_backend_buffer_free(gpu_buffer);
    }

    if (verbose && restored_count > 0) {
        LLAMA_LOG_INFO("SWLP: layer %d -> CPU: %d tensors restored (%.1f MB)\n",
            il, restored_count, restored_bytes / (1024.0 * 1024.0));
    }
}

// === Private method implementations ===

void llama_swlp::migrate_evict_layer(int il) {
    if (!state->model_ptr) return;
    auto & lg = state->layer_gpu[il];

#ifdef GGML_USE_CUDA
    if (state->migration_events[il]) {
        ggml_backend_cuda_wait_migration_event(state->gpu_backend,
            state->migration_events[il]);
        ggml_backend_cuda_migrate_event_destroy(state->migration_events[il]);
        state->migration_events[il] = nullptr;
    }
#endif

    if (lg.migration_host_buf) {
        ggml_backend_buffer_free(lg.migration_host_buf);
        lg.migration_host_buf = nullptr;
    }

    restore_layer_to_cpu(il, *state->model_ptr,
        lg.saved_data, lg.saved_buffers,
        lg.gpu_buffer, verbose);
    lg.gpu_buffer = nullptr;
    lg.saved_data.clear();
    lg.saved_buffers.clear();

    state->layer_in_gpu[il] = false;
    state->total_migrations++;
    state->total_migrated_bytes += state->layer_sizes_bytes[il];
    state->copy_count++;
}

void llama_swlp::migrate_load_layer(int il, bool use_async) {
    if (!state->model_ptr) return;
    auto & lg = state->layer_gpu[il];

    bool ok = migrate_layer_to_gpu(il, *state->model_ptr,
        state->gpu_backend,
        lg.saved_data, lg.saved_buffers,
        lg.gpu_buffer, verbose,
        use_async,
        use_async ? &state->migration_events[il] : nullptr,
        lg.migration_host_buf);
    if (!ok) {
        LLAMA_LOG_WARN("SWLP: failed to migrate layer %d to GPU, keeping on CPU\n", il);
        return;
    }

    state->layer_in_gpu[il] = true;
    state->total_migrations++;
    state->total_migrated_bytes += state->layer_sizes_bytes[il];
    state->copy_count++;
}

void llama_swlp::migrate_evict_range(int start, int end, int w_start, int pf_end) {
    for (int il = start; il < end; il++) {
        if (state->layer_fixed_gpu[il]) continue;
        bool should_be_gpu = (il >= w_start && il < pf_end);
        if (!should_be_gpu && state->layer_in_gpu[il]) {
            migrate_evict_layer(il);
        }
    }
}

void llama_swlp::migrate_load_range(int start, int end, int w_start, int pf_end) {
    for (int il = start; il < end; il++) {
        if (state->layer_fixed_gpu[il]) continue;
        bool should_be_gpu = (il >= w_start && il < pf_end);
        if (should_be_gpu && !state->layer_in_gpu[il]) {
            migrate_load_layer(il, state->async_migration_enabled);
        }
    }
}

void llama_swlp::prepare_migration() {
    if (!enabled) return;
    if (!state->backend_migration_ready) return;

    int64_t t0 = ggml_time_us();

    int w_start = state->window_start;
    int w_end = w_start + state->window_size;
    int pf_end = w_end + state->prefetch_depth;

    // Clamp to valid range
    if (w_end > state->num_layers)    w_end = state->num_layers;
    if (pf_end > state->num_layers)   pf_end = state->num_layers;
    if (w_start < 0)                  w_start = 0;

    const bool is_first_migration = !state->migration_initialized;
    const int  evict_start  = std::max(state->fixed_gpu_layers, is_first_migration ? state->fixed_gpu_layers : state->prev_window_start);
    const int  evict_end    = std::min(state->num_layers,        is_first_migration ? state->num_layers        : state->prev_prefetch_end);
    const int  load_start   = std::max(state->fixed_gpu_layers, w_start);
    const int  load_end     = std::min(state->num_layers,        pf_end);

    // Evict layers that should no longer be on GPU
    migrate_evict_range(evict_start, evict_end, w_start, pf_end);

    // Load layers that should now be on GPU
    migrate_load_range(load_start, load_end, w_start, pf_end);

    // Save current window for next incremental migration
    state->prev_window_start = w_start;
    state->prev_window_end   = w_end;
    state->prev_prefetch_end = pf_end;
    state->migration_initialized   = true;
    state->last_migrated_window    = w_end - w_start;

    if (verbose) {
        int gpu_count = 0;
        int evict_count = 0;
        int load_count = 0;
        for (int il = 0; il < state->num_layers; il++) {
            if (state->layer_in_gpu[il]) gpu_count++;
        }
        // Count migrations in this call by comparing prev vs current GPU state
        for (int il = evict_start; il < evict_end; il++) {
            if (il >= state->fixed_gpu_layers && !state->layer_fixed_gpu[il]) {
                if (il >= w_start && il < pf_end) continue; // still in window
                if (state->layer_in_gpu[il]) { /* would be evicted */ evict_count++; }
            }
        }
        for (int il = load_start; il < load_end; il++) {
            if (il >= state->fixed_gpu_layers && !state->layer_fixed_gpu[il]) {
                if (!state->layer_in_gpu[il]) { /* would be loaded */ load_count++; }
            }
        }
        if (evict_count > 0 || load_count > 0) {
            LLAMA_LOG_INFO("SWLP: window [%d,%d) pf+%d, migrate %d evict + %d load = %d layers (%s, %.1f ms)\n",
                w_start, w_end, state->prefetch_depth,
                evict_count, load_count, evict_count + load_count,
                is_first_migration ? "full" : "incremental",
                state->last_migration_us / 1000.0);
        } else {
            LLAMA_LOG_INFO("SWLP: window [%d,%d) pf+%d, %d/%d layers on GPU (%s, %.1f ms, no migration needed)\n",
                w_start, w_end, state->prefetch_depth, gpu_count, state->num_layers,
                is_first_migration ? "full" : "incremental",
                state->last_migration_us / 1000.0);
        }
    }

    state->last_migration_us = ggml_time_us() - t0;
    state->cumulative_migrate_us += state->last_migration_us;
}

void llama_swlp::ensure_window_ready() {
    if (!enabled) return;
    if (!state->async_migration_enabled) return;
    if (!state->gpu_backend) return;

    // Wait for ALL pending migration events on the compute stream.
    // This covers both window layers and prefetch layers (which may have
    // been loaded asynchronously but not yet entered the window).
    // After waiting, free the corresponding pinned host buffer.
    for (int il = 0; il < state->num_layers; il++) {
        if (il < state->fixed_gpu_layers || state->layer_fixed_gpu[il]) continue;

#ifdef GGML_USE_CUDA
        void * event = state->migration_events[il];
        if (event) {
            ggml_backend_cuda_wait_migration_event(state->gpu_backend, event);
            ggml_backend_cuda_migrate_event_destroy(event);
            state->migration_events[il] = nullptr;
        }
#endif

        // Free the pinned host buffer used for async H2D staging.
        // The data has been copied to GPU; the host buffer is no longer needed.
        if (state->layer_gpu[il].migration_host_buf) {
            ggml_backend_buffer_free(state->layer_gpu[il].migration_host_buf);
            state->layer_gpu[il].migration_host_buf = nullptr;
        }
    }
}
