# SWLP C++ Source Code Review

Review of 6 files: `llama-swlp.h`, `llama-swlp.cpp`, `llama-swlp-tensors.h`, `llama-context.cpp` (SWLP portions), `llama-graph.cpp` (SWLP portions), `llama-model.cpp` (SWLP portions). Plus supporting: `llama-swlp-cuda.h`, `llama-swlp-cuda.cpp`, `ggml-cuda.h`, `ggml-cuda.cu`.

---

## Review Summary

| Severity | Count |
|----------|:-----:|
| Correct | 8 |
| Fixable issue | 7 |
| Blocker | 0 |
| Note / Observation | 6 |

No blockers. The code is functionally correct for the non-async, non-expert-cache common path. Issues are concentrated in the async migration path, dead/unused code, and the expert cache which is currently non-functional.

---

## Correct

### C1 — `prepare_layer` / `prepare_migration` ordering (src/llama-swlp.cpp:210–273)
The original design bug (C1 from the plan) has been correctly implemented: `prepare_layer` is read-only for `layer_in_gpu` — it never sets `layer_in_gpu[il] = true`. That transition is reserved exclusively for `prepare_migration()`. The comment at line 225–227 documents this intent clearly.

### C2 — Window slide logic (src/llama-swlp.cpp:246–273)
`prepare_layer` slides `window_start` when the accessed layer `il` falls outside `[window_start, window_start + window_size)`. The slide correctly:
- Clamps to `[fixed_gpu_layers, num_layers - window_size]`
- Invalidates CUDA graphs for evicted layers (when `GGML_USE_CUDA`)
- Does NOT mutate `layer_in_gpu` (defers to `prepare_migration`)

### C3 — Incremental migration in `prepare_migration` (src/llama-swlp.cpp:849–1000)
The two-phase (full/incremental) migration logic is correct:
- First migration: full sweep evict + load across all layers
- Subsequent: delta-only evict old-exclusive, load new-only
- `prev_window_start/prev_window_end/prev_prefetch_end` are correctly saved after each migration
- `layer_fixed_gpu` layers are always skipped

### C4 — Async migration event lifecycle (ggml-cuda.cu:3298–3365)
The CUDA event API is correct:
- `cudaEventCreateWithFlags(..., cudaEventDisableTiming)` for lightweight synchronization
- `cudaMemcpyAsync` on dedicated migration stream
- `cudaEventRecord` on migration stream, `cudaStreamWaitEvent` on compute stream
- Destruction in `ensure_window_ready` after wait
- Migration stream is `cudaStreamNonBlocking` (common.cuh:1477–1478)

### C5 — Migration stream cleanup (ggml-cuda.cu:619–622)
`migration_streams[i]` are destroyed in the backend destructor alongside compute streams and cuBLAS handles. No leak.

### C6 — `restore_layer_to_cpu` GPU buffer free (src/llama-swlp.cpp:633–670)
`ggml_backend_buffer_free(gpu_buffer)` inside the function, followed by caller setting `lg.gpu_buffer = nullptr`. No double-free.

### C7 — Window rebuild on adaptive resize (src/llama-context.cpp:1333–1337)
`swlp_force_rebuild` correctly forces graph rebuild when `has_pending_window_change()`. The rebuild path calls `prepare_migration()` inside `build_graph`, so the new window size takes effect. The empty `if (swlp_force_rebuild)` block with comment is intentional documentation.

### C8 — Tensor list completeness (src/llama-swlp-tensors.h)
The template iterates over every tensor field in `llama_layer`. The file is comprehensive — every plausible tensor category is covered (norms, attention weights, FFN, MoE, Mamba, RWKV, BitNet, PosNet, ConvNeXt, ShortConv, NextN, etc.). The comment warning about staying in sync is appropriate.

---

## Fixable Issues

### F1 — `pending_migrations` vector is never written (src/llama-swlp.cpp:66, 1263–1264)
**Severity**: Low (dead code)  
**Evidence**: `pending_migrations` is defined as `std::vector<migration_record>` with struct fields `layer_idx`, `bytes`, `to_gpu`, `tensors`. It is read in `get_migration_records()` via `result.reserve(state->pending_migrations.size())` and looped. But NO code path ever pushes to `pending_migrations`. The `migrate_layer_to_gpu` function writes to `saved_tensor_data/saved_tensor_buffers` but never records a `migration_record`. The function `get_migration_records()` always returns an empty vector.

**Fix**: Either (a) remove `pending_migrations`, `get_migration_records()`, and `migration_record` from state/class, or (b) add push_back calls in the migration success paths in `prepare_migration` after each `migrate_layer_to_gpu`/`restore_layer_to_cpu` call.

### F2 — `migrate_tensor_data` static function is unused (src/llama-swlp.cpp:500–525)
**Severity**: Low (dead code)  
**Evidence**: `migrate_tensor_data` is defined as a static function with parameters `(tensor, src_data, src_nbytes, verbose)`. It is never called anywhere in the file. The actual migration uses `migrate_layer_to_gpu` and `restore_layer_to_cpu` which directly call `ggml_backend_tensor_set` / `ggml_backend_tensor_get`.

**Fix**: Remove the dead function.

### F3 — Prefetch zone async events not waited before eviction (src/llama-swlp.cpp:849–1000 + 1035–1065)
**Severity**: Medium  
**Description**: Consider this sequence:
1. Token 1: `prepare_migration` migrates layers [0, 6) (window=[0,4), prefetch=[4,6)). Async H2D copies for layers 4,5 are submitted on the migration stream. Events recorded for all 6 layers.
2. `ensure_window_ready` waits on events for window layers [0, 4) only. Events for layers 4,5 (prefetch zone) are **not** waited on and **not** destroyed.
3. Token 1 compute runs.
4. Token 2: window slides to [2, 6). `prepare_migration` runs incremental. Layers 4,5 are now in the window. The old events (from step 1) for layers 4,5 still exist and are waited on in `ensure_window_ready` — this is fine.

**But** consider a more aggressive scenario:
1. Token 1: migrate layers [0, 8) (window=[0,4), prefetch=[4,8)). Async events for layers 4-7.
2. Token 2: window slides to [4, 8). `prepare_migration` tries to evict layers 0-3. OK.
   — But what if Token 3 slides further to [6, 10)? Layers 4,5 need eviction. Their events from Token 1 still exist. `ensure_window_ready` was called for [4, 8) window but only waited on events for [4, 8)... wait, Token 2's `ensure_window_ready` waits on [4, 8) which includes 4,5. So the events are waited on and destroyed.

The real concern is more narrow: what if a layer was prefetched (H2D submitted async on migration stream, event NOT waited on) and then the window slides such that the layer is evicted (via `restore_layer_to_cpu` which reads GPU data synchronously on the compute stream) before the H2D copy completes? The read would get garbage data.

This can happen when:
- Migration stream H2D is still in flight (large tensors, e.g., 1GB at 12GB/s = 83ms)
- Next token's compute is fast (<20ms)
- Window slides far enough to evict the still-being-copied layer

**Fix**: In `ensure_window_ready`, iterate over ALL layers that have migration events, not just the current window. Or, at the start of each `prepare_migration` call, synchronize the migration stream (wait for all pending events first).

### F4 — Expert cache is non-functional (src/llama-swlp.cpp:1195–1218, 844–851, 1237–1261)
**Severity**: Medium  
**Evidence**:
- `record_expert_activations()` at line 844 is a no-op (early `return;` with FIXME comment at 845-846)
- Therefore `activation_counts[il]` are always all-zero
- `predict_experts` returns the first `top_k` experts sorted by activation (all 0, so first N experts are returned)
- `ensure_experts_cached` manages boolean flags but never migrates actual tensor data
- `prepare_layer_experts` now runs the full predict+ensure path (unlike the original c09eb8b1e which had a no-op stub with a FIXME)

The function `prepare_layer_experts` at line 1237 runs on every graph-build callback call (per layer, per token). It calls `predict_experts` (O(n_experts log n_experts) sort) + `ensure_experts_cached` (linear scan). This imposes measurable CPU overhead on MoE models with no benefit.

**Fix paths**:
- (a) Restore the early-return `#if 0` guard that was in c09eb8b1e, so `prepare_layer_experts` is a no-op until the expert migration infrastructure is implemented.
- (b) Implement the full expert pipeline: activation tracking via a compute-time hook, per-expert tensor splitting, and GPU/CPU migration.

### F5 — `set_fixed_gpu_layers` method exists but is never called (src/llama-swlp.cpp:474–490, src/llama-swlp.h:65)
**Severity**: Low (dead code)  
**Evidence**: The method is declared and defined. `llama-context.cpp` line 113 has a comment "When upstream is fixed, enable: swlp->set_fixed_gpu_layers(ngl);" but it's behind a disabled FIXME. The `set_backends` function (lines 413–468) already detects GPU-resident layers by checking buffer types, which is more robust.

The method also has `state->fixed_gpu_layers = 0` at the end, which is correct because `layer_fixed_gpu` is the authoritative source. But the loop sets `state->layer_fixed_gpu[il] = true` for the tail layers, and `state->layer_in_gpu[il] = true`. The `state->fixed_gpu_layers` field is zeroed, meaning the `il < state->fixed_gpu_layers` guard in `prepare_layer` (line 218) will never match. This is correct because `layer_fixed_gpu[il]` is the real guard.

**Fix**: Either remove the method (per plan item B1), or keep it as a safety net for when the upstream partial-offload bug is fixed.

### F6 — `layer_needed` vector allocated but never accessed (src/llama-swlp.cpp:30, constructor at ~132)
**Severity**: Low (dead memory)  
**Evidence**: `std::vector<bool> layer_needed` is initialized to `nl, false` and stored. It is never written (no `state->layer_needed[il] = ...` anywhere) and never read. The original A1 fix from the plan suggested using it in `prepare_layer`, but the actual fix chose to not mutate any state in `prepare_layer` at all. The vector is dead.

**Fix**: Remove `layer_needed` from `llama_swlp_state` and its constructor initializer.

### F7 — `prepare_layer` increments `cache_hits` even when the layer is NOT on GPU (src/llama-swlp.cpp:231, 234)
**Severity**: Low (cosmetic/metrics noise)  
**Evidence**: Both branches of `if (!state->layer_in_gpu[il])` increment `cache_hits`. The "miss" case where the layer needs to be loaded is counted as a "hit". This inflates the `cache_hits` counter and misrepresents the metric.

**Fix**:
```cpp
if (!state->layer_in_gpu[il]) {
    // layer not on GPU, marked as needed — prepare_migration will handle load
    // (no counter increment — this is not a cache hit)
} else {
    state->cache_hits++;
}
```

---

## Notes / Observations

### N1 — Window size auto-compute for MoE models (src/llama-swlp.cpp:367–383)
The auto window refinement for MoE models reduces the window to `num_layers * 15 / 100` (with min 2). This is heuristic-based and could be overly conservative for wide MoE models (e.g., 8 experts on small layers). The comment at line 375 is transparent about this being a "rough" adjustment.

### N2 — Async migration CPU-side buffer allocation (src/llama-swlp.cpp:561)
In `migrate_layer_to_gpu`, for each tensor, a `std::vector<uint8_t> host_buf(nbytes)` is allocated on the heap to hold the CPU data before copying to GPU. For large layers (~1GB), this is a temporary allocation of ~1GB. In the non-async path, this is a synchronous copy via `ggml_backend_tensor_set`. In the async path, the vector is kept alive until the async copy completes (it goes out of scope after `ggml_backend_cuda_set_tensor_migrate_async` returns, but CUDA captures the pointer and the copy is asynchronous). If the vector is destroyed before the copy completes, the data pointer becomes dangling.

**Risk**: HIGH. `std::vector<uint8_t> host_buf(nbytes)` goes out of scope at the end of the lambda iteration. The async copy `ggml_backend_cuda_set_tensor_migrate_async` takes `const void * data` which is `host_buf.data()`. If the H2D copy hasn't started when the vector is destroyed, the copy reads garbage.

Wait — the CUDA runtime generally submits the copy immediately on `cudaMemcpyAsync`, and the function returns once the copy is queued (not completed). The data must remain valid until the copy completes. The `host_buf` vector is destroyed at the end of the lambda, which is BEFORE the copy completes.

**This is a potential use-after-free / data corruption bug in the async migration path.**

Actually, let me re-read the code more carefully:

```cpp
for_each_tensor_in_layer(model.layers[il], [&](ggml_tensor * t) {
    if (!t || !t->buffer) return;

    size_t nbytes = ggml_nbytes(t);
    size_t alloc_size = ggml_backend_buft_get_alloc_size(gpu_buft, t);

    offset = GGML_PAD(offset, alignment);

    saved_tensor_data.push_back(t->data);
    saved_tensor_buffers.push_back(t->buffer);

    std::vector<uint8_t> host_buf(nbytes);         // LOCAL vector
    ggml_backend_tensor_get(t, host_buf.data(), 0, nbytes);

    t->buffer = out_gpu_buffer;
    t->data = (char*)gpu_base + offset;
    ggml_backend_buffer_init_tensor(out_gpu_buffer, t);

#ifdef GGML_USE_CUDA
    if (async) {
        ggml_backend_cuda_set_tensor_migrate_async(gpu_backend, t, host_buf.data(), 0, nbytes);
    } else
#endif
    {
        ggml_backend_tensor_set(t, host_buf.data(), 0, nbytes);
    }

    copied_bytes += nbytes;
    copied_count++;
    offset += alloc_size;
});  // host_buf DESTROYED here!
```

In the SYNC path: `ggml_backend_tensor_set` blocks until the copy completes. The `host_buf` is still valid during the call. Safe.

In the ASYNC path: `ggml_backend_cuda_set_tensor_migrate_async` submits `cudaMemcpyAsync` to the migration stream. The call returns once the copy is queued, NOT completed. Then `host_buf` is destroyed at the end of the lambda iteration. The CUDA copy may still be in-flight, reading from freed memory.

**This IS a real bug in the async migration path.** The `host_buf` vector must remain valid until the H2D copy completes.

**Fix options:**
1. Make `host_buf` persistent (e.g., save in `saved_tensor_data` or a new structure) and only free after the event is waited on.
2. Use a single pre-allocated staging buffer for all tensors in the layer.
3. Use pinned memory allocated via `ggml_backend_cuda_host_buffer_type()` for the staging buffer (faster copies + always valid).
4. Fall back to synchronous copy for async path (defeats the purpose but is safe).

This is the most serious issue in the codebase.

### N3 — `on_split_begin` is a no-op (src/llama-swlp.cpp:844–851)
The SWLP callback is registered and called per-split, but the handler is empty. The comment says "Migration is now performed in prepare_migration(). This callback is retained for future timing/profiling hooks." This is fine but adds per-split function call overhead.

### N4 — `estimate_active_memory_mb` and `estimate_total_memory_mb` use integer division (src/llama-swlp.cpp:700–710)
`return active_bytes / (1024 * 1024)` — truncates to MB. For `estimate_memory_reduction_ratio`, the float ratio is correct. Minor: `active_bytes` is computed from `non_layer_bytes + layer_sizes_bytes` for GPU-resident layers, but `layer_sizes_bytes` is the full layer's tensor byte sum, not accounting for GPU-specific buffer alignment padding. The estimate is approximate but reasonable for reporting.

### N5 — `llama_swlp_state` destructor cleanup order (src/llama-swlp.cpp:145–155)
The destructor iterates `num_layers` and frees `gpu_buffer` and destroys `migration_events`. However, if a `gpu_buffer` is non-null (meaning the layer has a GPU allocation), its `saved_data`/`saved_buffers` are still holding pointers to the original CPU buffers. The destructor frees the GPU buffer but does NOT restore tensor pointers to CPU buffers. If the context is destroyed while any tensors still reference the freed GPU buffer, this is a use-after-free.

In normal shutdown flow: `llama_context` destructor runs, which destroys backends, which frees all buffers, and then SWLP state is destroyed. The GPU buffer freed in the SWLP destructor may double-free if the backend has already freed it. However, the SWLP GPU buffer was allocated via `ggml_backend_buft_alloc_buffer`, which registers it with the backend. When the backend is destroyed, it may attempt to free all its buffers, including the SWLP ones.

**Risk**: Medium. Depends on backend destruction order. If the backend destructor frees all registered buffers before SWLP destruction, the SWLP destructor's `ggml_backend_buffer_free` will be a double-free.

Actually, on closer inspection: `ggml_backend_buft_alloc_buffer` returns a buffer handle, and `ggml_backend_buffer_free` is the correct way to free it. If the backend destructor frees all its buffers, it should know about this buffer. But SWLP allocated it independently of the scheduler's graph allocation. The backend's buffer tracking is separate from the scheduler's.

To be safe, the SWLP destructor should nullify the `gpu_buffer` pointers and skip the `ggml_backend_buffer_free` calls if the backend is already being destroyed. But there's no API to check this.

**Practical mitigation**: This is only triggered during context destruction, which implies process exit or full teardown. The OS will reclaim GPU memory regardless. Low priority.

### N6 — `ggml_backend_cuda_wait_migration_event` destroys event but `ggml_cuda_set_device` not called first (ggml-cuda.cu:3357–3361)
```cpp
void ggml_backend_cuda_wait_migration_event(ggml_backend_t backend, void * event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t) event, 0));
}
```
This does `cudaStreamWaitEvent` but does NOT set the CUDA device first. The `cudaStreamWaitEvent` call uses the stream handle, which is device-specific. So no `cudaSetDevice` is needed — the stream handle carries the device association. This is correct for CUDA driver API behavior.

---

## Code Quality / Consistency

### Naming conventions
- `maybe_print_swlp_stats` — renamed per plan, consistent with existing `maybe_` pattern in codebase.
- Class names use `llama_` prefix, matching project convention: `llama_swlp`, `llama_swlp_state`, `llama_swlp_cuda_graphs`.
- Member variables use `snake_case` (e.g., `window_size`, `num_layers`). Consistent.
- `GGML_USE_CUDA` guards are used correctly — all CUDA-specific code is `#ifdef`-guarded.

### Comment style
Comments are concise and explain non-obvious invariants. Examples:
- "Don't set layer_in_gpu here — prepare_migration() is the sole authority" (llama-swlp.cpp:225)
- "After initial migration, the window is already positioned correctly." (llama-swlp.cpp:249)
- "Migration is now performed in prepare_migration() before alloc_graph." (llama-swlp.cpp:847)

No excessive inline commentary. Consistent with project guidelines.

### Error handling
- Failed `migrate_layer_to_gpu` logs a warning and continues (skip that layer)
- `restore_layer_to_cpu` has no error return; failures propagate as assertions/aborts from ggml
- No exception handling (consistent with project's C-style error management)

---

## Files Not Modified (No SWLP content)

- `src/llama-model.cpp` — zero SWLP references. No changes needed.
- `src/llama-graph.cpp` — single `#include "llama-swlp.h"` include directive only. No SWLP logic.
- `src/llama-context.h` — declares `swlp_params`, `swlp` unique_ptr, and `maybe_print_swlp_stats()`. Clean.

---

## Summary of Priority Actions

| Priority | Issue | File | Risk |
|:--------:|-------|------|:----:|
| 🔴 P0 | Async path: `host_buf` destroyed before H2D copy completes (N2) | `llama-swlp.cpp:561-595` | Use-after-free, data corruption |
| 🟡 P1 | Prefetch events not synchronized before eviction (F3) | `llama-swlp.cpp:857-1000` | Data corruption on eviction |
| 🟡 P2 | Expert cache no-op wastes CPU (F4) | `llama-swlp.cpp:1237-1261` | Performance regression on MoE |
| 🟢 P3 | `pending_migrations`/`migrate_tensor_data` dead code (F1, F2) | `llama-swlp.cpp:66,500,1263` | Dead code, maintainability |
| 🟢 P4 | `cache_hits` inflated (F7) | `llama-swlp.cpp:231,234` | Misleading metrics |
| 🟢 P5 | `layer_needed` dead allocation (F6) | `llama-swlp.cpp:30` | Wasted memory |

---

## Files Reviewed

| File | Lines | SWLP-specific |
|------|-------|:-------------:|
| `src/llama-swlp.h` | 105 | 105 |
| `src/llama-swlp.cpp` | ~1340 | ~1340 |
| `src/llama-swlp-tensors.h` | 189 | 189 |
| `src/llama-context.cpp` | ~4080 | ~140 |
| `src/llama-context.h` | ~320 | 5 |
| `src/llama-graph.cpp` | ~3145 | 1 (include) |
| `src/llama-model.cpp` | ~N/A | 0 |
| `src/llama-swlp-cuda.h` | ~55 | 55 |
| `src/llama-swlp-cuda.cpp` | ~330 | 330 |
| `ggml/include/ggml-cuda.h` | ~60 | ~15 |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | ~5800 | ~70 |
| `ggml/src/ggml-cuda/common.cuh` | ~1500 | ~10 |
