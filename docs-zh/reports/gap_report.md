# SWLP Implementation Gap Report

> Generated: 2026-06-11  
> Scope: Code vs Documentation alignment audit  
> Method: Manual cross-reference of src/ and docs-zh/

---

## Critical Gaps (Documentation claims something is done that code does not do)

### GAP-1: Expert Cache "阶段1 已完成" — But `record_expert_activations` is a no-op

**Docs claim** (optimization.md:16, design.md:§3.5):
> "activation_counts 现在通过 hook ffn_moe_topk 正确更新"  
> "prepare_layer_experts 已从 no-op 启用"

**Reality**:
- `record_expert_activations()` at `src/llama-swlp.cpp:1182` immediately returns (`return;` at line 1183). The entire function body is `#if 0` disabled.
- `prepare_layer_experts()` at `src/llama-swlp.cpp:1296-1307` is also `#if 0` (line 1297) — the body is dead code. The function only has `GGML_UNUSED(il/token_count);`.
- `activation_counts[il][e]` is initialized to all zeros in `analyze_model()` and NEVER incremented anywhere in the runtime path.
- **Result**: Predictions from `predict_experts()` sort all-zero arrays; LRU eviction in `ensure_experts_cached()` is meaningless.

**Severity**: 🔴 HIGH — Core feature claimed working is actually non-functional.

### GAP-2: P4 Partial GPU Offload (SWLP side) "已完成" — But the caller is commented out

**Docs claim** (optimization.md:24):
> "P4 部分 GPU offload (SWLP 侧): set_fixed_gpu_layers + layer_fixed_gpu per-layer 跟踪, prepare_migration 正确跳过模型加载器已有的 GPU 层"

**Reality**:
- `set_fixed_gpu_layers()` at `src/llama-swlp.cpp:472` is fully implemented.
- But the ONLY call site (`src/llama-context.cpp:113`) is **commented out**: `// When upstream is fixed, enable: swlp->set_fixed_gpu_layers(ngl);`
- Instead, at line 114, SWLP is entirely disabled: `swlp.reset();`
- **Result**: The P4 fix is dead code. SWLP cannot run with partial ngl at all (0 < ngl < n_layers).

**Severity**: 🔴 HIGH — Feature is code-complete but never activated.

### GAP-3: CUDA Graph "完整实现" — But not integrated into inference loop

**Docs claim** (optimization.md:13):
> "CUDA 图完整实现 (cudaStreamBeginCapture/EndCapture)"

**Reality**:
- `llama_swlp_cuda_graphs` is a complete standalone class with capture/replay/invalidate for per-layer, embed, and output graphs.
- However, **none of the capture/replay methods are ever called** from `llama-context.cpp`:
  - `enable_cuda_graphs()` — never called
  - `begin_capture_layer()` — never called
  - `end_capture_layer()` — never called
  - `replay_layer_graph()` — never called
  - `print_cuda_graph_stats()` — never called
- The only SWLP CUDA methods called from the integration layer are: `annotate_graph()`, `prepare_migration()`, `ensure_window_ready()`, `on_split_begin()`, `record_compute_time()`, `record_layer_timing()`, `record_forward_type()`, `adapt_window()`.
- **Result**: CUDA graphs are a fully-implemented but disconnected subsystem. No graph capture ever occurs during inference.

**Severity**: 🔴 HIGH — Major feature claimed complete but not wired into the execution path.

### GAP-4: `--swlp-auto` CLI Flag — Only exists in bench tool, not main CLI

**Docs claim** (optimization.md:20):
> "✅ --swlp-auto CLI 标志 + bench 工具集成"

**Reality**:
- `--swlp-auto` exists ONLY in `examples/swlp-test/swlp-bench.cpp:59` (the bench test tool).
- **Not present** in `common/arg.cpp` at all — the main CLI (`llama-cli`, `llama-server`) has no `--swlp-auto` flag.
- The main CLI uses `--swlp-window -1` for auto-compute, but `--swlp-auto` as a dedicated flag only exists in the test tool.
- The README example (`README.md:27`) shows `--swlp-auto` with `llama-swlp-bench`, which is correct, but the optimization claim is misleading.

**Severity**: 🟡 MEDIUM — Partially true (bench tool), wrong for main CLI.

### GAP-5: Async Migration "已完成" — Not tested, no results in test report

**Docs claim** (optimization.md:25):
> "异步迁移 (PCIe 流水线化, v7): 专用 CUDA 迁移流 + cudaEvent 同步 + ensure_window_ready() + --swlp-async-migration 1 CLI"

**Reality**:
- The async migration code IS properly implemented: `migrate_layer_to_gpu()` has async path (line 585-613), `ensure_window_ready()` waits on events (line 1039-1068), the CLI arg is at `arg.cpp:2406-2414`.
- BUT the test report (`llama.cpp-swlp综合测试报告.md`) has **zero test results** for async migration. No configuration includes `--swlp-async-migration 1`.
- The `ensure_window_ready()` call is present at `llama-context.cpp:1365`, so the async path is theoretically invocable, but its effectiveness has never been measured.
- The planned migration stream (`S_migrate`) implementation at the ggml-cuda level and pinning interaction has not been validated.

**Severity**: 🟡 MEDIUM — Code exists, untested, no performance data.

### GAP-6: `use_pinned_copy` parameter exists but is never read

**Docs claim** (llama_swlp_params at `include/llama.h:300`):
> `bool use_pinned_copy; // use pinned (page-locked) memory for faster CPU↔GPU DMA`

**Reality**:
- The field is defined in params (`llama.h:300`), parsed from CLI (`arg.cpp:2399-2403`), stored in common_params (`common.h:556`), mapped to cparams (`common.cpp:1594`).
- But **`use_pinned_copy` is never read** in `llama-swlp.cpp` or `llama-context.cpp`. It's stored in params but the actual implementation doesn't consult it.
- The only pinned memory allocation happens inside the async migration path (line 551-556), which checks the `async` parameter, not `use_pinned_copy`.
- **Result**: Setting `--swlp-pinned-copy 1` has zero effect.

**Severity**: 🟡 MEDIUM — Feature appears in docs and CLI but is not wired.

---

## Minor Gaps (Mismatches between code and docs)

### GAP-7: Performance numbers in design doc are inconsistent with test report

**Docs** (design.md:§10.2):
```
| Qwen2.5-0.5B | 363 | 442 | +22% | W=8 |
| Qwen2.5-1.5B | 415 | 429 | +3%  | W=1 |
| Qwen2.5-3B   | 335 | 369 | +10% | W=16 |
| Qwen2.5-7B   | 362 | 375 | +4%  | W=1 PF=0 |
| Gemma 4-48L  | 21  | 21  | 0%   | W=4 |
| Tiny-Moe (2e) | 553 | 514 | -7% | W=4 |
| Phi-mini-MoE (16e) | 120 | 94 | -22% | W=4 (ngl=99) |
```

**Test report** (test.md:§4.1):
```
| M1 0.5B  | 339 | 343 | +1%  | W=8 Adaptive |
| M2 1.5B  | 382 | 412 | +8%  | W=4 Pinned |
| M3 3B    | 314 | 353 | +12% | W=24 |
| M4 7B    | 300 | 395 | +32% | W=26 (n-2) |
| M5 48L   | 16  | 18  | +12% | W=4 |
```

The numbers differ substantially (different baselines, different gains, different best configs). The design doc numbers appear to be from an earlier revision (pre-B0/B1/B2 fixes).

### GAP-8: Adaptive window recommended "关" in test report but "开" in optimization doc

- **Test report** (test.md:§9): Adaptive column = "关" for ALL model types (recommend off).
- **Optimization doc** (optimization.md:§4): Adaptive column = "开" for all sparse models (recommend on).
- **Code reality**: The adaptive path works (called from `llama-context.cpp:2441`), but with the fixes from `adapt_window()` (generation 4× EWMA, hysteresis 8). The test report says it underperforms fixed windows. The optimization doc is more optimistic.

### GAP-9: `has_prefetch()` — Dead code, never called

- Declared in `src/llama-swlp.h:26`, implemented at `src/llama-swlp.cpp:196`.
- **Zero call sites** in the entire codebase (documentation files excluded from search).
- This appears to be legacy API that was superseded by `prefetch_depth > 0` checks elsewhere.

### GAP-10: Design doc window sizes differ from actual code defaults

- **Design doc** (design.md:§10.3): Recommends W=8-16 for <1B, W=1-8 for 1-3B, W=1-16 for 3-7B, W=1-4 for 7B+.
- **Test report** (test.md:§9): Recommends W=16 for <1B, W=8 for 1-3B, W=n-2 for 3-7B, W=4-8 for 7B+, W=4 for MoE.
- **Code**: The auto-compute uses 40% of layers (`llama-swlp.cpp:85`: `num_layers * 40 / 100`), MoE refines to 15% (`llama-swlp.cpp:435`: `num_layers * 15 / 100`).

### GAP-11: `get_graph_stream()` returns nullptr unconditionally

- `src/llama-swlp-cuda.cpp:335`: `void * llama_swlp_cuda_graphs::get_graph_stream() const { return nullptr; }`
- This suggests a planned but unimplemented feature for retrieving the dedicated CUDA graph stream.

### GAP-12: Design doc claims `prepare_migration` is called twice in some paths

- Design doc (§4.1) shows `prepare_migration()` being called both at graph reuse and graph rebuild:
  ```
  ├── [若 SWLP 窗口变更] prepare_migration()  ← SWLP: 强制 re-migrate
  │          ├── prepare_migration()        ← SWLP: 迁移层
  ```
- Actual code at `llama-context.cpp:1338`: The `swlp_force_rebuild` path explicitly avoids double-calling prepare_migration (see comment at line 1337: "Don't call prepare_migration() here — the rebuild branch below will handle it").

---

## Features Code Has That Docs Don't Mention

### GAP-13: `auto_tune_adaptive_params()` — Not documented

- Method at `src/llama-swlp.cpp:447`, called from `llama-context.cpp:121`.
- Auto-tunes `ewma_alpha`, `adapt_interval`, `min_window` based on model size tiers.
- Not mentioned in the API reference (design.md:§11.2).

### GAP-14: `get_ewma_alpha()` / `get_adapt_interval()` — Not documented

- These public accessors at `src/llama-swlp.cpp:952-960` are not listed in the design doc's API section.

### GAP-15: `record_forward_type()` — Not documented as a public method

- Exists in the design doc's callback table (§4.2) but missing from the API reference (§11.2).
- Called from `llama-context.cpp:2440`.

### GAP-16: expert cache `predict_experts()` / `ensure_experts_cached()` — Documented in design but not integrated

- These functions exist and are public in the header but:
  - `predict_experts()` is called from `prepare_layer_experts()` which is `#if 0`
  - `ensure_experts_cached()` is called from `prepare_layer_experts()` which is `#if 0`
  - Neither is called from any integration code path

---

## Summary Table

| # | Gap | Location | Category | Severity |
|---|-----|----------|----------|----------|
| 1 | Expert cache activation tracking is no-op | `llama-swlp.cpp:1182-1193`, `:1296-1307` | Feature dead | 🔴 HIGH |
| 2 | P4 partial ngl support disabled | `llama-context.cpp:113-114` | Caller disabled | 🔴 HIGH |
| 3 | CUDA graphs not integrated | `llama-context.cpp` (never called) | Feature disconnected | 🔴 HIGH |
| 4 | `--swlp-auto` only in bench tool | `arg.cpp` (missing) | Doc overclaim | 🟡 MEDIUM |
| 5 | Async migration untested | Test report (no results) | Unvalidated | 🟡 MEDIUM |
| 6 | `use_pinned_copy` never read | `llama-swlp.cpp` (no reference) | Dead parameter | 🟡 MEDIUM |
| 7 | Stale performance numbers in design doc | `design.md:§10.2` | Data inconsistency | 🟡 MEDIUM |
| 8 | Adaptive recommendation conflict | `optimization.md:§4` vs `test.md:§9` | Doc inconsistency | 🟢 LOW |
| 9 | `has_prefetch()` dead code | `llama-swlp.cpp:196` | Dead code | 🟢 LOW |
| 10 | Design doc and test report config mismatch | Across docs | Doc inconsistency | 🟢 LOW |
| 11 | `get_graph_stream()` returns nullptr | `llama-swlp-cuda.cpp:335` | Stub | 🟢 LOW |
| 12 | Design doc flow outdated | `design.md:§4.1` | Doc inaccuracy | 🟢 LOW |
| 13 | `auto_tune_adaptive_params()` undocumented | Missing from API ref | Doc gap | 🟢 LOW |
| 14-16 | Various API methods undocumented/missing | `llama-swlp.h` vs `design.md:§11.2` | Doc gap | 🟢 LOW |

---

## Recommended Fix Priority

1. **P0**: Decide: enable expert cache (remove `#if 0` guards, fix activation tracking) or officially document it as "not implemented" and remove dead code.
2. **P0**: Either enable P4 partial ngl support (`swlp->set_fixed_gpu_layers(ngl)`) or document it as "blocked on upstream fix".
3. **P1**: Wire CUDA graphs into the inference loop or remove the subsystem.
4. **P1**: Run async migration benchmarks and update docs with real numbers.
5. **P2**: Remove `use_pinned_copy` dead parameter or wire it to actually enable pinned memory.
6. **P2**: Add `--swlp-auto` to `common/arg.cpp` or remove from bench tool docs.
7. **P3**: Align design doc performance table with test report.
8. **P3**: Resolve adaptive recommendation conflict across docs.
