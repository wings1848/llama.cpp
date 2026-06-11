# Review: SWLP Documentation, Scripts, and Code Consistency

**Review date**: 2026-06-11  
**Scope**: All SWLP `.md` docs (docs-zh/), test/build scripts (scripts/, root), git diff c09eb8b1e..HEAD  
**Review type**: Document accuracy, completeness, test correctness, build robustness, cross-document consistency

---

## Summary

| Severity | Count | Key Issues |
|:--------:|:-----:|-----------|
| 🔴 Blocker | 2 | `build_swlp_cuda.ps1` referenced but never created; `--swlp-async-migration` CLI example broken |
| 🟡 Moderate | 5 | Conflicting performance numbers, 3 inconsistent recommendation tables, dead code still present, section 8 placeholder text |
| 🟢 Note | 5 | Minor test-script quirks, outdated comments, Gemma naming inconsistency across script/doc |

---

## 1. 🔴 `build_swlp_cuda.ps1` Does Not Exist

**Evidence**:  
Only `build_swlp_cuda.bat` exists in the repo root. The `.ps1` is referenced in:

| File | Reference |
|------|-----------|
| `docs-zh/llama.cpp-swlp优化.md` §5 | `powershell -File build_swlp_cuda.ps1` (primary recommended command) |
| `docs-zh/README.md` build table | `build_swlp_cuda.ps1` listed as **recommended** script |
| `plan.md` Batch C | Proposes modifying `build_swlp_cuda.ps1` lines (file is assumed to exist) |

**Impact**:  
Users following the documentation's primary instruction will get a file-not-found error. The `.bat` file works correctly (it *does* have auto-detect logic — the `.bat` already implements what the plan proposed for the `.ps1`), so only the documentation is wrong. The bat file handles: env var override, 4-edition VS 2022 auto-search, incremental build, and `--clean` flag.

**Fix**:  
Either create the `.ps1` with the same logic as the `.bat`, or correct all doc references to say `build_swlp_cuda.bat` is the only script available.

---

## 2. 🔴 `--swlp-async-migration` Not Supported by Bench

**Evidence**:  
`examples/swlp-test/swlp-bench.cpp` does NOT parse `--swlp-async-migration`. The flag is only handled in `common/arg.cpp` (used by `llama-cli` / `llama-server`).  
But the README example runs it against the bench:

```
.\build_cuda_swlp\bin\llama-swlp-bench.exe model.gguf --window 8 --prefetch 2 --swlp-async-migration 1
```

This will print "Unknown option: --swlp-async-migration" and exit with code 1.

**Impact**:  
Users following this example will get an error. The async migration feature is real in the C++ code (CUDA stream migration + events), just not exposed via the bench CLI.

**Fix**:  
Either add `--swlp-async-migration` parsing to the bench, or correct the README to reference `llama-cli` or remove the bench example for async migration.

---

## 3. 🟡 Performance Numbers Inconsistent Across Documents

Three sets of numbers, all dated 2026-06-11, all for the same models:

| Model | 优化.md §10.2 (v5) | 测试报告 §4.1 (v4) | Delta |
|-------|:------------------:|:------------------:|:-----:|
| M1 0.5B | 363→442 (+22%) | 339→343 (+1%) | baseline diff 24 t/s |
| M2 1.5B | 415→429 (+3%) | 382→412 (+8%) | baseline diff 33 t/s |
| M3 3B | 335→369 (+10%) | 314→353 (+12%) | baseline diff 21 t/s |
| M4 7B | 362→375 (+4%) | 300→395 (+32%) | baseline diff 62 t/s |
| M5 48L | 21→21 (0%) | 16→18 (+12%) | baseline diff 5 t/s |

**Causes**:  
- The optimization doc says its data is "v5" (after buffer-alignment + leak fixes)  
- The test report is titled "v4" and was never updated to reflect v5 measurements  
- Neither document explains *why* the numbers differ so significantly (e.g. different GPU clock, different prompt length, different batch size, post-fix behavior change)

**Impact**:  
A reader cannot tell which table reflects current reality. The v5/v4 labels are noted but not explained.

**Fix**:  
Either re-run the test suite and update both docs with matching numbers, or add a clear explanation of what changed between v4 and v5 (and delete the outdated table).

---

## 4. 🟡 Three Conflicting Recommendation Tables

| Source | Adaptive (稀疏 <1B) | Window (3-7B) | Prefetch (7B+) |
|--------|:-------------------:|:-------------:|:--------------:|
| **优化.md §4** (v6 diff) | **开** | 1-16 | 0 |
| **技术设计.md §10.3** (v5) | **关** | 1-16 | 0 |
| **测试报告v4 §9** | **关** | **n-2** | **2** |

The optimization doc says Adaptive is **on** (recommended), the other two say **off**.  
Window for 3-7B ranges from "1-16" to "n-2" (single value, which for 28-layer M4 is 26).  
Prefetch for 7B+ is "0" in 2 docs but "2" in the test report.

**Impact**:  
Users receive contradictory advice depending on which document they read.  
**(Note**: The code's `auto_window` computes `40%` of layers, which for 28 layers ≈ 11, matching no table exactly.)

**Fix**:  
Reconcile to a single set of recommendations in one authoritative location, and have the other documents reference it. The optimization doc is the most recent (v6 diff) and should be authoritative.

---

## 5. 🟡 Section 8 of Tech Design Doc Has Corrupted Content

**Evidence**:  
`docs-zh/llama.cpp-swlp技术设计.md` lines 413–435 contain placeholder blanks:

```
### 8.1 当前设计

SWLP 操作在  的主线程中同步执行。迁移期间的  调用是同步的（阻塞直到拷贝完成）。

### 8.2 异步迁移 (v7)

通过  启用异步 PCIe 流水线化：

- **S_migrate**: 新增专用流 (), 处理所有 H2D/D2H 拷贝
- **Events**: 每个层有一个  (cudaEvent_t)
- **线程安全**: 所有 migration_events 访问在同一个宿主线程内 ( →  串行)
```

There are blank spaces where function names, stream names, and event type names should be. Additionally, sections 8.3/8.1/8.2 are **duplicated** — the old pre-v7 content about "未来方向" was not removed when the new v7 async migration section was inserted. The reader sees:

```
8.1 当前设计     (v7 — with blanks)
8.2 异步迁移    (v7 — with blanks)
8.3 未来方向    (empty)
8.1 当前设计     (old duplicate)
8.2 未来方向     (old duplicate)
```

**Impact**:  
The section is unreadable. The blanks suggest incomplete template text that was never filled in.

**Fix**:  
Fill in the blanks with correct content (e.g. "`llama_decode`", "`ggml_backend_tensor_set`", "`--swlp-async-migration 1`", "cudaStream_t", "cudaEvent_t") and remove the duplicated old section 8.1/8.2.

---

## 6. 🟡 Dead Code Remains: `layer_needed` and `set_fixed_gpu_layers`

**Evidence**:

- `src/llama-swlp.cpp:34` declares `std::vector<bool> layer_needed;`
- `src/llama-swlp.cpp:122` initializes `layer_needed(nl, false);`
- **Zero** reads or writes elsewhere in the entire codebase.

- `src/llama-swlp.h:65` declares `void set_fixed_gpu_layers(int n_layers);`
- `src/llama-swlp.cpp:474-488` defines the method body
- **Only caller** is `src/llama-context.cpp:113`, which is **commented out** (`// When upstream is fixed, enable: swlp->set_fixed_gpu_layers(ngl);`)

**Plan reference**:  
`plan.md` Batch B1 (M1) explicitly lists removing both of these. The changes were planned but never applied.

**Impact**:  
No runtime impact, but dead code adds maintenance burden and confuses readers. An `std::vector<bool>` per layer is trivial memory (~28 bytes for 28 layers), so no practical cost.

**Fix**:  
Remove the declaration, definition, struct member, and constructor initializer for both items as specified in plan.md Batch B1.

---

## 7. 🟡 `window == n_layers` Marked as Known Crash (Incorrect)

**Evidence**:  
`scripts/swlp_comprehensive_test.py` `is_known_crash()` (line ~98):
```python
def is_known_crash(ngl, window, n_layers):
    if window == n_layers:
        return True
```

But the SWLP constructor (`src/llama-swlp.cpp:155`):
```cpp
if (params.window_size == 0 || params.window_size >= num_layers) {
    enabled = false;
    return;
}
```

SWLP gracefully disables itself when `window >= num_layers`. The model runs as a normal full-GPU baseline — no crash. The stress test (test_id 301, `Window==n_layers`) will actually **pass**, but the output will misleadingly say "SKIP (expected)".

**Impact**:  
The stress test for boundary condition `W=n_layers` never validates that SWLP correctly handles this edge case. If the SWLP constructor were changed to crash instead of gracefully disable, the test would not catch it.

**Fix**:  
Change `is_known_crash` to *not* mark `window == n_layers` as a crash. Instead, let the test run normally and verify that SWLP correctly disables itself (e.g. by checking that TPS matches GPU baseline within tolerance).

---

## 8. 🟢 Minor: Test Script Passes Unnecessary Flags for Baselines

In `run_one()` the command always includes `--prefetch N` and `--adaptive` / `--pinned` flags even for baseline tests where `window=0` (SWLP disabled). The bench ignores these when window=0, so no functional issue, but it makes the command lines misleading.

---

## 9. 🟢 Minor: Gemma Naming Inconsistency

The test script uses `"Gemma4-Opus-48"` (no space, no hyphen between "Gemma" and "4").  
The docs already use `"Gemma 4"` (with space, per plan D1 fix).  
The naming should be synchronized. The docs have been fixed (no remaining "Gemma4" occurrences), but the script still uses the old name. Not a blocker since it's just a display label.

---

## 10. 🟢 Minor: `swlp_analyze.py` References Outdated Script Name

The docstring says "Reads result JSON files from `swlp_perf_test.py v2`" but the actual test suite is `swlp_comprehensive_test.py`. The output JSON format matches (same field names), so it works, but the docstring is misleading.

---

## Checklist Summary

| Area | Status | Notes |
|------|:------:|-------|
| Document vs code accuracy | ⚠️ | Mostly accurate except §8 blob corruption and `--swlp-async-migration` claim |
| Feature completeness | ✅ | All major features (SWLP core, adaptive, async migration, expert cache, CUDA graphs, pinned memory) documented |
| Test script correctness | ⚠️ | Minor: `is_known_crash` too broad for W=n_layers; `--swlp-async-migration` untestable via bench |
| Test script completeness | ✅ | 172 tests, 7 models, dense+moe, stress, dry-run mode, proper error handling |
| Build script robustness | ✅ | `.bat` has auto-detect, env var override, incremental build, error handling. `.ps1` is missing though |
| Cross-document consistency | ❌ | Performance numbers don't match; 3 contradictory recommendation tables; Adaptive on/off disputed |

---

## Recommended Actions (Prioritized)

1. **(🔴)** Either create `build_swlp_cuda.ps1` or correct all doc references to `.bat`
2. **(🔴)** Fix README `--swlp-async-migration` example (add flag to bench or change example to non-bench tool)
3. **(🟡)** Reconcile performance numbers and recommendation tables across all 3 docs
4. **(🟡)** Fix section 8 of `llama.cpp-swlp技术设计.md` (fill placeholders, remove duplicate)
5. **(🟡)** Remove dead code: `layer_needed` vector and `set_fixed_gpu_layers` method
6. **(🟡)** Fix `is_known_crash` to not mark `W == n_layers` as expected crash
7. **(🟢)** Synchronize Gemma naming between test script (`Gemma4-Opus-48`) and docs (`Gemma 4-Opus-48`)
8. **(🟢)** Update `swlp_analyze.py` docstring to reference correct test script name
