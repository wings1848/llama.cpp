# Script Consolidation Review

## Files analyzed
- `scripts/swlp_comprehensive_test.py` (~530 lines) — "v3" test suite
- `scripts/swlp_perf_test.py` (~430 lines) — "v2" performance test
- `scripts/swlp_analyze.py` (~135 lines) — standalone analysis

---

## A. Duplication — What is shared between `swlp_comprehensive_test.py` and `swlp_perf_test.py`?

Approximately 75% of perf_test.py's logic is directly duplicated in comprehensive_test.py:

| Component | Status |
|---|---|
| `GPUMonitor` class | Both have one — **similar but not identical** (see §B.2) |
| `is_known_crash()` | Identical logic |
| `MODELS` dict (dense models) | Same 5 dense models (M1–M5), but **model names differ** slightly (e.g. `"Qwen2.5-0.5B"` vs `"Qwen2.5-0.5B-Instruct"`) |
| `run_one()` function | Nearly identical CLI construction and output handling |
| `run_model()` function | Same overall flow: iterate tests × modes, run, collect, print, save |
| `generate_tests()` / `generate_dense_tests()` | Same baseline + window + prefetch + adaptive + pinned pattern |
| CLI args `--model`, `--mode`, `--output`, `--dry-run` | Identical semantics |
| `BENCH_EXE`, `SCRIPT_DIR`, `PROJECT_ROOT`, `CUSTOM_MODELS_DIR` | Identical paths |
| `N_GEN_TOKENS`, `N_PROMPT_TOKENS`, `N_WARMUP`, `N_ITERS` | Same constants |

**Verdict: perf_test.py should be deleted, with its few unique features ported into comprehensive_test.py.**

---

## B. Dead code & bugs

### B.1 Dead code in `swlp_perf_test.py`

| Location | Issue |
|---|---|
| Line 93: `"skip_full_gpu": False` in M5 config | **Never read** anywhere in the file. Dead config key. |
| Line 218: `if ngl_ok(0, 0, n):` guarding CPU baseline test | **Always true** — `is_known_crash(0, 0, n)` always returns False because ngl=0 and window=0 satisfy neither crash condition. Dead guard. |
| Lines 250–252: `def ngl_ok(ngl, window, n)` | Single-use wrapper around `is_known_crash()` — called only from the dead guard above. Entire function dead. |
| Line 393: `rec["pp_tps_pp"] = d.get("pp_tps")` | **Duplicate key** — `pp_tps` is already captured on line 385 in the same block. Clearly a copy-paste artifact. |
| `--expert-cache` in CLI (line 265) | Always passed with value 0 since perf tests never set `expert_cache` in test configs. Harmless but unnecessary. |

### B.2 GPUMonitor — differences between the two scripts

**They are NOT identical.** The comprehensive_test.py version is strictly better.

| Aspect | comprehensive_test.py | perf_test.py | Winner |
|---|---|---|---|
| Process tracking | Has `track_pid(pid)` — tracks the **benchmark subprocess** RSS | Tracks `psutil.Process()` — its **own (Python) process** RSS | comprehensive |
| RAM sample keys | `ram_used_mb`, `ram_free_mb` (system), `bench_ram_rss_mb` (subprocess) | `ram_rss_mb` (self-Python), `ram_used_mb` (system) | comprehensive (tracks meaningful subprocess memory) |
| Summary keys | `ram_used_peak_mb`, `ram_used_avg_mb`, `bench_ram_rss_peak_mb` | `ram_rss_peak_mb`, `ram_used_peak_mb` | comprehensive (average too) |
| `vram_free_avg_mb` | **Missing** in summary | Present in summary | perf adds one extra metric |
| Constructor | `self._bench_proc = None` | `self._proc = psutil.Process()` | comprehensive (lazy init via track_pid) |

**Recommendation**: Port `vram_free_avg_mb` from perf's summary to comprehensive's GPUMonitor. Otherwise comprehensive's version is the canonical one.

### B.3 Bug: Missing `gpu_util_avg_pct` extraction in comprehensive_test.py

In `comprehensive_test.py` `run_model()`, the per-test record gets GPU data via:

```python
gpu = d.get("gpu", {})
for k, v in gpu.items():
    rec[f"gpu_{k}"] = v       # creates rec["gpu_gpu_util_avg_pct"]
rec["vram_peak_mb"] = gpu.get("vram_used_peak_mb", -1)
rec["temp_peak_c"]   = gpu.get("temp_peak_c", -1)
rec["ram_peak_mb"]   = gpu.get("ram_used_peak_mb", -1)
rec["bench_ram_rss_peak_mb"] = gpu.get("bench_ram_rss_peak_mb", -1)
```

Notice: `gpu_util_avg_pct` is **never** extracted to the top-level. It ends up buried in `gpu_gpu_util_avg_pct` (with double prefix). Meanwhile `perf_test.py` does:

```python
rec["gpu_util_avg_pct"] = g.get("gpu_util_avg_pct", -1)
```

This means `swlp_analyze.py` (which looks for `r.get("gpu_util_avg_pct", 0)`) gets 0 for comprehensive test results. **Fix**: add the same explicit extraction line to comprehensive_test.py.

### B.4 Model naming inconsistencies

| ID | perf_test.py | comprehensive_test.py |
|---|---|---|
| M1 | `"Qwen2.5-0.5B-Instruct"` | `"Qwen2.5-0.5B"` |
| M5 | `"Gemma-4-Opus-48"` | `"Gemma4-Opus-48"` |

Minor — but could confuse cross-referencing results.

---

## C. Merge plan — Can `perf_test.py` be removed entirely?

**Yes, with the following caveats.**

### C.1 What perf_test.py tests that comprehensive_test.py does NOT

1. **Prefetch sweep for W=8** (test IDs 13, 14) — comprehensive only sweeps W=4 prefetch variants, missing W=8 PF=0 and PF=2.
2. **`alt_path` model resolution** — perf tries a secondary model path as fallback; comprehensive just fails.
3. **Model validation** (`validate_model`) — perf checks for LFS pointer files; comprehensive doesn't.
4. **CSV output** — perf produces `.csv` alongside `.json`.
5. **Retry logic** — perf's `run_one` retries on failure (max_retries=2); comprehensive has no retries.
6. **`--list` option** — perf can list models + their availability.
7. **`vram_free_avg_mb`** in GPUMonitor summary.

### C.2 What comprehensive_test.py adds beyond perf_test.py

- **MoE models** (E1: Tiny-Moe 2 experts, E2: Phi-mini-MoE 16 experts) with expert cache sweep and expert prefetch
- **Stress tests** (edge cases: W=n-1, partial ngl, adaptive edge cases, aggressive prefetch)
- **Bug scanner** (`scan_bugs`) that reports failures
- **Built-in analysis** (`analyze_results`) with model-level summaries, baseline comparison, expert cache impact tables
- **`--phase`** selector (all/dense/moe/stress/quick)
- **Subprocess RAM tracking** in GPUMonitor (benchmark RSS, not Python RSS)
- **`expected_skip`** markers for known crash patterns to distinguish "expected" from "real bug"

### C.3 Recommended merge

```
Before merge:   perf_test.py  (~430 lines, ~75% duplicate)
                comprehensive_test.py  (~530 lines)
                
After merge:    comprehensive_test.py  (~600 lines — add ~70 lines for missing features)
                perf_test.py  [DELETE]
```

**Action items for comprehensive_test.py:**
1. Add W=8 prefetch variants (test IDs 13, 14) to `generate_dense_tests`
2. Add `alt_path` resolution with fallback (port from perf)
3. Add `validate_model` (LFS pointer check)
4. Add CSV output alongside JSON
5. Add `--list` CLI option
6. Add retry logic to `run_one` (2 attempts)
7. Add `vram_free_avg_mb` to GPUMonitor.summary()
8. Fix missing `gpu_util_avg_pct` extraction

---

## D. Optimization opportunities in comprehensive_test.py

### D.1 Duplicate `_dedup` calls
`_dedup` is called inside each generator (`generate_dense_tests`, `generate_moe_tests`, `generate_stress_tests`) AND again in `main()` after concatenation. Since the test ID ranges don't overlap (dense=1–19, MoE=101–230, stress=301–321), the `_dedup` in `main()` is a guaranteed no-op. **Remove it.**

### D.2 Redundant flatten in `run_one` + `run_model`
`run_one` already flattens GPU summary into the data dict:
```python
data["gpu"] = g
data.update({f"gpu_{k}": v for k, v in g.items()})
```

Then `run_model` flattens again:
```python
gpu = d.get("gpu", {})
for k, v in gpu.items():
    rec[f"gpu_{k}"] = v
```

This means every result rec has both `gpu_vram_used_peak_mb` (from second flatten) and `vram_used_peak_mb` (from first flatten via `data.update`). The double flatten is harmless but wasteful. Either:
- Remove the second flatten and rely on explicit extraction + `data` having already-flattened keys, OR
- Remove the first flatten in `run_one` and keep the second (cleaner separation of concerns).

Given that comprehensive_test.py already has explicit extraction (`rec["vram_peak_mb"] = gpu.get(...)`) after the second flatten, **removing the first flatten in `run_one`** would be clearer. The `data["gpu"]` dict is sufficient for passing the GPU summary to `run_model`.

But this could break someone reading raw JSON output files — they might expect `gpu_vram_used_peak_mb` at the top level. So maybe leave it.

### D.3 `time.sleep(1.0)` in `run_model` loop
There's a 1-second cooldown between tests to let GPU cool down. This adds ~10+ seconds per model. Consider making it configurable via `--cooldown` with a default of 0.5s.

### D.4 Redundant overwrite of JSON results
The results file is rewritten on every test iteration:
```python
jpath = model_dir / f"results_{ts}.json"
with open(jpath, 'w') as f:
    json.dump(all_results, f, indent=2)
```

This is fine for crash recovery. But for large test suites (~100 tests), writing the full result list 100 times is wasteful. Could write per-test JSON files and aggregate at the end, but this is a minor concern.

### D.5 Tight coupling between `run_one` and `run_model`
`run_one` returns `{"error": True/False, "data": {...}}`. `run_model` unpacks `d = r["data"]` and manually extracts every field. This is fragile — if the bench adds new output keys, `run_model` must be updated. Consider a more generic approach: pass through all keys from the bench output, plus overlay GPU metrics. But this is a design choice, not a bug.

---

## E. GPUMonitor — current status after fix

The fix (`r.get("gpu")` → `d.get("gpu")`) is verified correct in both scripts:

**comprehensive_test.py `run_one()`** (line ~330):
```python
data = json.load(f)
...
data["gpu"] = g
...
return {"error": False, "data": data}
```

**comprehensive_test.py `run_model()`** (line ~415):
```python
d = r["data"]           # correct: uses returned data, not raw result
gpu = d.get("gpu", {})  # ✅ FIXED: was r.get("gpu")
```

**perf_test.py `run_one()`** (line ~300):
```python
data = json.load(f)
data["gpu"] = g
...
return {"error": False, "data": data}
```

**perf_test.py `run_model()`** (line ~393):
```python
d = r["data"]           # correct
g = d.get("gpu", {})    # ✅ correct, no bug here
```

Both scripts are internally consistent. The fix is complete.

### Minor GPUMonitor issue (both scripts)
The `_poll` method's exception handling is too broad. If `nvidia-smi` fails 10 times in a row (e.g. GPU busy), the sample list will be mostly empty dicts. The `summary()` method filters with `cond=lambda x: x > 0` which handles this, but it means a fully failed monitor run returns `-1` for everything. This is acceptable behavior.

---

## F. `swlp_analyze.py` — overlap assessment

**Keep as-is.** It serves a different purpose:

| Feature | comprehensive_test.py built-in | swlp_analyze.py |
|---|---|---|
| Runs tests | Yes | No — reads saved results |
| Summary tables | Yes (one-shot, stdout) | Yes (re-runnable, stdout) |
| Cross-model comparison | Per-model only | Has explicit cross-model table |
| Chart generation | No | Yes (via matplotlib, optional) |
| GPU health summary | In per-model output | Has GPU temp/util summary |
| Bug scanning | Yes | No |

swlp_analyze.py does NOT overlap with comprehensive_test's built-in analysis because:
1. It works **post-hoc** on saved results (useful for comparing runs from different dates/configs).
2. It has **chart generation** (comprehensive has none).
3. It provides a **cross-model comparison table** that comprehensive doesn't have.

**However**, swlp_analyze.py currently reads `results_*.json` files recursively and expects fields like `vram_peak_mb`, `gpu_util_avg_pct`. Comprehensive_test saves results to `all_results.json` and per-model files, and its field names differ (`vram_peak_mb` works, `gpu_util_avg_pct` does NOT — see §B.3). If you keep both, update swlp_analyze.py to check for `gpu_gpu_util_avg_pct` as a fallback, or fix the field extraction in comprehensive_test.py.

---

## Summary: Actionable edits

### Must-fix (comprehensive_test.py)
1. **Add `gpu_util_avg_pct` extraction** (fixes analysis compatibility):
   ```python
   rec["gpu_util_avg_pct"] = gpu.get("gpu_util_avg_pct", -1)
   ```
   Add after line ~419 alongside the other explicit extractions.

### Should-fix before removing perf_test.py
2. **Add W=8 prefetch variants** to `generate_dense_tests` (restore coverage):
   - Add test IDs 13 (W=8, PF=0) and 14 (W=8, PF=2)
   - Port from perf_test.py lines 229–234
3. **Add `alt_path` fallback** in model path resolution
4. **Add `validate_model`** (LFS pointer check)
5. **Add CSV output** alongside JSON
6. **Add `--list` CLI option**
7. **Add retry logic** (max_retries=2) to `run_one`
8. **Add `vram_free_avg_mb`** to GPUMonitor.summary()

### Nice-to-have
9. **Remove `_dedup`** call from `main()` (it's guaranteed no-op)
10. **Unify model names** across both scripts
11. **Remove `time.sleep(1.0)`** or make it configurable via `--cooldown`

### perf_test.py — delete after porting
12. Before deletion, remove dead code: `pp_tps_pp`, `skip_full_gpu`, `ngl_ok` function.

### swlp_analyze.py — keep, but update for compatibility
13. Add fallback reading of `gpu_gpu_util_avg_pct` when `gpu_util_avg_pct` is absent.
