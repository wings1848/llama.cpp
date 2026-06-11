# Review: `scripts/swlp_comprehensive_test.py`

**File:** `C:/Users/butte/Code_Project/swlp/llama.cpp/scripts/swlp_comprehensive_test.py` (715 lines)  
**Review date:** 2026-06-11  
**Scope:** Bugs, dead code, logic errors, optimization opportunities, design issues (10 focus areas)

---

## 1. BUGS

### Bug 1: `desc` field missing from `rec` — `scan_bugs` always shows `"?"`

**Location:** Lines 425–436 (rec dict) vs line 516 (`scan_bugs`)

`scan_bugs()` constructs each bug entry with `"desc": r.get("desc", "?")`, expecting `desc` to be in the per-result record. But `run_model` never copies `test["desc"]` into `rec`. The `desc` field is used only for the console print (line 420) and then discarded.

**Impact:** Every bug report entry shows `desc: "?"` regardless of which test configuration failed, making bug reports worthless for identifying the failing scenario.

**Fix:** Add `"desc": test.get("desc", "")` to the `rec` dict at line 425–436.

---

### Bug 2: Anomaly-detection loop is pure dead code (lines 524–530)

```python
# Check for anomalies
for r in all_results:
    if r.get("error"):
        continue
    # Gen TPS > 2x GPU baseline is suspicious
    # Lower TPS with SWLP vs baseline for dense models with small windows
    pass
```

The `pass` does nothing. The analysis comments describe what should be checked but no code is written. This makes the `scan_bugs` output incomplete — it reports crashes but never flags performance regressions or suspicious speedups.

**Fix:** Implement the checks or remove the dead loop.

---

### Bug 3: Outer `except subprocess.TimeoutExpired` in `run_one` is unreachable (lines 384–387)

The inner `try/except` (lines 348–357) already catches `subprocess.TimeoutExpired` from `proc.communicate(timeout=600)`, calls `monitor.stop()`, and either `continue`s or `return`s. The only code after that inner handler is `monitor.stop()` followed by file/return-code checks — none of which can raise `TimeoutExpired`. The outer `except subprocess.TimeoutExpired` can never fire.

**Impact:** Dead code that gives a false sense of safety. If a different exception type were intended here (e.g., `OSError` from pipe operations), it would never be caught.

---

### Bug 4: `generate_dense_tests` — Gap at `test_id=11` masks a missing PF=1 configuration

**Location:** Lines 210–218

```python
for w, pfs in [(4, [0, 2]), (8, [0, 2])]:
    ...
    for pf in pfs:
        if w == 4:
            tid = 10 if pf == 0 else 12
        else:
            tid = 13 if pf == 0 else 14
```

Only `pf=0` and `pf=2` are tested in the prefetch sweep. `pf=1` (the default) is already covered by the window-sweep tests (tids 6, 7). But `pf=1` with `W=4` and `W=8` in the prefetch *context* (same `ngl=99`, different prefetch values) is not explicitly named as a prefetch test. The gap at `tid=11` is confusing: a reader expects `tid=11` to exist and must trace the logic to understand why it's skipped.

**Severity:** Low (cosmetic). Add a comment explaining the intentional gap.

---

### Bug 5: `generate_stress_tests` has different `add`-function semantics than `generate_dense_tests`

**In `generate_dense_tests` / `generate_moe_tests`:** If `is_known_crash` returns True, the test is simply NOT added (silently omitted).

**In `generate_stress_tests`:** If `is_known_crash` returns True, the test IS added with `expected_skip=True`.

This inconsistency means:
- For `--phase all`, a dense model gets stress tests with `expected_skip=True` AND its own dense tests that silently omit the same config.
- The `--dry-run` output for dense models will show all stress-test entries including `[EXPECTED_SKIP]`, but the dense generator's omitted tests won't appear at all.
- A user trying to understand why a config is missing for a dense model has no output to inspect.

**Severity:** Medium — confusing behavior that could mask missing coverage.

---

### Bug 6: `bench_ram_rss_peak_mb` extracted into `rec` but excluded from CSV

**Location:** Line 476 (extracted) vs lines 491–494 (CSV keys)

`rec["bench_ram_rss_peak_mb"]` is populated (line 476) but the CSV field list does not include it. With `extrasaction='ignore'`, this valuable per-process RAM data silently disappears from the CSV output.

Similarly missing from CSV (present in rec or GPU summary but dropped):
- `vram_avg_mb`, `vram_free_min_mb`, `vram_free_avg_mb`
- `temp_avg_c`
- `mem_bw_util_avg_pct`
- `sm_clock_avg_mhz`, `mem_clock_avg_mhz`
- `ram_used_avg_mb`
- **All PP-specific metrics** (`pp_mean_ms`, `pp_p50_ms`, `pp_p95_ms`, `pp_std_ms`, `pp_iter_ms`) — PP mode runs are in the CSV but their PP metrics are lost
- `gen_iter_ms`
- `exit_code`
- `model_type`, `n_layers`, `quant`, `desc`

---

## 2. DEAD / REDUNDANT CODE

### 2a. Outer `except subprocess.TimeoutExpired` (lines 384–387)

Already described in Bug 3. Unreachable dead code.

### 2b. Anomaly-detection skeleton (lines 524–530)

Already described in Bug 2. `pass` is pure dead code.

### 2c. `_dedup` may mask generator bugs

```python
def _dedup(tests):
    seen = set()
    for t in tests:
        key = (t["test_id"],)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique
```

If a generator accidentally produces two tests with the same `test_id` (which would be a generator bug), the second is silently dropped with no warning. This could mask test-generation regressions during maintenance. The function is also called on each generator individually but never on the combined list when `--phase all` appends stress tests to dense/moe tests.

### 2d. Redundant model-path size check

`resolve_model_path` already checks `st_size > 1000` before returning a path. Then `validate_model` (called immediately after by `run_model`) checks `st_size < 1000` again. This is a harmless double-check but adds an extra `stat()` syscall.

### 2e. Redundant `gpu` field flattening

In `run_one` (lines 373–375):
```python
data["gpu"] = g
data.update({f"gpu_{k}": v for k, v in g.items()})
```

Then in `run_model` (lines 468–470):
```python
gpu = d.get("gpu", {})
for k, v in gpu.items():
    rec[f"gpu_{k}"] = v
```

GPU fields are flattened to `gpu_*` keys twice — once in `run_one` (into `data`) and once in `run_model` (into `rec`). The first flattening is never used because `rec` is built from `d.get("gpu", {})` plus the individual field extraction (lines 471–476). The `gpu_*` keys in `data` are dead by the time they reach the caller.

---

## 3. GPU MONITOR LIFECYCLE

### 3a. No thread leak in normal paths

`start()` creates a daemon thread. `stop()` signals the thread and joins it with a 3-second timeout. In the normal path (process completes within timeout), the lifecycle is:
1. `monitor = GPUMonitor()` → `_bench_proc = None`
2. `monitor.track_pid(proc.pid)` → stores pid
3. `monitor.start()` → spawns daemon thread, collects samples
4. `proc.communicate(timeout=600)` → process completes
5. `monitor.stop()` → signals stop, joins thread

This is correct.

### 3b. Potential thread leak on unexpected exception

If `proc.communicate(timeout=600)` raises a non-`TimeoutExpired` exception (e.g., `OSError` from broken pipe, `ValueError` from closed file descriptor), the inner `except` does not catch it. The exception propagates to the outer `except Exception` (lines 388–390) which does NOT call `monitor.stop()` before `continue`/`return`. The daemon thread continues running until the main process exits.

On retry (if `attempt < MAX_RETRIES - 1`), a new `GPUMonitor` is created and a second daemon thread leaks. Across many models and retries, this could accumulate several leaked threads.

**Severity:** Low in practice (daemon threads die with the process), but could cause issues in long-running test sessions.

### 3c. `nvidia-smi` called every 200ms

Each poll spawns a subprocess. For a 600-second benchmark run, that's **3000 subprocess invocations** per test. This is expensive and could interfere with GPU timing measurements, especially since `nvidia-smi` itself briefly acquires the GPU driver lock.

**Recommendation:** Add a `pynvml` backend (NVML direct bindings) when available, falling back to `nvidia-smi`.

---

## 4. RETRY LOGIC IN `run_one`

### 4a. Correct behavior for each outcome

| Outcome | Inner handler | Retries? | monitor.stop called? |
|---------|---------------|----------|---------------------|
| Timeout (`communicate` raises `TimeoutExpired`) | Inner `except` kills process, calls `communicate()`, calls `monitor.stop()`, then `continue`/`return` | Yes (if attempts remain) | Yes |
| Non-zero exit code | `monitor.stop()` already called at line 358, then rc check | Yes (if attempts remain) | Yes |
| No output file | `monitor.stop()` already called, then file check | Yes (if attempts remain) | Yes |
| Unexpected exception | Outer `except Exception` — no `monitor.stop()` | Yes (if attempts remain) | **No** |

### 4b. `proc.kill()` + `proc.communicate()` on timeout

On Windows, `Popen.kill()` calls `TerminateProcess`, which is forceful. The subsequent `proc.communicate()` (with no timeout) should return quickly because the process is dead. However, if the process has children that don't die, `communicate()` could hang. Consider `proc.kill()` followed by `proc.wait(timeout=5)` instead.

### 4c. Output file overwritten across retry attempts

`out_file = os.path.join(tmpdir, f"result_{os.getpid()}.json")` — same filename for every test and every retry attempt within the same `TemporaryDirectory`. This is fine because each run is sequential, but:
- A stale file from a previous retry could falsely satisfy `os.path.exists() + os.path.getsize() > 10` if the new run doesn't overwrite it fast enough. However, `proc.communicate()` waits for process completion, so the new process has finished and (presumably) overwritten the file by the time we check. This is safe in practice.

---

## 5. MODEL PATH RESOLUTION

### 5a. `resolve_model_path` falls through to a non-existent path

```python
def resolve_model_path(cfg):
    primary = CUSTOM_MODELS_DIR / cfg["path"]
    if primary.exists() and primary.stat().st_size > 1000:
        return str(primary)
    if cfg.get("alt_path"):
        ...
    return str(primary)  # primary may not exist!
```

If neither primary nor alt paths exist, it returns the (non-existent) primary path. The caller (`run_model`) calls `validate_model` to catch this, so it's handled gracefully. But the function's contract is ambiguous — it says "resolve" but sometimes returns an invalid path.

**Suggestion:** Return `None` when no valid path is found, and let the caller handle the absence explicitly.

### 5b. `alt_path` resolution checks CWD-relative and project-models-relative

`alt = Path(cfg["alt_path"])` is relative to the current working directory, while `alt2 = PROJECT_MODELS_DIR / cfg["alt_path"]` is relative to `PROJECT_MODELS_DIR`. This dual-resolution is odd: CWD-relative paths could resolve to `PROJECT_ROOT / qwen25-7b-gguf/...` if CWD is PROJECT_ROOT, which overlaps with `PROJECT_MODELS_DIR / qwen25-7b-gguf/...` since `PROJECT_MODELS_DIR = PROJECT_ROOT / "models"`. They are different directories (`<root>/qwen25-7b-gguf/` vs `<root>/models/qwen25-7b-gguf/`). This seems intentional but fragile.

---

## 6. DATA EXTRACTION FROM BENCH OUTPUT

### 6a. GPU fields extracted into `rec` correctly

All fields from the GPU summary are correctly mapped:

| `rec` key | GPU summary key |
|-----------|-----------------|
| `vram_peak_mb` | `vram_used_peak_mb` |
| `vram_avg_mb` | `vram_used_avg_mb` |
| `temp_peak_c` | `temp_peak_c` |
| `gpu_util_avg_pct` | `gpu_util_avg_pct` |
| `ram_peak_mb` | `ram_used_peak_mb` |
| `bench_ram_rss_peak_mb` | `bench_ram_rss_peak_mb` |

### 6b. `pp_tps` read in gen mode (line 460)

```python
else:  # gen mode
    rec["pp_tps"] = d.get("pp_tps")
```

This reads the prompt-processing throughput even when running in generation-only mode. If the bench outputs only gen metrics when `--gen` is passed, `pp_tps` will always be `None` in gen mode rows. This is harmless but could be confusing in the CSV.

### 6c. PP-specific metrics missing from CSV for PP-mode rows

`pp_mean_ms`, `pp_p50_ms`, `pp_p95_ms`, `pp_std_ms`, `pp_iter_ms` are stored in `rec` (lines 451–456) but excluded from the CSV field list (lines 491–494). So CSV has `pp_tps` but none of the other PP timing distribution metrics.

---

## 7. CSV OUTPUT

### 7a. `extrasaction='ignore'` is correct

Using `csv.DictWriter` with `extrasaction='ignore'` ensures extra keys in `rec` are silently dropped rather than raising an exception. This is the right choice for a test script, but it means CSV can silently lose data.

### 7b. Missing fields (present in `rec`, absent from CSV)

As enumerated in Bug 6 — notably `bench_ram_rss_peak_mb`, all PP timing distribution metrics, `exit_code`, `desc`, `gen_iter_ms`. This reduces the CSV's utility for post-hoc analysis.

### 7c. `pp_tps` in CSV for gen-mode rows

Rows for gen-mode runs will have an empty `pp_tps` cell (because the bench may not output PP metrics when `--gen` is passed). Rows for pp-mode runs will have an empty `gen_*` cells. This is expected but makes the CSV harder to work with in tools like Pandas (mixed missing data patterns).

---

## 8. TEST GENERATORS

### 8a. No duplicate test IDs across generators

| Generator | Test ID range | Overlap risk |
|-----------|--------------|-------------|
| `generate_dense_tests` | 1, 2, 4–10, 12–16, 19 | None |
| `generate_moe_tests` | 101, 102, 112, 114, 201–204, 210, 211, 220, 230 | None |
| `generate_stress_tests` | 301–308, 320, 321 | None |

However, when `--phase all` appends stress tests to dense/moe tests, the combination is NOT deduplicated. If a future code change introduces an overlap (e.g., a new dense test at ID 301), the combined list would have duplicate IDs silently.

### 8b. `_dedup` called per-generator only

Each generator calls `_dedup` on its own output. But in `main`:
```python
tests = generate_moe_tests(mid, cfg)
if args.phase in ("all",):
    tests += generate_stress_tests(mid, cfg)
```

The combined list is NOT deduplicated. This is fine now (no overlaps), but fragile.

### 8c. `generate_moe_tests` — unused test_id 110, 111, 113

`add(110 + w, ...)` produces test IDs 112 and 114 (for w=2 and w=4). Test IDs 110, 111, 113 are unused. This is cosmetic but could confuse readers.

### 8d. `generate_stress_tests` test 321 uses computed window

```python
add(321, 99, n-4 if n-4 >= 2 else 2, 1, True, False, "Adaptive large start")
```

The window value varies per model (20 for M1/n=24, 8 for E1/n=12, etc.). This makes cross-model comparison harder — test 321 for one model is not comparable to test 321 for another.

---

## 9. CLI ARGUMENT HANDLING

### 9a. `--list` and `--dry-run` interaction is correct

```python
if args.list:
    ...list models and return immediately...
```

When `--list` is specified, the script exits before checking `BENCH_EXE` existence and before any `--dry-run` logic. There is no conflict.

### 9b. `--dry-run` bypasses BENCH_EXE check

```python
if not BENCH_EXE.exists() and not args.dry_run:
    print(f"ERROR: {BENCH_EXE} not found. Build first.")
    sys.exit(1)
```

`--dry-run` works without the benchmark binary existing. Correct.

### 9c. `--phase` with `"quick"` selects both dense and moe but NOT stress

`args.phase in ("all", "dense", "quick")` → dense models selected  
`args.phase in ("all", "moe", "quick")` → moe models selected  
`args.phase in ("all",)` → stress tests added

So `--phase quick` runs both dense and moe tests but without the stress test additions. This is reasonable.

### 9d. `--model` with invalid ID prints error but continues

```python
if mid not in MODELS:
    print(f"Unknown model: {mid}")
    continue
```

The error is printed but execution continues with the next model. For a CLI tool, this is acceptable.

---

## 10. PYTHON ANTI-PATTERNS & CODE QUALITY

### 10a. `time.sleep(1.0)` in the test-result loop (line 480)

```python
with open(jpath, 'w') as f:
    json.dump(all_results, f, indent=2)
time.sleep(1.0)
```

There is no comment explaining why a 1-second delay is needed after each test result. This appears to be cargo-cult programming. If it's for GPU cooldown, that should be documented. If it's accidentally left in, it adds a 1-second delay per test, which for 100+ tests means 100+ seconds of unnecessary waiting.

### 10b. JSON written after every result (lines 477–478)

The full `all_results` list is serialized to JSON after every single test result. For a model with 30 tests × 2 modes = 60 results, the file is written 60 times, each write serializing the entire growing list. This is O(n²) serialization for no benefit (only the final state matters).

**Suggestion:** Write the JSON only at the end of `run_model`.

### 10c. Wide imports (line 14)

```python
import argparse, json, os, subprocess, sys, time, csv, tempfile, threading
```

PEP 8 recommends one import per line.

### 10d. `if hasattr(self, '_t')` instead of explicit `__init__` attribute (line 109)

```python
def stop(self):
    self._stop = True
    if hasattr(self, '_t'):
        self._t.join(timeout=3.0)
```

More Pythonic: initialize `self._t = None` in `__init__` and check `if self._t is not None`.

### 10e. `nvidia-smi` polling from subprocess in tight loop (every 200ms)

As noted in 3c — 3000 subprocess spawns per benchmark run. This is heavy. Consider `pynvml` or at least increasing the polling interval.

### 10f. `print(f" SKIP (expected)")` vs `print(f" FAIL: ...")` inconsistency (lines 443–446)

Expected-skip tests print `SKIP (expected)` (no newline because the preceding `print` uses `end=""`). Unexpected failures print `FAIL: ...` with a leading space. This is a cosmetic issue: the output for expected skips lacks context about what was skipped.

### 10g. `scan_bugs` skips expected-skip results but `analyze_results` does not

```python
# scan_bugs:
if r.get("error") and not r.get("expected_skip"):
    # report as bug

# analyze_results:
success = [r for r in all_results if not r.get("error")]
# includes results that have error=False (excluding expected skips since they have error=True)
```

Expected skips are excluded from both bug reports (correctly) and success analysis (correctly). But the print summary `f"{ok_count}/{len(all_results)} ok, {skip_count} expected skips"` could mislead: `ok_count` does NOT include `skip_count`. If 10 results: 5 ok, 3 expected skips, 2 real failures → "5/10 ok, 3 expected skips" = 5 + 3 = 8, but 2 failures are unaccounted for. The math works but a casual reader might add `ok_count + skip_count` and expect it to equal `len(all_results)`.

---

## SUMMARY

| Severity | Count | Key items |
|----------|-------|-----------|
| **Bug** | 6 | `desc` missing from rec (bug reports show "?"); anomaly-detection loop is dead code; unreachable Timeout handler; `add` semantics differ between generators; CSV drops `bench_ram_rss_peak_mb`; PP metrics absent from CSV |
| **Dead code** | 4 | Outer Timeout handler; anomaly-detection skeleton; `_dedup` silently drops duplicates; GPU flattening runs twice |
| **Thread leak** | 1 | Unexpected exception between `monitor.start()` and `monitor.stop()` leaves daemon thread running |
| **Optimization** | 4 | JSON written O(n²) times; `nvidia-smi` every 200ms (3000 subs/run); unnecessary `time.sleep(1.0)`; unnecessary model-path double `stat()` |
| **Quality** | 7 | Wide imports; `hasattr` anti-pattern; undocumented sleeps; confusing log messages; fragile `alt_path` dual-resolution; CSV field list incomplete; missing comments on intentional test-ID gaps |

No critical correctness bugs (the suite will run and produce output). The main risks are:
1. **Bug reports are useless** because `desc` is never stored (Bug 1).
2. **Performance regressions go undetected** because the anomaly checker is not implemented (Bug 2).
3. **GPU metrics are silently dropped from CSV** (Bug 6).
4. **O(n²) JSON writing and 200ms subprocess polling** waste time and system resources.
