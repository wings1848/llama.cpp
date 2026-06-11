# Progress Tracking

## Task: SWLP Deep-Dive Gap Analysis (2026-06-11)

### Done
- Read all 12 specified source/documentation files
- Analyzed code vs docs alignment across 5 source modules + 5 docs
- Produced `gap_report.md` with 16 identified gaps (3 HIGH, 3 MEDIUM, 10 LOW)

### Key Findings
- **3 HIGH severity**: Expert cache activation is no-op (`#if 0`), P4 partial ngl support is commented out (`swlp.reset()`), CUDA graphs have no integration caller
- **3 MEDIUM**: `--swlp-auto` only in bench tool, async migration untested, `use_pinned_copy` parameter never read
- **10 LOW**: Doc inconsistencies, dead code, undocumented methods

### Recommended Next Steps
1. P0 triage: Fix or formally de-scope expert cache, CUDA graphs, and P4
2. P1: Wire in async migration benchmarks and validate performance claims
3. P2: Clean up dead parameters and align docs with reality

---

## Task: P0 Code Restructuring + 3 Bug Fixes (2026-06-11)

### Done
- Split `llama-swlp.cpp` (1375→449 lines) into 5 modules: migrate, adaptive, moe, stats
- Extracted internal state into `llama-swlp-internal.h`
- Created consistent `.h` files for each module (incl. new `llama-swlp-migrate.h`)
- Removed `#if 0` dead code blocks and unused fields (`HYSTERESIS_THRESHOLD`, `bytes_copied`)
- Refactored `prepare_migration()` extracting 4 helper methods
- Deleted duplicate `build_swlp_cuda.ps1`
- Moved root-level management .md files into `docs-zh/`

### Reviewed & Fixed (by code review + subagent)
- Critical: Unguarded `ggml_backend_cuda_host_buffer_type()` call (fixed)
- 6 warnings/suggestions cleaned up

### Bug Fixes (this session)
- **swlp-auto 无法初始化** (C8): `llama-context.cpp` 创建条件 `>0` 排除 auto 模式 `=-1`，已修复
- **verbose 无迁移统计** (C9): `prepare_migration()` 新增 evict/load 数量和耗时日志
- **测试框架假阳性** (T5): `_scan_bugs()` 改为同 ngl 精确比较，跳过无匹配基线的检测

### Compiled & Verified
- VS 2022 x64 build: 0 errors, 0 new warnings
- 22/22 tests passed, 0 anomalies
- swlp-auto + verbose: verified working (PP=628 t/s vs 327 t/s before fix, +92%)
