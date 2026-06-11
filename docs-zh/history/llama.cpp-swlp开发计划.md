# SWLP 架构评审与重构/优化方案

## 目标
对 llama.cpp 项目中 SWLP (Sliding-Window Layer Pipeline) 相关代码进行架构评审，给出文件拆分、合并精简、重构优化和目录结构调整的完整可行方案。

---

## 一、代码量总览

| 类别 | 文件 | 行数 | 说明 |
|------|------|------|------|
| 核心实现 | `src/llama-swlp.cpp` | 1375 | 窗口管理/迁移/自适应/MoE缓存/统计/CUDA转发 |
| 核心头文件 | `src/llama-swlp.h` | 101 | 公开接口 |
| CUDA实现 | `src/llama-swlp-cuda.cpp` | 343 | CUDA图捕获 (PIMPL) |
| CUDA头文件 | `src/llama-swlp-cuda.h` | 48 | CUDA图接口 |
| 张量模板 | `src/llama-swlp-tensors.h` | 190 | 层/模型张量枚举 (与 llama-model.h 耦合) |
| **核心合计** | | **2057** | |
| | | | |
| 上下文集成 | `src/llama-context.cpp` | +46 | 散落在4182行文件中 |
| 公共API | `include/llama.h` | +21 | `llama_swlp_params` 结构体 |
| CLI参数 | `common/arg.cpp` | +71 | 散落在4253行文件中 |
| 公共配置 | `common/common.cpp/h` | +16 | 参数传递 |
| ggml集成 | `ggml/` 3个文件 | +19 | 回调机制 |
| 构建脚本 | `build_swlp_cuda.*` | 176 | 两个重复的脚本 |
| 测试代码 | `examples/swlp-test/` | 654 | test + bench |
| Python测试 | `scripts/` | 1129 | swlp_test.py + yaml |
| 文档 | `docs-zh/` + 根目录 | 大量 | 多处重复 |

---

## 二、架构问题诊断

### 问题 1：`llama-swlp.cpp` (1375行) 包含过多关注点

当前文件承担了至少 7 个独立职责：

| 职责 | 大致行数 | 关键函数 |
|------|---------|---------|
| 窗口滑动管理 | ~120行 | `prepare_layer()`, `need_layer_move()`, `is_layer_gpu()` |
| 层迁移引擎 | ~350行 | `migrate_layer_to_gpu()`, `restore_layer_to_cpu()`, `prepare_migration()`, `ensure_window_ready()` |
| 自适应EWMA调优 | ~120行 | `auto_tune_adaptive_params()`, `adapt_window()` |
| MoE专家缓存 | ~120行 | `predict_experts()`, `ensure_experts_cached()`, `prepare_layer_experts()` |
| 模型分析 | ~90行 | `analyze_model()` |
| 统计/诊断打印 | ~60行 | `print_stats()`, `estimate_*` |
| CUDA图转发 | ~25行 | `begin_capture_layer()` 等 (thin wrappers) |

**建议**: 按职责拆分为多个文件。

### 问题 2：`llama-swlp-tensors.h` 耦合风险

- 190行的模板函数逐字段枚举 `llama_layer` 的每一个张量成员
- 文件注释明确标注 "must stay in sync with llama-model.h"
- 同时提供层级和模型级两种遍历
- 张量枚举覆盖了 normal norm/attn/ffn/mamba/rwkv/rope/bitnet/altup/kimi/DSA/posnet/convnext/shortconv/nextn 等 170+ 个字段

**建议**: 将张量枚举改为由 `llama_layer` 本身提供（如静态注册数组），消除外部文件手动维护的风险。

### 问题 3：`#if 0` 死代码块

两处被禁用的代码：
- `record_expert_activations()` (约50行, 行824-857) - 从GPU读取MoE专家选择结果，已知在大型MoE上挂死
- `prepare_layer_experts()` (约10行, 行800-813) - 专家张量预取，标记为 "P0 phase 2"

这些代码代表了未完成的功能路径，保留它们增加了阅读负担，且可能被误认为活动代码。

### 问题 4：`prepare_migration()` 函数过长

该函数 (~170行) 包含首次迁移和增量迁移两个几乎独立的代码路径，每个路径里面又包含驱逐阶段和加载阶段，中间还有多层嵌套的条件和CUDA事件处理。两个代码路径有大量重复的CUDA事件等待/释放逻辑。

### 问题 5：根目录项目管理文件堆积

根目录有 10+ 个与SWLP开发过程相关的 markdown 文件：
- `plan.md` (本次分析将被覆盖)
- `progress.md`, `context.md`, `comprehensive_test_review.md`
- `review-code.md`, `review-docs.md`, `test_review.md`, `test_scripts_review.md`
- `gap_report.md`, `docs_review.md`

这些是开发过程的产物，不应该作为永久代码资产留在根目录。

### 问题 6：构建脚本重复

`build_swlp_cuda.ps1` (93行) 和 `build_swlp_cuda.ps1` (83行) 功能几乎完全相同，都需要在修改构建参数时同步更新两份。

### 问题 7：`common/arg.cpp` 中SWLP参数分散

4253行的 arg.cpp 文件缺乏模块化组织，SWLP的71行参数定义与其他参数交错在一起，没有逻辑分组。

---

## 三、重构方案

### 方案 A：文件拆分

#### A.1 拆分 `src/llama-swlp.cpp` → 5个文件

```
src/llama-swlp.cpp          → 保留公共接口和调度 (~200行)
src/llama-swlp-migrate.cpp  → 迁移引擎: migrate_layer_to_gpu, restore_layer_to_cpu, prepare_migration, ensure_window_ready (~380行)
src/llama-swlp-adaptive.cpp → 自适应调优: auto_tune_adaptive_params, adapt_window, record_layer_timing, hysteresis逻辑 (~140行)
src/llama-swlp-moe.cpp      → MoE专家缓存: predict_experts, ensure_experts_cached, prepare_layer_experts, 模型分析中的MoE部分 (~130行)
src/llama-swlp-stats.cpp    → 统计打印: print_stats, estimate_*, get_ewma_alpha, get_adapt_interval (~80行)
```

#### A.2 对应的头文件调整

```
src/llama-swlp.h                  → 保持简洁的公开接口
src/llama-swlp-internal.h         → 新增，暴露内部函数供拆分后的文件使用
src/llama-swlp-migrate.h          → 迁移引擎头文件
src/llama-swlp-adaptive.h         → 自适应调优头文件
src/llama-swlp-moe.h              → MoE专家缓存头文件
src/llama-swlp-stats.h            → 统计打印头文件
```

#### A.3 CUDA文件保持独立（已经合理）

```
src/llama-swlp-cuda.cpp/h        → 不变，PIMPL设计良好
```

### 方案 B：合并精简

#### B.1 `llama-swlp-tensors.h` 重写为模型自注册

**当前问题**: 190行模板函数外部枚举 `llama_layer` 的所有张量，与 `llama-model.h` 强耦合。

**建议方案**: 在 `llama-model.h` 的 `llama_layer` 中添加一个静态张量指针数组或 `for_each` 方法，由模型定义本身维护。这样当添加新张量时，编译器会提示同步更新。

具体做法：
```cpp
// llama-model.h 中添加
struct llama_layer {
    // ... existing fields ...
    // SWLP-compatible tensor visitor
    template<typename F>
    void visit_tensors(F && fn) const {
        // normalization
        fn(attn_norm); fn(attn_norm_b);
        // ... all tensors defined here alongside their declaration
    }
};
```

这样 `llama-swlp-tensors.h` 可以简化为对 `layer.visit_tensors()` 的调用，甚至可以直接废弃。

**优先级**: 中（需要改动 llama-model.h，影响范围较大）

#### B.2 构建脚本合并

删除 `build_swlp_cuda.ps1`，在 `build_swlp_cuda.ps1` 中同时支持 cmd 和参数传递。或者只保留一个，因为 CMakeLists.txt 已经是最权威的构建定义。

**建议**: 保留 `build_swlp_cuda.ps1`（Windows上更通用），删除 `.ps1`。

#### B.3 项目文档整理

根目录下的项目管理文件全部移到 `docs-zh/history/` 或删除：

```
plan.md                     → docs-zh/history/llama.cpp-swlp开发计划.md
progress.md                 → docs-zh/history/ 或删除（已过时）
context.md                  → docs-zh/history/ 或删除
comprehensive_test_review.md → docs-zh/reports/
review-code.md              → docs-zh/reports/
review-docs.md              → docs-zh/reports/
test_review.md              → docs-zh/reports/
test_scripts_review.md      → docs-zh/reports/
gap_report.md               → docs-zh/reports/
docs_review.md              → docs-zh/reports/
```

### 方案 C：重构改进建议

#### C.1 提取 SWLP 的后端接口抽象

当前 SWLP 直接调用 `ggml_backend_cuda_*` 函数。可以通过在 llama.h 中定义一组迁移后端操作，由 ggml 后端实现，使 SWLP 与 CUDA 解耦。

```cpp
// ggml/include/ggml-backend.h 中添加
struct ggml_backend_migration_i {
    void * (*create_event)(ggml_backend_t backend);
    void   (*destroy_event)(void * event);
    void   (*wait_event)(ggml_backend_t backend, void * event);
    // ... H2D async copy, etc.
};
```

**优先级**: 低（这是架构演进，当前 CUDA ifdef 方式可行）

#### C.2 `prepare_migration()` 重构

将首次迁移和增量迁移提取为两个私有函数：

```cpp
void prepare_migration_first(int w_start, int w_end, int pf_end);
void prepare_migration_incremental(int w_start, int w_end, int pf_end,
    int old_w_start, int old_w_end, int old_pf_end);

// 公共辅助
void evict_layer(int il);
void load_layer_to_gpu(int il);
```

减少重复代码和嵌套深度。

#### C.3 MoE代码清理

两处 `#if 0` 代码：
- `record_expert_activations()`: 添加清晰的注释说明挂死原因和计划，或移到独立的 TODO 文件
- `prepare_layer_experts()`: 同上，标记为 "P0 phase 2" 能力

如果这些功能在可预见的未来不会实现，应该移除而非注释掉。

#### C.4 SWLP CLI 参数模块化

在 `common/arg.cpp` 中，所有SWLP参数应该集中在同一个区域，用明显的分隔注释标记：

```cpp
// ============================================================
// SWLP (Sliding-Window Layer Pipeline) options
// ============================================================
// ... 所有 SWLP 相关的 10 个参数定义 ...
// ============================================================
```

当前它们已经在一段连续的区域内（行2371-2418），这是好的。但头文件 `common/common.h` 中 SWLP 参数相关的结构体成员散落在 `gpt_params` 中。

#### C.5 llama-context.cpp 中 SWLP 初始化逻辑提取

当前 SWLP 初始化 (~20行) 散在 `llama_context` 构造函数的多个位置。建议提取为：

```cpp
void llama_context::init_swlp(const llama_context_params & params) {
    // centralized SWLP initialization
}
```

这不会减少总行数，但提高了可读性和可维护性。

### 方案 D：目录结构优化

推荐最终目录结构：

```
src/
  llama-swlp.cpp              # 公共接口 (200行)
  llama-swlp.h                # 公开头文件
  llama-swlp-internal.h       # 内部头文件（llama_swlp_state定义）
  llama-swlp-migrate.cpp      # 迁移引擎 (380行)
  llama-swlp-migrate.h
  llama-swlp-adaptive.cpp     # 自适应调优 (140行)
  llama-swlp-adaptive.h
  llama-swlp-moe.cpp          # MoE专家缓存 (130行)
  llama-swlp-moe.h
  llama-swlp-stats.cpp        # 统计打印 (80行)
  llama-swlp-stats.h
  llama-swlp-cuda.cpp         # CUDA图 (343行，不变)
  llama-swlp-cuda.h
  llama-swlp-tensors.h        # 张量枚举 (简化后 < 50行，或废弃)

scripts/
  swlp/                       # 新目录，所有Python测试和配置集中
    swlp_test.py              # 627行
    swlp_test_config.yaml     # 331行
    download_models.py        # 171行

examples/
  swlp-test/                  # 不变
    swlp-test.cpp
    swlp-bench.cpp
    CMakeLists.txt

build_swlp_cuda.ps1           # 保留，删除 build_swlp_cuda.ps1

docs-zh/
  history/                    # 开发历史文档
    llama.cpp-swlp开发日志.md
    llama.cpp-swlp开发计划.md  # ← moved from root plan.md
    llama.cpp-swlp状态.md
  reports/                    # 评审报告
    comprehensive_test_review.md  # ← moved from root
    review-code.md                # ← moved from root
    ... (其他评审文件)
  llama.cpp-swlp技术设计.md
  llama.cpp-swlp故障排查与常见问题.md
```

---

## 四、优先级排序

### P0 (高优先级 - 影响可读性和可维护性)

| 序号 | 任务 | 文件 | 工作量 |
|------|------|------|--------|
| 1 | **拆分 llama-swlp.cpp** 为 5 个文件 | `src/llama-swlp*.cpp/h` | 2-3h |
| 2 | **清理 `#if 0` 死代码** | `src/llama-swlp.cpp` | 15min |
| 3 | **重构 prepare_migration()** 提取子函数 | `src/llama-swlp-migrate.cpp` | 1h |
| 4 | **清理根目录项目管理文件** | 根目录 | 30min |
| 5 | **删除重复构建脚本** (.ps1) | 根目录 | 5min |

### P1 (中等优先级 - 改善架构)

| 序号 | 任务 | 文件 | 工作量 |
|------|------|------|--------|
| 6 | **MoE 代码清理** - 将 dead code 标记清楚或归档 | `src/llama-swlp-moe.cpp` | 30min |
| 7 | **SWLP 初始化逻辑提取** | `src/llama-context.cpp` | 1h |
| 8 | **scripts/ 目录重组** (swlp 子目录) | `scripts/` | 30min |

### P2 (低优先级 - 长期优化)

| 序号 | 任务 | 文件 | 工作量 |
|------|------|------|--------|
| 9 | **llama-swlp-tensors.h 重构为模型自注册** | `src/llama-model.h`, `src/llama-swlp-tensors.h` | 3-4h |
| 10 | **ggml 迁移后端接口抽象** | `ggml/include/ggml-backend.h` | 4-6h |

---

## 五、实施计划（按顺序执行）

### 阶段 1：立即清理（~1h）

```
1.1 清理根目录文件
    - 将 plan.md, progress.md, context.md 移到 docs-zh/history/
    - 将 review-*.md, gap_report.md, comprehensive_test_review.md 移到 docs-zh/reports/
    - 删除 build_swlp_cuda.ps1

1.2 清理死代码
    - 移除 record_expert_activations() 中 #if 0 块 (行824-857)
    - 移除 prepare_layer_experts() 中 #if 0 块 (行800-813)
    - 用 TODO 注释替代，指向 issue tracker

验证: git diff --stat 确认仅文件移动和删除
```

### 阶段 2：核心文件拆分（~3h）

```
2.1 创建新文件
    - src/llama-swlp-internal.h       (llama_swlp_state 移至此处)
    - src/llama-swlp-migrate.cpp/h     (迁移引擎)
    - src/llama-swlp-adaptive.cpp/h    (自适应调优)
    - src/llama-swlp-moe.cpp/h         (MoE专家缓存)
    - src/llama-swlp-stats.cpp/h       (统计打印)

2.2 拆分逻辑
    llama-swlp.cpp 保留:
      - 构造函数/析构函数
      - has_prefetch, need_layer_move, prepare_layer
      - get_window_start, get_window_size, is_layer_gpu
      - annotate_graph, annotate_expert_topk, on_split_begin
      - set_backends, set_model, set_fixed_gpu_layers
      - for_each_layer_tensor
      - set_async_migration, is_async_migration
      - CUDA graph forwarding wrappers
      - record_layer_timing, record_compute_time, record_forward_type

    llama-swlp-migrate.cpp:
      - migrate_layer_to_gpu, restore_layer_to_cpu
      - prepare_migration (使用提取后的子函数)
      - ensure_window_ready

    llama-swlp-adaptive.cpp:
      - auto_tune_adaptive_params
      - adapt_window (含 hysteresis 逻辑)

    llama-swlp-moe.cpp:
      - analyze_model 中的 MoE 部分
      - predict_experts, ensure_experts_cached, prepare_layer_experts

    llama-swlp-stats.cpp:
      - print_stats, estimate_active_memory_mb, estimate_total_memory_mb
      - get_memory_reduction_ratio, get_ewma_alpha, get_adapt_interval

2.3 更新 CMakeLists.txt
    - src/CMakeLists.txt 添加新的 .cpp 文件

验证: 编译通过，swlp-test 运行正常
```

### 阶段 3：重构 prepare_migration() (~1h)

```
3.1 提取子函数
    - evict_layer_range(start, end, skip_fixed)
    - load_layer_range(start, end, skip_fixed)
    - 消除首次迁移和增量迁移之间的重复代码

验证: 所有现有行为保持不变，swlp-test 通过
```

### 阶段 4：SWLP 初始化提取 (~1h)

```
4.1 在 llama-context.cpp 中
    - 提取 init_swlp() 私有方法
    - 将分散的 SWLP 初始化调用集中到此方法
    - 构造函数中简化为一行 init_swlp(params)

验证: 编译通过，context 创建正常
```

### 阶段 5：scripts 重组 (~30min)

```
5.1 创建目录结构
    - scripts/swlp/
    - 移动 swlp_test.py, swlp_test_config.yaml, download_models.py 到 scripts/swlp/

5.2 更新引用
    - 更新 swlp_test.py 中的相对路径（如果存在）

验证: swlp_test.py 可正常导入和执行
```

### 阶段 6：长期优化（可选，需更多讨论）

```
6.1 llama-swlp-tensors.h 重构
    - 需要与模型维护者协调
    - 在 llama_model.h 的 llama_layer 中添加 visit_tensors() 方法
    - 使张量声明和枚举定义在同一位置

6.2 ggml 后端迁移接口抽象
    - 需要与 ggml 维护者协调
    - 在 ggml-backend 中添加迁移操作接口
```

---

## 六、文件修改清单

### 修改的文件
- `src/llama-swlp.cpp` - 拆分为多个文件，保留核心接口
- `src/llama-swlp.h` - 移出内部状态定义
- `src/CMakeLists.txt` - 添加新 .cpp 文件
- `src/llama-context.cpp` - 提取 init_swlp()

### 新增的文件
- `src/llama-swlp-internal.h` - llama_swlp_state 定义
- `src/llama-swlp-migrate.cpp` - 迁移引擎
- `src/llama-swlp-migrate.h`
- `src/llama-swlp-adaptive.cpp` - 自适应调优
- `src/llama-swlp-adaptive.h`
- `src/llama-swlp-moe.cpp` - MoE专家缓存
- `src/llama-swlp-moe.h`
- `src/llama-swlp-stats.cpp` - 统计打印
- `src/llama-swlp-stats.h`

### 删除的文件
- `build_swlp_cuda.ps1` - 与 .bat 重复
- `plan.md` - 移到 docs-zh/history/ (本文件即替换)
- `progress.md` - 移到 docs-zh/history/
- `context.md` - 移到 docs-zh/history/
- `comprehensive_test_review.md` - 移到 docs-zh/reports/
- `review-code.md` - 移到 docs-zh/reports/
- `review-docs.md` - 移到 docs-zh/reports/
- `test_review.md` - 移到 docs-zh/reports/
- `test_scripts_review.md` - 移到 docs-zh/reports/
- `gap_report.md` - 移到 docs-zh/reports/
- `docs_review.md` - 移到 docs-zh/reports/

### 移动的文件
- `scripts/swlp_test.py` → `scripts/swlp/swlp_test.py`
- `scripts/swlp_test_config.yaml` → `scripts/swlp/swlp_test_config.yaml`
- `scripts/download_models.py` → `scripts/swlp/download_models.py`
- 根目录 .md 管理文件 → `docs-zh/history/` 或 `docs-zh/reports/`

### 不变的文件
- `src/llama-swlp-cuda.cpp/h` - PIMPL设计合理
- `src/llama-swlp-tensors.h` - 短期不变，长期重构
- `examples/swlp-test/` - 全部不变
- `include/llama.h` - 不变
- `ggml/` 下所有文件 - 不变
- `build_swlp_cuda.ps1` - 保留
- `docs-zh/` 下技术文档 - 不变

---

## 七、风险与依赖

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 拆分文件引入编译错误 | 高 | 逐步拆分，每次验证编译 |
| llama-swlp-tensors.h 重构需要改动 llama-model.h | 中 | 延迟到阶段6，先保持现有实现 |
| 移动文件破坏现有引用 | 低 | 使用 `git mv` 保留历史 |
| 死代码删除后有人需要 | 低 | dead code 保留在 git 历史中 |
| common/arg.cpp 模块化不被上游接受 | 低 | 仅做局部注释分隔，不做文件拆分 |
| 拆分后的头文件依赖关系复杂 | 中 | llama-swlp-internal.h 作为内部共享头文件 |

---

## 八、验证检查清单

完成后逐项验证：

- [ ] `cmake --build .` 编译无错误无警告
- [ ] `llama-swlp-test` 运行正常
- [ ] `llama-swlp-bench` 运行正常
- [ ] SWLP CLI 参数全部可用 (`--swlp-window`, `--swlp-adaptive` 等)
- [ ] 无 SWLP 时 (window_size=0) 行为和拆分前一致
- [ ] `git log --follow` 可追踪移动文件的历史
- [ ] 根目录清洁（无临时项目管理文件）
- [ ] `scripts/swlp/swlp_test.py` 正常运行
