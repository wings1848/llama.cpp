# SWLP 自适应窗口算法优化分析

> 本文档分析 SWLP 自适应窗口算法中 `PCIE_BANDWIDTH_GBPS` 硬编码常量对跨设备部署的影响，以及缺失/待优化的关键因素和改进方案。

---

## 1. 硬编码常量分析

### 1.1 代码位置与作用域

**文件**：`src/llama-swlp-adaptive.cpp:57-61`

```cpp
// PCIe 3.0 x16 ~12 GB/s = 12e9 bytes/s -> time_us = bytes / (12e9) * 1e6 = bytes / 12000
size_t layer_bytes = state->layer_sizes_bytes[il];
constexpr double PCIE_BANDWIDTH_GBPS = 12.0;
int64_t migrate_us = (int64_t)(layer_bytes / (PCIE_BANDWIDTH_GBPS * 1000.0));
state->layer_migrate_us[il] = migrate_us;
```

这个常量只出现在 `record_layer_timing()` 中，写入 `state->layer_migrate_us[il]`。

### 1.2 数据流追踪：硬编码常量无实际影响

```
record_layer_timing()  →  layer_migrate_us[il]  →  ⛔ 无人读取
prepare_migration()    →  last_migration_us     →  ✅ adapt_window() 使用
                           (实际 ggml_time_us() 测量值)
```

| 变量 | 写入位置 | 使用方 | 受硬编码影响？ |
|------|---------|--------|:------------:|
| `layer_migrate_us[il]` | `record_layer_timing()` 用 12 GB/s 估算写入 | **从未被任何代码读取** | 无影响 |
| `last_migration_us` | `prepare_migration()` 用 `ggml_time_us()` 实测 | `adapt_window()` → EWMA → ratio 决策 | **无影响** |
| `ewma_migrate_us` | `adapt_window()` 基于 `last_migration_us` EWMA 平滑 | ratio 公式的分母 | **无影响** |

`adapt_window()` 的注释也明确说明了这一点（第 94-96 行）：

```cpp
// Use ACTUAL measured migration time from prepare_migration() instead of
// the PCIe-bandwidth estimate. The measured time reflects real conditions.
int64_t measured_migrate_us = state->last_migration_us;
```

**结论：当前自适应决策不依赖硬编码常量。** 它在任何 PCIe 设备（3.0/4.0/5.0、x4/x8/x16、NVLink）上都能根据实测迁移时间自适应运行。

### 1.3 潜在风险（技术债务）

`layer_migrate_us` 虽未被读取，但留下了隐患——如果有人后续在 `adapt_window()` 中引入基于 `layer_migrate_us` 的**初始窗口预选**逻辑（例如在第一次迁移前做窗口估计），这个硬编码值就会导致严重误判：

| 真实设备 | 实际单向带宽 | 硬编码值 | 偏差 |
|---------|:----------:|:--------:|:----:|
| PCIe 3.0 x16 | ~12 GB/s | 12 GB/s | 0% |
| PCIe 4.0 x16 | ~25-30 GB/s | 12 GB/s | -52%~-60% |
| PCIe 5.0 x16 | ~50-60 GB/s | 12 GB/s | -76%~-80% |
| PCIe 3.0 x4 | ~3 GB/s | 12 GB/s | +300% |
| NVLink (A100) | ~300 GB/s | 12 GB/s | -96% |
| Apple M-series | 统一内存(~0) | 12 GB/s | N/A |

此外，开发者阅读代码时可能误以为这个常量在起作用，导致对跨设备性能行为产生错误预期。

---

## 2. 缺失/待优化的因素

### 2.1 `model_size / n_layers` 未作为连续参数影响调优

**现状**：`auto_tune_adaptive_params()` 使用 `total_model_bytes` 进行离散分档：

```
< 1 GB  → alpha=0.12, interval=8
< 3 GB  → alpha=0.18, interval=4
< 8 GB  → alpha=0.24, interval=2
>= 8 GB → alpha=0.28, interval=2
```

**问题**：
- 分档阈值是硬编码的，边界附近不连续
- 没有使用 **per-layer size**（即 `(total_model_bytes - non_layer_bytes) / n_layers`）
- MoE 模型中每层更大，但 alpha/interval 与 Dense 模型使用相同档次
- `model_size / n_layers` 直接决定迁移单层的耗时，是自适应算法最直接的输入参数

### 2.2 PCIe 带宽未被自动探测

**现状**：`record_layer_timing()` 中使用的 `PCIE_BANDWIDTH_GBPS = 12.0` 是 PCIe 3.0 x16 的典型值。

**问题**：
- 实测迁移时间（`last_migration_us`）已被 `adapt_window()` 使用，但**首次迁移前的初始窗口**（`auto_window` = 40% of layers）完全没有考虑带宽
- `layer_migrate_us[il]` 基于错误带宽写入，若将来被使用会造成误判
- 不同 `ngl`（固定层数）情况下剩余可滑动层数不同，所需的窗口策略也不同

### 2.3 逐层大小不均衡

**现状**：自适应算法将所有层等同对待——`ewma_layer_us` 是窗口内各层的**平均**计算时间。

**问题**：
- MoE 层比 Dense 层大 2-8x，但自适应决策中使用的是平均值
- 较大的层迁移耗时更长，应降低其在窗口中的权重
- 若窗口边界正好落在大层上，迁移成本比平均估算高得多

### 2.4 异步迁移与计算的 Overlap 未被计入

**现状**：`adapt_window()` 中的 ratio 为：

```
ratio = (window_size × ewma_layer_us) / (ewma_migrate_us + 1)
```

**问题**：
- 异步迁移（`async_migration_enabled=true`）时，H2D 传输与上一层的 GPU 计算可以**流水线重叠**
- 当前 ratio 将迁移时间视为完全串行的开销，高估了实际迁移成本
- 在 PCIe 5.0 或 NVLink 等高速互联上，overlap 比例更高，当前策略可能导致窗口偏小

### 2.5 CUDA Graph 失效成本未纳入

**现状**：当层被驱逐时，对应的 CUDA Graph 被 `invalidate_layer_graph(il)` 标记为无效，下次需要重新 capture。

**问题**：
- Graph recapture 有几十到几百微秒的开销
- 频繁的窗口调整导致频繁的 graph rebuild
- 自适应算法没有将 graph rebuild 成本计入迁移开销

---

## 3. 改进建议

### 3.1 [方案A] 实测带宽自动校准（推荐，低成本高收益）

在首次迁移后自动推算真实 PCIe 带宽，替换硬编码常量：

```cpp
// llama-swlp-adaptive.cpp — 新增校准方法
void llama_swlp::calibrate_pcie_bandwidth() {
    if (!state || state->last_migration_us <= 0 || state->total_migrated_bytes == 0) {
        return;
    }
    // total_migrated_bytes / (last_migration_us * 125 * 1024) → GB/s
    // 125 = 1000*1000 / (8*1000) 的简化：μs → s，bytes → bits，bits → GB
    double measured_bw = (double)state->total_migrated_bytes
                         / (state->last_migration_us * 125.0 * 1024.0);
    state->measured_pcie_bw_gbps = measured_bw;
    if (verbose) {
        LLAMA_LOG_INFO("SWLP: measured PCIe bandwidth = %.1f GB/s\n", measured_bw);
    }
}

// record_layer_timing() — 使用实测带宽
double bw = state->measured_pcie_bw_gbps > 0.0
            ? state->measured_pcie_bw_gbps
            : PCIE_BANDWIDTH_GBPS;  // fallback
state->layer_migrate_us[il] = (int64_t)(layer_bytes / (bw * 1000.0));
```

**调用时机**：在 `prepare_migration()` 设置 `last_migration_us` 后，`adapt_window()` 之前。

**优点**：
- 自动适应所有 PCIe 世代和链路宽度
- 对 NVLink、C2C 互联同样有效
- 零人工配置
- 改动范围小（~20 行）

**需要添加的内部状态**：
```cpp
// llama-swlp-internal.h — state 结构体中新增
double measured_pcie_bw_gbps = 0.0;  // 0.0 = 尚未校准
```

### 3.2 [方案B] PCIe 链路参数主动探测（更彻底，跨平台）

在 `set_backends()` 时查询 GPU 链路能力，用于初始窗口选择和带宽校准：

```cpp
// llama-swlp.cpp — set_backends() 扩展
void llama_swlp::detect_pcie_config() {
    if (!state->gpu_backend) return;

#ifdef GGML_USE_CUDA
    int pcie_gen = 0, pcie_width = 0;
    // 通过 CUDA Runtime API 查询
    cudaDeviceGetAttribute(&pcie_gen,   cudaDevAttrPciGen3Support, device_id);
    cudaDeviceGetAttribute(&pcie_width, cudaDevAttrPciLinkWidth,   device_id);

    // Gen 代际带宽估算（单向，扣除~10%协议开销）
    // Gen3: ~1.0 GB/s per lane, Gen4: ~2.0, Gen5: ~3.9, Gen6: ~7.5
    static const double bw_per_lane[] = {0.0, 0.0, 0.0, 0.985, 1.969, 3.938, 7.500};
    double bw = (pcie_gen > 0 && pcie_gen <= 6) ? bw_per_lane[pcie_gen] * pcie_width : 12.0;

    state->measured_pcie_bw_gbps = bw;
    if (verbose) {
        LLAMA_LOG_INFO("SWLP: detected PCIe Gen%d x%d = %.1f GB/s\n",
            pcie_gen, pcie_width, bw);
    }
#endif
}
```

**优点**：
- 首次迁移前就能获得准确的带宽估算
- 可用于初始窗口选择（而非固定的 40%）

**局限**：
- 需要后端特定 API（CUDA/HIP/Vulkan 各平台不同）
- 虚拟机/直通场景可能无法准确探测

### 3.3 引入 `model_size / n_layers` 作为连续参数

将 `auto_tune_adaptive_params()` 的离散分档改为连续映射：

```cpp
// llama-swlp-adaptive.cpp — auto_tune_adaptive_params() 优化
// 计算平均每层权重大小（GB）
size_t layer_bytes_total = state->total_model_bytes - state->non_layer_bytes;
float avg_layer_gb = (float)layer_bytes_total
                     / (state->num_layers * 1024.0f * 1024.0f * 1024.0f);

// 连续映射：大层 → 迁移更贵 → alpha 稍大（反应更快）
// 范围 0.10 ~ 0.35，假设层大小 0.1 GB ~ 5 GB
float alpha_base = 0.10f + std::min(avg_layer_gb * 0.05f, 0.25f);
state->ewma_alpha = std::clamp(alpha_base, 0.10f, 0.35f);

// adapt_interval 也用 per-layer size 连续映射
// 大层 → 迁移成本高 → 不要频繁调整
if (state->adapt_interval == 0) {
    if      (avg_layer_gb < 0.3f) state->adapt_interval = 8;
    else if (avg_layer_gb < 1.0f) state->adapt_interval = 4;
    else if (avg_layer_gb < 3.0f) state->adapt_interval = 2;
    else                          state->adapt_interval = 1;
}
```

### 3.4 计入异步迁移的 Overlap 效应

当 `async_migration_enabled=true` 时，有效迁移成本应降低：

```cpp
// llama-swlp-adaptive.cpp — adapt_window() 中 ratio 计算优化
float effective_migrate_us = state->ewma_migrate_us;

if (state->async_migration_enabled) {
    // 异步模式下，迁移可以与上一层的 GPU 计算重叠。
    // 保守估计：重叠比例为 min(迁移时间, 计算时间) 的 50%。
    float overlap = std::min(state->ewma_migrate_us, state->ewma_layer_us);
    effective_migrate_us = std::max(1.0f, state->ewma_migrate_us - overlap * 0.5f);
}

float ratio = (state->window_size * state->ewma_layer_us) / effective_migrate_us;
```

### 3.5 逐层大小加权迁移成本

将每层的实际大小纳入迁移成本估算，而非使用窗口内平均值：

```cpp
// llama-swlp-adaptive.cpp — adapt_window() 中细化
// 计算窗口内各层的加权平均迁移时间
float weighted_migrate_sum = 0.0f;
float total_weight = 0.0f;
for (int il = w_start; il < w_end && il < state->num_layers; il++) {
    float weight = (float)state->layer_sizes_bytes[il];
    if (weight > 0) {
        weighted_migrate_sum += weight * state->layer_migrate_us[il];
        total_weight += weight;
    }
}
float avg_layer_migrate_us = (total_weight > 0)
    ? weighted_migrate_sum / total_weight
    : state->ewma_migrate_us;
```

---

## 4. 综合改进路线图

| 优先级 | 改进项 | 影响范围 | 代码量 | 风险 |
|:-----:|--------|---------|:-----:|:----:|
| P0 | **方案A：实测带宽自动校准** | `llama-swlp-adaptive.cpp` + `internal.h` | ~20 行 | 低 |
| P1 | **`model_size/n_layers` 连续参数化** | `llama-swlp-adaptive.cpp` | ~15 行 | 低 |
| P1 | **异步 Overlap 计入 ratio** | `llama-swlp-adaptive.cpp` | ~8 行 | 低 |
| P2 | **逐层大小加权迁移成本** | `llama-swlp-adaptive.cpp` | ~15 行 | 中 |
| P2 | **方案B：PCIe 链路主动探测** | `llama-swlp.cpp` + CUDA 后端 | ~30 行 | 中 |
| P3 | **CUDA Graph rebuild 成本纳入** | `llama-swlp-adaptive.cpp` + `cuda.cpp` | ~25 行 | 高 |

### 推荐实施顺序

1. **方案A**（P0）→ 消除硬编码常量的技术债务，自动适配所有设备
2. **`model_size/n_layers` + 异步 Overlap**（P1）→ 提升自适应精度
3. **逐层加权**（P2）→ 处理 MoE 等大小不均衡场景
4. **方案B + Graph rebuild 成本**（P2/P3）→ 极致优化

---

## 5. 跨设备部署影响总结

| 设备类型 | 当前影响 | 方案A后 | 方案A+B后 |
|---------|:-------:|:-------:|:---------:|
| PCIe 3.0 x16 | 无（实测为准） | 校准值≈12 GB/s，不变 | 初始 window 合理 |
| PCIe 4.0 x16 | 无 | 校准值≈28 GB/s，`layer_migrate_us` 正确 | 初始 window 更大 |
| PCIe 5.0 x16 | 无 | 校准值≈55 GB/s | 初始 window 可更大 |
| PCIe 3.0 x4 | 无 | 校准值≈3 GB/s | 初始 window 更小 |
| NVLink | 无 | 校准值≈300-600 GB/s | 初始 window 接近全量 |
| Apple M-Unified | N/A | 校准值≈极大（共享内存） | SWLP 自动降级 |
| 虚拟机/直通 | 无 | 校准值反映实际直通带宽 | 适配虚拟化环境 |

**核心结论**：当前代码在不同设备上**功能正确**，但 `layer_migrate_us` 作为技术债务需要在未来扩展前清理。方案A 以极低成本消除了这一隐患，并使可能的 future extension（初始窗口预选、per-layer 迁移成本估算）获得正确的数据基础。
