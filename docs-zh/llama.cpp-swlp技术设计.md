# llama.cpp SWLP 技术设计文档

Sliding-Window Layer Pipeline (SWLP) 是为 llama.cpp 设计的动态 GPU 层流式传输引擎。

---

## 1. 设计动机

### 1.1 背景与问题
在传统 GPU 推理中，为了实现高速计算，模型的所有权重必须一次性装入并常驻显存（VRAM）。这给消费级 GPU 运行超大参数模型（如 7B+ 模型）带来了物理内存屏障。
传统的 `--n-gpu-layers` (NGL) 方案只支持静态层划分，其缺点是：
* 若层全部上 GPU（NGL 设为最大），显存不足则直接 OOM。
* 若层只有部分上 GPU，CPU 上计算的部分层会拖慢整体推理性能。

### 1.2 SWLP 的核心思路
由于 Transformer 架构推理是**逐层串行计算**的，任意时刻仅有当前层在执行前向传播。SWLP 充分利用该局部性原理：
1. **显存滑动窗口**：仅在 GPU 显存内开辟对应窗口大小的动态 Buffer。
2. **异步传输**：在 GPU 计算当前层时，异步预取后继层，驱逐已过期层，利用 PCIe 传输与 GPU 计算进行流水线重叠。

```
时间轴: ───────────────────────────────────────────>
         [layer 0] [layer 1] [layer 2] [layer 3] ...
GPU窗口:  [████████████████████]  ← 滑动 ──>
                                  迁移 迁移
CPU驻留: [████████]               [████████] [████████]
```

---

## 2. 架构总览

### 2.1 代码目录结构
* `src/llama-swlp.h`：核心公共接口，定义了 `llama_swlp` 控制类与外部配置参数。
* `src/llama-swlp.cpp`：流传输引擎核心逻辑，包含窗口状态维护、层数据重定向与异步迁移细节。
* `src/llama-swlp-cuda.h` / `.cpp`：CUDA 图（CUDA Graph）捕获、重放与生命周期失效管理。
* `src/llama-swlp-tensors.h`：统一的张量遍历器，避免硬编码遍历导致的新增权重张量漏检。
* `src/llama-context.cpp`：推理计算图的集成点，在执行图构建与分配时调用 SWLP 对应钩子（Hooks）。

### 2.2 状态管理结构 (llama_swlp_state)
SWLP 控制器的状态通过私有成员 `llama_swlp_state` 维护，其主要数据结构如下：
```cpp
struct llama_swlp_state {
    // 窗口配置与状态
    int window_size;
    int prefetch_depth;
    int num_layers;
    int window_start;
    int fixed_gpu_layers;                  // ngl 指定的固定驻留层数
    std::vector<bool> layer_in_gpu;         // 标记层是否在 GPU
    std::vector<bool> layer_fixed_gpu;       // 标记层是否被模型加载器固定
    std::vector<size_t> layer_sizes_bytes;   // 每层权重大小统计

    // 后端引用与 GPU 内存池
    ggml_backend_t gpu_backend;
    ggml_backend_t cpu_backend;
    struct layer_gpu_state {
        ggml_backend_buffer_t gpu_buffer;                     // 该层专属的 GPU Buffer
        std::vector<void*> saved_data;                        // 原始 CPU 权重指针备份
        std::vector<ggml_backend_buffer_t> saved_buffers;     // 原始 CPU 权重 Buffer 备份
    };
    std::vector<layer_gpu_state> layer_gpu;

    // 异步迁移 PCIe 流水线 (v7)
    bool async_migration_enabled;
    std::vector<void*> migration_events;                     // cudaEvent_t 事件句柄，用于流同步

    // 自适应调参 (EWMA)
    bool adaptive_enabled;
    std::vector<int64_t> layer_compute_us;
    std::vector<int64_t> layer_migrate_us;
    float ewma_layer_us;                                     // 计算耗时滑动平均值
    float ewma_migrate_us;                                   // 迁移耗时滑动平均值
    float ewma_alpha;
};
```

---

## 3. 核心算法设计

### 3.1 窗口滑动算法
每次前向计算推进到层 `il` 时，均会执行窗口边界检测：
1. **固定层过滤**：若 `il < fixed_gpu_layers`，则该层属于永久驻留层，跳过滑动逻辑。
2. **边界滑移**：若 `il` 越过当前窗口范围 `[window_start, window_start + window_size)`：
   * 计算新边界：`new_start = clamp(il - window_size + 1, fixed_gpu_layers, num_layers - window_size)`。
   * 驱逐旧边界以外的层（即释放 GPU 内存，回写数据并重置指针）。
   * 更新 `window_start = new_start`。

### 3.2 层迁移算法 (Evict & Load)
迁移在计算图内存分配前（即 `prepare_migration()`）执行，分两步：
* **Phase 1: 驱逐 (Evict)**
  遍历所有不在新窗口 `[window_start, window_start + window_size + prefetch_depth)` 内部且 `layer_in_gpu[il] == true` 的层：
  1. 将修改后的层数据从 GPU 读回至 CPU (如 KV 缓存权重等)。
  2. 将张量的 `data` / `buffer` 指针恢复为 `saved_data` 和 `saved_buffers` 指向的 CPU 初始内存。
  3. 释放对应的 GPU `gpu_buffer`，标记 `layer_in_gpu[il] = false`。
* **Phase 2: 加载 (Load)**
  遍历新范围 `[window_start, window_start + window_size + prefetch_depth)` 内所有未加载到 GPU 的层：
  1. 为该层计算总体张量大小，并分配单块连续 GPU 显存 Buffer。
  2. 备份当前的 CPU 内存映射。
  3. 逐个重定向该层张量的 `data`/`buffer` 指针至新分配的 GPU 内存段。
  4. 执行 H2D 拷贝：
     * **同步模式**：阻塞式 `ggml_backend_tensor_set`。
     * **异步模式**：在迁移专用流上非阻塞入队传输，并插入同步 `cudaEvent_t`。

### 3.3 自适应窗口大小 (EWMA Adaptive Tuning)
若启用 `--swlp-adaptive`，引擎会在推理计算阶段动态修正窗口大小：
* **原理**：让计算耗时与迁移传输耗时相匹配：
  $$\text{Ratio} = \frac{\text{window\_size} \times \text{EWMA}_{\text{计算}}}{\text{EWMA}_{\text{单层迁移}}}$$
* **调节逻辑**：
  * 若 $\text{Ratio} < 0.5$（计算太快，传输来不及），则调大窗口以隐藏 PCIe 延迟。
  * 若 $\text{Ratio} > 2.0$（传输富余），则缩减窗口大小以节省显存。
* **平滑消噪**：针对 Gen 推理模式，平滑因子设为 $\alpha \times 0.25$ 以抑制突发毛刺，并设定 hysteresis 迟滞窗口以防止抖动。

---

## 4. 推理循环集成

SWLP 与 `llama-context.cpp` 的推理主循环高内聚集成：

```
[llama_decode]
   │
   ├── 窗口发生改变？─ 是 ─> [prepare_migration()] (执行 Phase 1 驱逐 与 Phase 2 异步加载)
   │                       
   ├── 核心调度循环 (逐层图构建)
   │     ├── [prepare_layer(il)]  ==> 窗口前滑，触发失效 CUDA Graph
   │     └── [执行 GPU 计算 Kernels]
   │
   ├── [GPU 硬件计算完成] (同步)
   │
   ├── [record_layer_timing(il)]  ==> 收集层计算耗时与实际迁移时间
   └── [adapt_window()]           ==> 自适应更新窗口大小配置
```

---

## 5. 内存模型与线程安全

### 5.1 layer_fixed_gpu 静态固定机制
为了避免重入与显存双重分配（模型加载器初始显存与 SWLP per-layer 显存冲突），SWLP 采用了自动探测机制：
* 推理初始阶段，SWLP 扫描所有层张量的 buffer 属性。
* 若检测到该层已预载在 GPU (即由于 NGL 设置导致部分层已在 GPU buffer 中)，则自动置 `layer_fixed_gpu[il] = true`。
* 处于该状态的层在迁移周期内被自动豁免，以保障多后端协同的逻辑健壮性。

### 5.2 异步迁移流同步 (v7 Pipeline)
通过 `--swlp-async-migration 1` 启用异步传输通道：
* **计算流 (S_compute)**：在默认流上执行 GPU 计算 Kernel。
* **迁移流 (S_migrate)**：专门开辟一条非阻塞 CUDA 拷贝流负责 H2D 数据流动。
* **同步设计**：
  * 层 `il` 异步拷贝指令提交至 `S_migrate`。
  * 在 `S_migrate` 上排队录入事件 `migration_events[il]`。
  * 在 `S_compute` 计算流调度计算核心前，调用 `ensure_window_ready` 指示 `S_compute` 等待 `migration_events[il]` 事件，实现流的流水线乱序执行安全。

---

## 6. CUDA 图 (CUDA Graph) 支持

由于滑动窗口不断重新分配 Buffer，张量的 GPU 虚拟内存物理地址会周期性改变，导致整图失效。
* **重用设计**：SWLP 采用**按层捕获**的 CUDA Graph 设计。
* **失效机制**：当窗口发生滑动，导致层 `il` 的物理指针因重新分配而更改时，精确调用 `invalidate_layer_graph(il)` 销毁这一层的 `cudaGraphExec_t`，并在下一轮解码执行时自动触发重新捕获（Re-capture），其余未发生迁移的层对应的 CUDA Graph 依旧有效。

---

## 7. 张量迭代器设计 (llama-swlp-tensors.h)

为消除硬编码遍历在未来引入新权重张量时的漏检隐患，SWLP 采用基于宏展开和模板元函数的遍历层张量设计：
```cpp
template<typename F>
static void for_each_tensor_in_layer(const llama_layer & layer, F && fn) {
    fn(layer.attn_norm);
    fn(layer.attn_norm_b);
    fn(layer.wq);
    fn(layer.wk);
    fn(layer.wv);
    // ... 枚举全部权重与偏置张量 ...
}
```
* **约束要求**：当主线 `llama-model.h` 的 `llama_layer` 结构体有属性调整时，该文件需同步进行编译匹配定义。

---

## 8. 设计决策记录

* **单层单块 Buffer 映射**：减少 GPU 显存频繁分配产生的碎片与调用耗时。
* **双向原样还原 (saved_data)**：备份原始指针，防止在回滚与进程析构时出现悬空指针或内存泄漏。
* **安全降级**：若发生 GPU 分配异常，强制回退至 CPU 端计算，确保正确性不因软硬件物理限制而降低。

---

## 9. 开发与贡献指南

为了指导其他开发者适配新模型、维护代码，开发时需遵守以下核心设计约定：

### 9.1 张量枚举器契约 (llama-swlp-tensors.h)
若主线 `llama-model.h` 的 `llama_layer` 结构体中新增了任何权重或偏置张量，**必须**同步在此文件中的 `for_each_tensor_in_layer` 宏/模板展开块中进行注册。如果遗漏，新增的张量将不会随窗口滑动进行 GPU 迁移，导致在计算到该张量时由于物理指针错误直接引发崩溃。

### 9.2 异步同步准则 (Event-driven Synchronization)
* **不变式 1**：在对任何属于层 `il` 的张量提交 GPU 计算 Kernel 前，必须通过 `ensure_window_ready()` 强制使计算流 `S_compute` 等待迁移流 `S_migrate` 的完成事件。
* **不变式 2**：修改张量 `data` / `buffer` 指针应在主线程串行执行，确保在指针保存与重设后才插入同步 Event，避免读取到脏指针。

### 9.3 自动化测试验证
在提交任何代码修改前，均应使用内置的自动化测试套件执行回归验证：
```powershell
# 执行快速基准测试
python scripts/swlp_test.py --phase quick --mode both
```
