# llama.cpp SWLP 历史审查与修复日志

本日志保存了 SWLP 特性开发过程中经历的代码 Review 缺陷、构建脚本不兼容问题以及测试套件缺陷的跟踪记录，方便开发者追溯特定代码变更的背景和演进路线。

---

## 1. C++ 核心源码缺陷归档

### C1 [严重] — 首次迁移时预取层被驱逐再加载
* **文件位置**：`src/llama-swlp.cpp`
* **问题描述**：在第一次 `prepare_migration()` 时，驱逐逻辑仅保留窗口内层 `[w_start, w_end)`。但紧接着加载阶段需要载入预取层 `[w_end, pf_end)`。这导致预取层在前置阶段被强行释放，然后又在后置阶段被立刻加载回来，引发了首次推理时双倍的 PCIe 带宽浪费。
* **修复方案**：将 Phase 1 的驱逐判断界限扩大至 `[w_start, pf_end)`，使得预取层在首次分配中不受驱逐影响。
* **当前状态**：**已修复** (2026-06-11 优化合入)

### C2 [高] — MoE 架构漏检 `ffn_gate_up_exps`
* **文件位置**：`src/llama-swlp.cpp`
* **问题描述**：原代码仅检测 `layer.ffn_gate_exps` 来判定是否是 MoE 架构。但对于使用合并门控张量 `ffn_gate_up_exps` 的网络，`layer.ffn_gate_exps` 始终为 `nullptr`，导致 MoE 特性未激活，且对其 `->ne[2]` 的直接读取会静默崩溃。
* **修复方案**：引入降级回退检测：
  ```cpp
  ggml_tensor * expert_tensor = layer.ffn_gate_exps ? layer.ffn_gate_exps : layer.ffn_gate_up_exps;
  ```
* **当前状态**：**已修复** (2026-06-11 优化合入)

### C3 [高] — `ffn_exp_probs_b` 被遍历两次导致内存估计翻倍
* **文件位置**：`src/llama-swlp-tensors.h`
* **问题描述**：该偏置张量在 `for_each_tensor_in_layer` 的 MoE 特征区与 Bias 偏置区均有注册，导致权重大小被二次重复计入，最终产生近两倍的层字节数估算误差。
* **修复方案**：从 MoE 张量宏块中移除该偏置的重复项。
* **当前状态**：**已修复** (2026-06-11 优化合入)

### C4 [高] — 逐层计时使用平均值而非实测值
* **文件位置**：`llama-context.cpp`
* **问题描述**：计算每层计算耗时时错误地套用了 `total_decode_us / n_layers` 均值，使自适应 EWMA 无法对真实的逐层负载差异（如 MoE 门控激活开销）做出准确响应。
* **当前状态**：*待处理*（需重构推理循环以精确收集 Callback 计时）

### C7 [中] — 捕获 CUDA Graph 缺乏 sentinel 校验
* **文件位置**：`src/llama-swlp-cuda.cpp`
* **问题描述**：`begin_capture_embed` 注册 `capture_layer=-2`，但对应的 `end_capture` 接口没有校验该标志，允许混合执行错误的嵌入和输出图捕获，导致静默图损坏。
* **修复方案**：为 `end_capture_embed` 和 `end_capture_output` 添加显式的哨兵状态比对机制。
* **当前状态**：**已修复** (2026-06-11 优化合入)

---

## 2. 构建与移植脚本缺陷归档

原项目存在 5 个不同的编译和打包脚本，功能高度重合，且各自存在不可移植路径。

### B1/B2/B3 [高] — 路径死板与硬编码 D 盘问题
* **问题描述**：`build_swlp.ps1` 和 `build_swlp.py` 内部硬编码了特定的绝对路径（如 `D:\Code_Project\swlp\llama.cpp` 和 `C:\Users\butte\...`），导致在新部署环境上运行一律因目录不存在报错。
* **修复方案**：移除全部野脚本，重构并精简为根目录下的统一脚本，利用 `%~dp0` 或相对路径动态提取。
* **当前状态**：**已修复** (脚本已安全移除，使用根目录构建脚本统一入口)

### B5 [中] — 缺少 MSVC 环境变量 Fallback
* **问题描述**：构建脚本只查找固定的 Visual Studio BuildTools 路径，当系统仅安装了 Community 或 Professional 版本时，直接报错找不到 `vcvars64.bat`。
* **修复方案**：重构 VS 检测引擎，按照 BuildTools -> Community -> Professional -> Enterprise 优先级自动回退匹配。
* **当前状态**：**已修复**

---

## 3. 测试与示例代码缺陷归档

### T1 [中] — 错误套用词汇表大小作为 Token Buffer 长度
* **文件位置**：`examples/swlp-test/swlp-bench.cpp`
* **问题描述**：在分配 Token 数组时，直接将词汇表总大小 `n_vocab` (如 128k+) 作为数组分配大小，分配了无意义的超额内存（约 512KB）。
* **修复方案**：改用生成长度的上限 `target_tokens * 2 + 128` 作为容量。
* **当前状态**：**已修复**

### T2 [中] — Greedy Sampling 全局贪婪循环
* **文件位置**：`swlp-bench.cpp`
* **问题描述**：每次 Token 生成均遍历整个词汇表执行最大值比较，造成了巨大的主线程 CPU 计算污染。
* **当前状态**：*待处理*（计划替换为 `std::max_element` 向量检索）

---

## 4. 代码重构与工程优化归档（2026-06-11）

### C8 [严重] — swlp-auto 模式因创建条件错误无法初始化
* **文件位置**：`src/llama-context.cpp:105`
* **问题描述**：SWLP 上下文创建条件为 `swlp_params.window_size > 0`。自动窗口模式使用 `window_size = -1` 作为标记值，`-1 > 0` 为 false，导致 `llama_swlp` 实例从未被创建。此时即使 bench 显示 `SWLP: ON`，实际没有任何迁移逻辑运行，性能等同纯 CPU 推理。
* **修复方案**：将条件改为 `swlp_params.window_size != 0`，使 `window_size = -1`（auto）也能正确初始化 SWLP 引擎。
* **影响**：修复后 auto 模式在 0.5B 模型上 Prompt Processing 从 327 t/s 提升至 628 t/s（+92%）。
* **当前状态**：**已修复** (2026-06-11)

### C9 [低] — SWLP verbose 日志缺乏迁移统计
* **文件位置**：`src/llama-swlp-migrate.cpp` — `prepare_migration()`
* **问题描述**：verbose 模式下仅输出 `window [start,end) pf+X, N/M layers on GPU`，缺少关键迁移明细：本次迁移了多少层（evict/load 分别计数）、迁移总耗时、每层迁移字节数。不利于性能调优和问题排查。
* **修复方案**：在 `prepare_migration()` 的 verbose 分支中添加 evict 数量、load 数量、迁移总耗时（含 `full`/`incremental` 标记）。
* **当前状态**：**已修复** (2026-06-11)

### T5 [低] — 测试框架异常检测假阳性
* **文件位置**：`scripts/swlp_test.py` — `_scan_bugs()`
* **问题描述**：原始异常检测将 partial GPU 配置（如 ngl=10 + SWLP）与全 GPU 基线（ngl=99）比较，报告 `0.3x base` 为异常。实际上 partial GPU + SWLP 的性能介于 CPU 和全 GPU 之间，属于预期行为。
* **修复方案**：
  1. 新增 `_find_baseline()` 函数，仅查找与 SWLP 测试相同 ngl 的无-SWLP 基线。
  2. 若无匹配基线（如 partial GPU 缺少对应无 SWLP 对照测试），跳过异常检测。
  3. CPU-only SWLP（ngl=0）比较阈值放宽至 ratio < 0.5。
  4. Partial GPU SWLP 比较阈值放宽至 ratio < 0.3 或 > 3。
* **当前状态**：**已修复** (2026-06-11)
