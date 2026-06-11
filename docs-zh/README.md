# llama.cpp SWLP 用户指南

Sliding-Window Layer Pipeline (SWLP) 是动态 GPU 层流式传输引擎。它通过在 GPU 上维护一个滑动窗口，仅缓存当前计算及其附近的层，并在推理过程中异步将权重在 CPU 和 GPU 之间流式传输，从而在极小显存占用下运行超大模型。

---

## 0. 文档索引

### 核心指南 (根目录下)
* **[README.md](README.md)**：本用户指南，涵盖快速上手、构建说明与推荐配置。
* **[llama.cpp-swlp技术设计.md](llama.cpp-swlp技术设计.md)**：架构设计、核心算法实现细节以及**开发与贡献指南**。
* **[llama.cpp-swlp状态.md](llama.cpp-swlp状态.md)**：项目当前活跃状态、路线图规划以及已知限制。
* **[llama.cpp-swlp故障排查与常见问题.md](llama.cpp-swlp故障排查与常见问题.md)**：故障排除 FAQ 手册，包含编译警告、性能调优和运行时错误的处理。

### 详细数据与日志
* **[reports/llama.cpp-swlp测试报告.md](reports/llama.cpp-swlp测试报告.md)**：稀疏与 MoE 模型的完整跑分表格、各参数对比以及显存节省实测数据。
* **[history/llama.cpp-swlp开发日志.md](history/llama.cpp-swlp开发日志.md)**：历史审查清单、缺陷细节与变更 Changelog 归档。

---

## 1. 核心概念

在任意时刻，Transformer 模型的计算是逐层进行的，仅有当前层在 GPU 前向计算。SWLP 利用这一原理：
* **滑动窗口**：GPU 显存中仅缓存滑动窗口大小的层。
* **预取与驱逐**：在 GPU 计算当前层时，异步在后台预取后续层，并驱逐过期层，利用 PCIe 带宽掩盖传输延迟。

```
时间轴: ───────────────────────────────────────────>
         [layer 0] [layer 1] [layer 2] [layer 3] ...
GPU窗口:  [████████████████████]  ← 滑动 ──>
                                  迁移 迁移
CPU驻留: [████████]               [████████] [████████]
```

---

## 2. 构建说明

SWLP 支持 Windows 环境下的 MSVC + CUDA 编译。

### 构建命令
```cmd
# 增量构建
build_swlp_cuda.bat

# 全量清理并重建
build_swlp_cuda.bat --clean
```
> ⚠️ **注意**：Git Bash 环境因 GCC/MSVC 编译器冲突无法直接构建，请使用 PowerShell 或 CMD 运行上述脚本。

> ⚠️ PowerShell 用户请使用 `cmd /c build_swlp_cuda.bat` 或直接调用 MSVC 环境后运行 `cmake --build`。

---

## 3. 快速上手

编译完成后，二进制程序位于 `build_cuda_swlp/bin/` 目录下。

### 3.1 运行基准测试 (bench)
```powershell
# 基准测试（设置窗口大小为 8，预取深度 1）
.\build_cuda_swlp\bin\llama-swlp-bench.exe <model_path> --ngl 0 --window 8 --prefetch 1

# 自动窗口模式（根据模型层数自动计算）
.\build_cuda_swlp\bin\llama-swlp-bench.exe <model_path> --ngl 0 --swlp-auto --prefetch 1

# 自适应窗口模式（EWMA 动态调节）
.\build_cuda_swlp\bin\llama-swlp-bench.exe <model_path> --ngl 0 --window 4 --adaptive --prefetch 1

# 查看详细迁移日志
.\build_cuda_swlp\bin\llama-swlp-bench.exe <model_path> --ngl 0 --window 4 --prefetch 1 --swlp-verbose

# 启用异步 PCIe 传输流水线（bench 工具不会自动启用，需显式传递）
.\build_cuda_swlp\bin\llama-swlp-bench.exe <model_path> --ngl 0 --window 8 --prefetch 1 --async-migration
```
> **注意**：使用 `--swlp-auto` 时，窗口大小自动设为 `max(2, min(n_layers-1, n_layers * 40%))`。

### 3.2 运行主程序 (cli / server)
在 `llama-cli` 或 `llama-server` 中，可以通过命令行参数启用 SWLP：
```powershell
# 启用 8 层滑动窗口，异步 PCIe 迁移，将 0 层固定在 GPU (全 CPU 起步滑动)
# 异步迁移在 CUDA 后端可用时会自动启用，无需显式 --swlp-async-migration 1
.\build_cuda_swlp\bin\llama-cli.exe -m <model_path> --window 8 --ngl 0 -p "Hello"

# 部分 GPU 卸载 (例如固定前 16 层在 GPU，其余层通过窗口滑动管理)
.\build_cuda_swlp\bin\llama-cli.exe -m <model_path> --window 8 --ngl 16 -p "Hello"
```

> **自动启用以来的变化 (2026-06-11)**：自该版本起，SWLP 检测到 CUDA 后端可用时会自动启用异步迁移（`async_migration = true`）。用户不再需要传递 `--swlp-async-migration 1`。如因调试需要强制禁用异步迁移，传递 `--swlp-async-migration 0` 后日志会有明确提示说明已被自动覆盖。

---

## 4. 推荐配置

基于大规模测试的实测结果，以下为不同模型的推荐配置：

| 模型类型 | 规模 | 窗口大小 (Window) | 预取深度 (Prefetch) | 自适应调参 (Adaptive) | 专家缓存 (Expert Cache) | 锁页内存 (Pinned) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **稀疏模型 (Dense)** | <1B | 2-16 | 1 | 关 | - | 关 |
| **稀疏模型 (Dense)** | 1-3B | 4-8 | 1 | **开** | - | 关 |
| **稀疏模型 (Dense)** | 3-7B | 4-8 | 1 | **开** | - | 关 |
| **稀疏模型 (Dense)** | 7B+ | 4-8 | 1 | **开** | - | 关 |
| **MoE (≤4 激活专家)** | 任意 | 4 | 1 | **开** | 2 | 关 |
| **MoE (>4 激活专家)** | 任意 | 4 | 1 | 关 | 4 | **开** |

> **异步迁移 (async_migration)**：自 2026-06-11 起，CUDA 后端编译下会自动启用，无需额外参数。

### 通用优化规则
1. **预取深度**：推荐设置 `prefetch=1`（最平衡）。设为 0 会降低吞吐，设为 2 可能会导致 PCIe 拥堵，在部分模型上产生反效果。
2. **自适应调节**：对于稀疏（Dense）模型，强烈推荐开启自适应窗口大小（`--swlp-adaptive 1`），以平衡传输时间与计算时间。
3. **部分显存固定 (Partial Offload)**：将 `0 < ngl < n_layers` 层固定在 GPU 上，能有效提高基线性能，剩余的层使用 SWLP 窗口滑动。

---

## 5. 显存节省与性能分析

SWLP 的显存节省幅度和推理速度与 `--ngl` 参数紧密相关：

### 5.1 显存节省潜力
理论显存公式如下：
$$\text{VRAM}_{\text{active}} = \text{VRAM}_{\text{非层张量}} + \sum_{il \in \text{窗口} \cup \text{预取}} \text{LayerSize}[il]$$

以 **Qwen2.5-7B Q4_K_M**（28 层，约 4.7 GB 权重）在 `ngl=0`、`window=11`、`prefetch=1` 条件下为例：
* **VRAM 占用**：仅需保留 12 层在 GPU 中（约 2.0 GB），加上 KV 缓存和非层张量，总计约 **3.0 GB**。
* **相比全 GPU 卸载 (5.7 GB)**：**节省显存约 47%**。

### 5.2 显存固定机制 (ngl=99)
* **注意**：如果设置 `ngl=99`（或大于模型层数），模型加载器会默认将所有层放入 GPU。SWLP 会自动识别并标记 `layer_fixed_gpu[il] = true`，**此时 SWLP 不会执行驱逐与迁移，也无法节省显存**。

### 5.3 显存压力与 SWLP 性能增益关系
根据实测，SWLP 的性能吞吐增益与显存压力呈反比：
* **显存富余 (余量 > 3 GB)**：SWLP 的 Gen 性能增益较低 (+0~8%)。
* **显存紧平衡 (余量 ~2 GB)**：SWLP 能够获得中等增益 (+8~9%)。
* **显存极其紧张 (余量 ~300 MB)**：计算与传输重叠最充分，SWLP 生成吞吐增益达到峰值 (**+20%** 以上)。

---

## 6. CLI 参数与环境变量速查

### 6.1 核心参数（`llama-cli` / `llama-server`）
* `--window N`：GPU 层滑动窗口大小（0 表示禁用，-1 表示根据模型层数自动计算）。
* `--swlp-auto`：自动窗口模式，等同于 `--window -1`。
* `--swlp-prefetch N`：预取未来层数（默认 1）。
* `--swlp-adaptive 0\|1`：是否开启自适应窗口大小调节（默认 1）。
* `--swlp-async-migration 0\|1`：是否启用异步 PCIe 传输流水线（CUDA 后端可用时自动启用，无需显式设置）。开启后利用 CUDA 多流重叠传输与计算。
* `--swlp-expert-cache N`：MoE 专家缓存容量（0 表示禁用，仅对 MoE 架构有效）。
* `--swlp-pinned-copy 0\|1`：是否对 PCIe 拷贝使用锁页内存 (Pinned Memory)（默认 1）。
* `--swlp-verbose`：打印详细的迁移和流转日志（包含 evict/load 数量、迁移耗时）。

### 6.2 环境变量
* `GGML_CUDA_FORCE_STREAMS`：设置为非零值以覆盖/硬编码 CUDA 流配置。
