# llama.cpp SWLP 故障排查与常见问题 (FAQ)

本指南列出了在编译和使用 SWLP (Sliding-Window Layer Pipeline) 过程中可能遇到的常见编译警告、性能调优障碍以及运行时错误的解决方法。

---

## 1. 编译相关问题

### 1.1 MSVC 编译警告 C4190 (UDT returns)
* **警告信息**：`warning C4190: 'ggml_graph_view' has C-linkage specified, but returns UDT 'ggml_cgraph' which is not C-compatible`
* **问题成因**：由于在外部 `extern "C"` Linkage 接口中值返回（Return-by-value）了 C++ 类或结构体。在旧版本中，由于 `swlp` 结构体内含了 `std::vector`，这不仅会触发该警告，还会在运行时由于跨 DLL 边界析构和内存布局不匹配导致崩溃（Access Violation `0xC0000005`）。
* **解决方法**：该警告已被新版本完全修复。目前 `ggml_cgraph` 已改回纯 POD 结构（以 `void *` 指针替代 `std::vector`），在 MSVC 下完全二进制兼容。如果仍看到残余警告，这只是 MSVC 对大结构体值返回的常规 C 兼容提示，不影响运行稳定性。

---

## 2. 显存与内存相关问题

### 2.1 为什么我的 GPU 显存完全没有节省？
* **可能原因 1**：设置了 `--n-gpu-layers 99`（或设定的 NGL 大于模型实际层数）。此时模型加载器初始会将所有的层放入 GPU。SWLP 的自动探测机制会检测到这一状态，为了避免重复分配和不必要的显存开销，会强制将这些层全部标记为 `layer_fixed_gpu` (静态固定)，跳过滑动迁移逻辑。
  * **解决方案**：若要实现真正的显存节省，请确保您的 NGL 设为 0（`--ngl 0`）或部分显存层（如 `--ngl 16`），同时设定合理的滑动窗口大小（如 `--window 8`）。
* **可能原因 2**：运行的模型层数原本就小于或等于您设定的窗口大小。例如，对于 16 层的模型，设定了 `--window 16`。
  * **解决方案**：将窗口设定为显著小于总层数的值（例如设置为 4 或 8），方能释放多余的显存空间。

### 2.2 运行中遇到 CUDA Out of Memory (OOM)
* **可能原因**：自适应窗口大小开启后，分配窗口过大；或者预取深度（`--swlp-prefetch`）太大。
* **解决方案**：
  1. 适当调小滑动窗口，例如显式指定为恒定大小：`--window 4`，并关闭自适应（`--swlp-adaptive 0`）。
  2. 减少预取深度或关闭预取：`--swlp-prefetch 0`。
  3. 减小推理时的 Context 长度（`-c` 参数），因为 KV Cache 的大小也会占用大量显存。

---

## 3. 性能与速度相关问题

### 3.1 开启 SWLP 后，生成速度 (TPS) 极慢或发生卡顿
* **可能原因 1**：没有开启异步迁移流水线。如果不开启异步，权重拷贝（H2D）会完全阻塞主线程的推理计算。
  * **解决方案**：自 2026-06-11 起，CUDA 后端编译下异步迁移会自动启用。如果您使用此版本之前的代码，需显式加上 `--swlp-async-migration 1`。如果日志显示 `SWLP: async migration disabled`，请确认：
    1. 使用 CUDA 后端编译（`-DGGML_CUDA=ON`）。
    2. 运行 `llama-swlp-bench` 时传递 `--async-migration` 参数（bench 工具不会自动启用）。
* **可能原因 2**：硬件物理限制（主板 PCIe 带宽极低）。SWLP 的核心原理是利用计算隐藏 PCIe 拷贝延迟，单通道 DDR4 内存或主板 PCIe 插槽运行在低速模式（如 PCIe 3.0 x2 或 x4）会导致传输瓶颈无法被掩盖。
  * **解决方案**：
    1. 确保显卡插在主板满速的主 PCIe 插槽中（PCIe 3.0/4.0 x16）。
    2. 检查 CPU 内存是否运行在双通道模式下（单通道 RAM 会极大地拉低从 CPU 侧读取权重的速度）。
* **可能原因 3**：窗口设置太小导致频繁发生 GPU 内存重分配。
  * **解决方案**：适当加大窗口（如 `--window 8`），这会在占用稍微多一点显存的前提下，大幅减少指针失效与显存分配的额外开销。

### 3.5 使用 `--swlp-auto`（自动窗口）后性能没有提升
* **现象**：开启 `--swlp-auto` 后，Prompt Processing 速度与纯 CPU 推理几乎相同（如 327 t/s vs 317 t/s），`--swlp-verbose` 无任何 SWLP 日志输出。
* **可能原因**：
  * **旧版本 bug**（2026-06-11 之前）：`llama_context` 初始化 SWLP 的创建条件为 `window_size > 0`，但 auto 模式使用 `window_size = -1` 作为标记值，`-1 > 0` 为 false，导致 SWLP 引擎从未被创建。
  * **解决方案**：更新代码至最新版本（commit `ecd4c35` 之后），此 bug 已修复。
* **验证方法**：使用 `--swlp-verbose` 运行，若看到以下日志则 SWLP 正确初始化：
  ```
  SWLP: analyzed model: ...
  SWLP: backend migration ready (GPU backend available)
  SWLP: window moved to [1, 10) for layer 9
  ```

### 3.3 swlp-bench 测试中 async migration 显示 disabled
* **现象**：使用 `llama-swlp-bench` 进行性能测试时，日志显示 `SWLP: async migration disabled`，即使有 CUDA 后端。
* **原因**：`llama-swlp-bench` 不会像主程序那样自动启用异步迁移，且该工具旧版完全未支持 `--async-migration` 参数。
* **解决方案**：
  1. 确认使用最新代码（包含 `--async-migration` 支持的版本）。
  2. 运行测试时加上 `--async-migration`：
     ```bash
     ./llama-swlp-bench model.gguf --window 8 --prefetch 1 --async-migration
     ```
  3. 日志应显示 `SWLP: async migration enabled`。

### 3.4 小模型上 SWLP Gen 速度反而低于纯 CPU
* **现象**：对于 <1B 的小模型，开启 SWLP 后生成（Generation）速度比纯 CPU 更慢（如 64 t/s vs 83 t/s）。
* **原因**：小模型每层计算时间极短（微秒级），而 SWLP 的 GPU 层迁移（evict/load）引入毫秒级开销。迁移开销 > GPU 加速收益，导致净性能下降。
* **推荐方案**：
  * 小模型（<1B）建议关闭 SWLP（`--window 0`），直接使用纯 CPU 或全 GPU 卸载（`--ngl 99`）。
  * 对于 3B+ 模型，SWLP 的 Gen 收益才会转为正数。模型越大，收益越明显。
