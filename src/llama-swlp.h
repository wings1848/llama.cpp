#pragma once

#include "llama.h"
#include <memory>
#include <vector>
#include <functional>

struct llama_swlp_state;

struct ggml_cgraph;
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
class llama_model;

#ifdef GGML_USE_CUDA
struct llama_swlp_cuda_graphs;
#endif

struct llama_swlp {
    llama_swlp(const llama_swlp_params & params, int num_layers);
    ~llama_swlp();

    llama_swlp(const llama_swlp &) = delete;
    llama_swlp & operator=(const llama_swlp &) = delete;

    bool is_enabled() const { return enabled; }
    bool is_verbose() const { return verbose; }
    bool has_prefetch() const;

    bool need_layer_move(int il) const;
    void prepare_layer(int il);

    void prepare_layer_experts(int il, int token_count);
    std::vector<int> predict_experts(int il, int top_k) const;
    void ensure_experts_cached(int il, const std::vector<int> & expert_ids);

    int  get_window_start() const;
    int  get_window_size() const;
    bool is_layer_gpu(int il) const;

    void annotate_graph(struct ggml_cgraph * gf, int n_layers);
    void annotate_expert_topk(int il, struct ggml_tensor * topk_tensor);
    void record_expert_activations();
    void on_split_begin(int split_idx);
    void prepare_migration();

    // Async migration: ensures all window-layer H2D copies are complete
    // before compute begins.  Call after prepare_migration() and before graph_compute.
    void ensure_window_ready();

    // Enable/disable async PCIe pipelining (double-buffer + CUDA streams).
    void set_async_migration(bool enable);
    bool is_async_migration() const;

    int64_t get_last_migration_us() const;
    int64_t get_last_compute_us() const;

    void set_backends(ggml_backend_t gpu, ggml_backend_t cpu);
    void set_model(const llama_model * model);
    void set_fixed_gpu_layers(int n_layers);
    void analyze_model(const llama_model & model);
    void auto_tune_adaptive_params();

    void for_each_layer_tensor(int il, const llama_model & model,
        std::function<void(ggml_tensor*)> fn) const;

    void record_layer_timing(int il, int64_t compute_us);
    void record_compute_time(int64_t compute_us);
    void record_forward_type(bool is_generation);
    void adapt_window();
    bool has_pending_window_change() const;
    void apply_pending_window();
    void print_stats() const;
    float get_ewma_alpha() const;
    int   get_adapt_interval() const;

    size_t estimate_active_memory_mb() const;
    size_t estimate_total_memory_mb() const;
    float  get_memory_reduction_ratio() const;

#ifdef GGML_USE_CUDA
    void enable_cuda_graphs();
    bool begin_capture_layer(int il, void * stream);
    bool end_capture_layer(int il, void * stream);
    bool replay_layer_graph(int il, void * stream) const;
    void invalidate_layer_graph(int il);
    bool is_layer_graph_valid(int il) const;
    void print_cuda_graph_stats() const;
#endif

private:
    bool enabled = false;
    bool verbose = false;
    std::unique_ptr<llama_swlp_state> state;
    std::vector<struct ggml_tensor *> expert_topk_tensors;

    // Migration helpers (extracted for clarity, see llama-swlp-migrate.cpp after split)
    void migrate_evict_layer(int il);
    void migrate_load_layer(int il, bool use_async);
    void migrate_evict_range(int start, int end, int w_start, int pf_end);
    void migrate_load_range(int start, int end, int w_start, int pf_end);

#ifdef GGML_USE_CUDA
    std::unique_ptr<llama_swlp_cuda_graphs> cuda_graphs;
#endif
};
