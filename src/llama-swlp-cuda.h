#pragma once

#ifdef GGML_USE_CUDA

#include <cstddef>
#include <vector>
#include <memory>

struct llama_swlp_cuda_graphs_impl;

struct llama_swlp_cuda_graphs {
    llama_swlp_cuda_graphs(int n_layers);
    ~llama_swlp_cuda_graphs();

    // Capture must be called as a begin/end pair around actual kernel launches:
    //   graphs->begin_capture(il, stream);
    //   // submit layer-il work to stream ...
    //   graphs->end_capture(il, stream);
    bool begin_capture(int il, void * cuda_stream);
    bool end_capture(int il, void * cuda_stream);

    bool replay_layer_graph(int il, void * cuda_stream);
    void invalidate_graph(int il);
    bool is_graph_valid(int il) const;
    bool is_graph_captured(int il) const;
    bool all_valid() const;
    int num_captured() const;
    void print_stats() const;

    // Embed graph capture (same begin/end pattern)
    bool begin_capture_embed(void * cuda_stream);
    bool end_capture_embed(void * cuda_stream);
    bool replay_embed_graph(void * cuda_stream);
    void invalidate_embed_graph();

    // Output graph capture (same begin/end pattern)
    bool begin_capture_output(void * cuda_stream);
    bool end_capture_output(void * cuda_stream);
    bool replay_output_graph(void * cuda_stream);
    void invalidate_output_graph();

    void * get_graph_stream() const;

private:
    std::unique_ptr<llama_swlp_cuda_graphs_impl> impl;
};

#endif // GGML_USE_CUDA
