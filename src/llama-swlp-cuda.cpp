#ifdef GGML_USE_CUDA

#include "llama-swlp-cuda.h"
#include "llama-impl.h"
#include "ggml.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

// ============================================================
// Implementation (PIMPL)
// ============================================================

struct llama_swlp_cuda_graphs_impl {
    int num_layers;

    // Per-layer graph state
    std::vector<cudaGraphExec_t> layer_exec;
    std::vector<bool>            graph_valid;
    std::vector<bool>            graph_captured;

    // In-progress capture state (one at a time)
    cudaStream_t capture_stream  = nullptr;
    int          capture_layer   = -1;

    // Embed / output graphs
    cudaGraphExec_t embed_exec   = nullptr;
    cudaGraphExec_t output_exec  = nullptr;

    // Statistics
    int graphs_captured    = 0;
    int graphs_invalidated = 0;
    int graphs_replayed    = 0;

    bool cuda_available = false;

    llama_swlp_cuda_graphs_impl(int n_layers)
        : num_layers(n_layers)
        , layer_exec(n_layers, nullptr)
        , graph_valid(n_layers, false)
        , graph_captured(n_layers, false)
    {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        cuda_available = (err == cudaSuccess && count > 0);
        if (!cuda_available) {
            LLAMA_LOG_WARN("SWLP CUDA: no CUDA devices available, graphs disabled\n");
        }
    }

    ~llama_swlp_cuda_graphs_impl() {
        for (int i = 0; i < num_layers; i++) {
            if (layer_exec[i]) {
                cudaGraphExecDestroy(layer_exec[i]);
                layer_exec[i] = nullptr;
            }
        }
        if (embed_exec)  { cudaGraphExecDestroy(embed_exec);  embed_exec  = nullptr; }
        if (output_exec) { cudaGraphExecDestroy(output_exec); output_exec = nullptr; }
    }

    bool begin_capture(int il, cudaStream_t stream) {
        if (!cuda_available || !stream) return false;
        if (capture_stream != nullptr) {
            LLAMA_LOG_WARN("SWLP CUDA: capture already in progress (layer %d), ignoring\n",
                capture_layer);
            return false;
        }

        cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            LLAMA_LOG_WARN("SWLP CUDA: begin_capture layer %d failed: %s\n",
                il, cudaGetErrorString(err));
            return false;
        }

        capture_stream = stream;
        capture_layer  = il;
        return true;
    }

    bool end_capture(int il, cudaStream_t stream) {
        if (!cuda_available || !stream) return false;
        if (capture_stream != stream || capture_layer != il) {
            LLAMA_LOG_WARN("SWLP CUDA: end_capture mismatch: expected layer %d, got %d\n",
                capture_layer, il);
            return false;
        }

        cudaGraph_t graph = nullptr;
        cudaError_t err = cudaStreamEndCapture(stream, &graph);
        capture_stream = nullptr;
        capture_layer  = -1;

        if (err != cudaSuccess) {
            LLAMA_LOG_WARN("SWLP CUDA: end_capture layer %d failed: %s\n",
                il, cudaGetErrorString(err));
            return false;
        }

        cudaGraphExec_t exec = nullptr;
        err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        cudaGraphDestroy(graph);

        if (err != cudaSuccess) {
            LLAMA_LOG_WARN("SWLP CUDA: graph instantiate layer %d failed: %s\n",
                il, cudaGetErrorString(err));
            return false;
        }

        // Replace previous exec
        if (layer_exec[il]) {
            cudaGraphExecDestroy(layer_exec[il]);
        }
        layer_exec[il]    = exec;
        graph_captured[il] = true;
        graph_valid[il]   = true;
        graphs_captured++;

        LLAMA_LOG_DEBUG("SWLP CUDA: layer %d graph captured\n", il);
        return true;
    }

    bool replay(cudaGraphExec_t exec, cudaStream_t stream) {
        if (!cuda_available || !exec || !stream) return false;

        cudaError_t err = cudaGraphLaunch(exec, stream);
        if (err != cudaSuccess) {
            LLAMA_LOG_WARN("SWLP CUDA: graph launch failed: %s\n",
                cudaGetErrorString(err));
            return false;
        }
        return true;
    }
};

// ============================================================
// Public interface
// ============================================================

llama_swlp_cuda_graphs::llama_swlp_cuda_graphs(int n_layers) {
    impl = std::make_unique<llama_swlp_cuda_graphs_impl>(n_layers);

    LLAMA_LOG_INFO("SWLP CUDA: per-layer graph manager initialized (%d layers, %s)\n",
        n_layers, impl->cuda_available ? "CUDA available" : "CUDA unavailable");
}

llama_swlp_cuda_graphs::~llama_swlp_cuda_graphs() = default;

bool llama_swlp_cuda_graphs::begin_capture(int il, void * cuda_stream) {
    if (!impl || il < 0 || il >= impl->num_layers) return false;
    return impl->begin_capture(il, (cudaStream_t)cuda_stream);
}

bool llama_swlp_cuda_graphs::end_capture(int il, void * cuda_stream) {
    if (!impl || il < 0 || il >= impl->num_layers) return false;
    return impl->end_capture(il, (cudaStream_t)cuda_stream);
}

bool llama_swlp_cuda_graphs::replay_layer_graph(int il, void * cuda_stream) {
    if (!impl || il < 0 || il >= impl->num_layers) return false;
    if (!impl->graph_valid[il]) return false;
    if (!cuda_stream) return false;

    cudaStream_t stream = (cudaStream_t)cuda_stream;
    bool ok = impl->replay(impl->layer_exec[il], stream);
    if (ok) { impl->graphs_replayed++; }
    return ok;
}

void llama_swlp_cuda_graphs::invalidate_graph(int il) {
    if (!impl || il < 0 || il >= impl->num_layers) return;

    if (impl->graph_valid[il]) {
        impl->graph_valid[il] = false;
        impl->graph_captured[il] = false;
        impl->graphs_invalidated++;

        if (impl->layer_exec[il]) {
            cudaGraphExecDestroy(impl->layer_exec[il]);
            impl->layer_exec[il] = nullptr;
        }
    }
}

bool llama_swlp_cuda_graphs::is_graph_valid(int il) const {
    if (!impl || il < 0 || il >= impl->num_layers) return false;
    return impl->graph_valid[il];
}

bool llama_swlp_cuda_graphs::is_graph_captured(int il) const {
    if (!impl || il < 0 || il >= impl->num_layers) return false;
    return impl->graph_captured[il];
}

bool llama_swlp_cuda_graphs::all_valid() const {
    if (!impl) return false;
    for (int i = 0; i < impl->num_layers; i++) {
        if (!impl->graph_valid[i]) return false;
    }
    return true;
}

int llama_swlp_cuda_graphs::num_captured() const {
    return impl ? impl->graphs_captured : 0;
}

void llama_swlp_cuda_graphs::print_stats() const {
    if (!impl) return;

    LLAMA_LOG_INFO("SWLP CUDA Graphs: %d/%d layers captured, %d invalidated, %d replayed\n",
        impl->graphs_captured, impl->num_layers,
        impl->graphs_invalidated, impl->graphs_replayed);
    LLAMA_LOG_INFO("  all valid: %s, cuda: %s\n",
        all_valid() ? "yes" : "no",
        impl->cuda_available ? "available" : "unavailable");
}

bool llama_swlp_cuda_graphs::begin_capture_embed(void * cuda_stream) {
    if (!impl || !cuda_stream) return false;

    cudaStream_t stream = (cudaStream_t)cuda_stream;
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        LLAMA_LOG_WARN("SWLP CUDA: begin_capture_embed failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    impl->capture_stream = stream;
    impl->capture_layer  = -2; // sentinel for embed
    return true;
}

bool llama_swlp_cuda_graphs::end_capture_embed(void * cuda_stream) {
    if (!impl || !cuda_stream) return false;
    if (impl->capture_layer != -2) {
        LLAMA_LOG_WARN("SWLP CUDA: end_capture_embed without matching begin (layer=%d)\n", impl->capture_layer);
        return false;
    }

    cudaStream_t stream = (cudaStream_t)cuda_stream;
    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    impl->capture_stream = nullptr;
    impl->capture_layer  = -1;

    if (err != cudaSuccess) {
        LLAMA_LOG_WARN("SWLP CUDA: end_capture_embed failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);

    if (err != cudaSuccess) {
        LLAMA_LOG_WARN("SWLP CUDA: embed graph instantiate failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    if (impl->embed_exec) { cudaGraphExecDestroy(impl->embed_exec); }
    impl->embed_exec = exec;
    return true;
}

bool llama_swlp_cuda_graphs::replay_embed_graph(void * cuda_stream) {
    if (!impl || !impl->embed_exec || !cuda_stream) return false;

    cudaStream_t stream = (cudaStream_t)cuda_stream;
    return impl->replay(impl->embed_exec, stream);
}

void llama_swlp_cuda_graphs::invalidate_embed_graph() {
    if (!impl || !impl->embed_exec) return;
    cudaGraphExecDestroy(impl->embed_exec);
    impl->embed_exec = nullptr;
}

bool llama_swlp_cuda_graphs::begin_capture_output(void * cuda_stream) {
    if (!impl || !cuda_stream) return false;

    cudaStream_t stream = (cudaStream_t)cuda_stream;
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        LLAMA_LOG_WARN("SWLP CUDA: begin_capture_output failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    impl->capture_stream = stream;
    impl->capture_layer  = -3; // sentinel for output
    return true;
}

bool llama_swlp_cuda_graphs::end_capture_output(void * cuda_stream) {
    if (!impl || !cuda_stream) return false;
    if (impl->capture_layer != -3) {
        LLAMA_LOG_WARN("SWLP CUDA: end_capture_output without matching begin (layer=%d)\n", impl->capture_layer);
        return false;
    }

    cudaStream_t stream = (cudaStream_t)cuda_stream;
    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    impl->capture_stream = nullptr;
    impl->capture_layer  = -1;

    if (err != cudaSuccess) {
        LLAMA_LOG_WARN("SWLP CUDA: end_capture_output failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);

    if (err != cudaSuccess) {
        LLAMA_LOG_WARN("SWLP CUDA: output graph instantiate failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    if (impl->output_exec) { cudaGraphExecDestroy(impl->output_exec); }
    impl->output_exec = exec;
    return true;
}

bool llama_swlp_cuda_graphs::replay_output_graph(void * cuda_stream) {
    if (!impl || !impl->output_exec || !cuda_stream) return false;

    cudaStream_t stream = (cudaStream_t)cuda_stream;
    return impl->replay(impl->output_exec, stream);
}

void llama_swlp_cuda_graphs::invalidate_output_graph() {
    if (!impl || !impl->output_exec) return;
    cudaGraphExecDestroy(impl->output_exec);
    impl->output_exec = nullptr;
}

void * llama_swlp_cuda_graphs::get_graph_stream() const {
    return nullptr;
}

#endif // GGML_USE_CUDA
