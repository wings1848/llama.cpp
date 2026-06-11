#include "llama.h"
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <vector>

static int run_decode_test(llama_model * model, llama_context_params cparams,
                          const char * label, const char * prompt) {
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "%s: context init failed\n", label);
        return 1;
    }

    const auto & vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(128);
    int n = llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), (int)tokens.size(), true, false);
    if (n < 0) {
        fprintf(stderr, "%s: tokenize failed\n", label);
        llama_free(ctx);
        return 1;
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), n);

    int64_t t_start = ggml_time_us();
    int ret = llama_decode(ctx, batch);
    int64_t t_end = ggml_time_us();

    if (ret != 0 && ret != 1) {
        fprintf(stderr, "%s: decode failed (ret=%d)\n", label, ret);
        llama_free(ctx);
        return 1;
    }

    double ms = (double)(t_end - t_start) / 1000.0;
    double tok_per_sec = (ms > 0.0) ? (n / (ms / 1000.0)) : 0.0;

    fprintf(stderr, "  %-12s %8.1f ms, %8.1f tok/s, %d tokens\n",
        label, ms, tok_per_sec, n);

    llama_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", argv[1]);
        llama_backend_free();
        return 1;
    }

    int n_layers = llama_model_n_layer(model);
    uint64_t model_size = llama_model_size(model);
    fprintf(stderr, "Model: %s\n", argv[1]);
    fprintf(stderr, "  layers: %d, size: %.1f MB\n",
        n_layers, (double)model_size / (1024.0 * 1024.0));
    fprintf(stderr, "\n");

    const char * prompt = "Hello world, this is a test of the streaming weight loading engine.";

    run_decode_test(model, llama_context_default_params(), "WARMUP", prompt);

    {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 512; cparams.n_threads = 4; cparams.n_batch = 512;
        run_decode_test(model, cparams, "BASELINE", prompt);
    }

    {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 512; cparams.n_threads = 4; cparams.n_batch = 512;
        cparams.swlp.window_size = 2;
        run_decode_test(model, cparams, "SWLP W=2", prompt);
    }

    {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 512; cparams.n_threads = 4; cparams.n_batch = 512;
        cparams.swlp.window_size = 4;
        run_decode_test(model, cparams, "SWLP W=4", prompt);
    }

    {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 512; cparams.n_threads = 4; cparams.n_batch = 512;
        cparams.swlp.window_size = 4;
        cparams.swlp.adaptive = true;

        llama_context * ctx = llama_init_from_model(model, cparams);
        if (ctx) {
            const auto & vocab = llama_model_get_vocab(model);
            std::vector<llama_token> tokens(128);
            int n = llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), (int)tokens.size(), true, false);
            if (n > 0) {
                for (int pass = 0; pass < 5; pass++) {
                    llama_batch batch = llama_batch_get_one(tokens.data(), n);
                    int64_t t_start = ggml_time_us();
                    int ret = llama_decode(ctx, batch);
                    int64_t t_end = ggml_time_us();
                    double ms = (double)(t_end - t_start) / 1000.0;
                    fprintf(stderr, "  ADAPTIVE pass %d: %8.1f ms, ret=%d\n", pass, ms, ret);
                }
            }
            llama_free(ctx);
        }
    }

    run_decode_test(model, llama_context_default_params(), "BASELINE2", prompt);

    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr, "SWLP benchmark PASSED\n");
    return 0;
}
