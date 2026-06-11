#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

// ============================================================
// Configuration
// ============================================================

struct bench_config {
    const char * model_path;
    int n_gpu_layers = 999;

    // SWLP params
    int  window_size      = 0;
    int  prefetch_depth   = 1;
    int  expert_cache     = 0;
    bool expert_prefetch  = false;
    bool adaptive         = false;
    bool use_pinned_copy  = false;
    float ewma_alpha      = 0.3f;
    bool  alpha_auto      = true;
    int   adapt_interval  = 0;   // 0 = auto

    // context
    int n_ctx     = 4096;
    int n_threads = 4;
    int n_batch   = 2048;

    // test modes
    int n_prompt_tokens = 0;   // 0 = use --prompt string; >0 = auto-generate
    int n_gen           = 0;   // 0 = prompt-processing only; >0 = generation mode
    int n_warmup        = 1;
    int n_iters         = 5;
    const char * prompt  = "Hello world, this is a test of the sliding window layer pipeline.";
    const char * out_file = nullptr; // JSON result output file
    bool verbose         = false;    // enable SWLP verbose logging
};

// ============================================================
// Help
// ============================================================

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s <model.gguf> [options]\n"
        "\n"
        "Model:\n"
        "  --ngl N             GPU layers (default: 999 = all)\n"
        "\n"
        "SWLP:\n"
        "  --window N          Window size (-1=auto, 0=off, default: 0)\n"
        "  --swlp-auto         Auto-select optimal window size\n"
        "  --prefetch N        Prefetch depth (default: 1)\n"
        "  --expert-cache N    MoE expert cache size (default: 0)\n"
        "  --expert-prefetch   Enable MoE expert prefetch\n"
        "  --adaptive          Enable adaptive window tuning\n"
        "  --pinned            Enable pinned memory for DMA\n"
        "  --swlp-alpha F      EWMA smoothing factor (0.1-1.0, default: auto)\n"
        "  --swlp-adapt-interval N  Adaptive adjustment interval in decodes (default: auto)\n"
        "  --swlp-verbose      Enable SWLP migration logging\n"
        "\n"
        "Context:\n"
        "  --ctx N             Context size (default: 4096)\n"
        "  --threads N         CPU threads (default: 4)\n"
        "  --batch N           Batch size (default: 2048)\n"
        "\n"
        "Test mode (choose one):\n"
        "  --prompt-tokens N   Auto-generate prompt of N tokens (default: uses --prompt)\n"
        "  --prompt STR        Custom prompt text\n"
        "  --gen N             Generation mode: generate N tokens, measure per-token latency\n"
        "\n"
        "Output:\n"
        "  --output PATH       Write JSON results to file (avoids stdout noise)\n"
        "  --warmup N          Warmup iterations (default: 1)\n"
        "  --iters N           Benchmark iterations (default: 5)\n"
        "  --help              Show this help\n",
        prog);
}

// ============================================================
// Argument parsing
// ============================================================

static bool parse_int_arg(int argc, char ** argv, int & i, const char * flag, int & out) {
    if (strcmp(argv[i], flag) == 0 && i + 1 < argc) {
        out = atoi(argv[++i]);
        return true;
    }
    return false;
}

static bool parse_float_arg(int argc, char ** argv, int & i, const char * flag, float & out) {
    if (strcmp(argv[i], flag) == 0 && i + 1 < argc) {
        out = (float)atof(argv[++i]);
        return true;
    }
    return false;
}

// ============================================================
// Statistics
// ============================================================

struct latency_stats {
    double mean_ms;
    double min_ms;
    double max_ms;
    double p50_ms;
    double p95_ms;
    double std_ms;
    double tps;
};

static latency_stats compute_stats(const std::vector<double> & values_ms, int n_tokens) {
    latency_stats s = {};
    if (values_ms.empty()) return s;

    int n = (int)values_ms.size();
    double sum = 0;
    s.min_ms = values_ms[0];
    s.max_ms = values_ms[0];
    for (double v : values_ms) {
        sum += v;
        if (v < s.min_ms) s.min_ms = v;
        if (v > s.max_ms) s.max_ms = v;
    }
    s.mean_ms = sum / n;

    // stddev
    double sq = 0;
    for (double v : values_ms) {
        sq += (v - s.mean_ms) * (v - s.mean_ms);
    }
    s.std_ms = std::sqrt(sq / n);

    // percentiles
    std::vector<double> sorted = values_ms;
    std::sort(sorted.begin(), sorted.end());
    s.p50_ms = sorted[n / 2];
    s.p95_ms = sorted[(int)(n * 0.95)];

    s.tps = (s.mean_ms > 0) ? (n_tokens / (s.mean_ms / 1000.0)) : 0;
    return s;
}

// ============================================================
// JSON output helper
// ============================================================

static void fprintf_json_str(FILE * f, const char * key, const char * val) {
    fprintf(f, "\"%s\":\"", key);
    for (const char * p = val; *p; p++) {
        switch (*p) {
            case '\\': fputs("\\\\", f); break;
            case '"':  fputs("\\\"", f); break;
            case '\n': fputs("\\n", f); break;
            case '\r': fputs("\\r", f); break;
            case '\t': fputs("\\t", f); break;
            default:   fputc(*p, f); break;
        }
    }
    fputc('"', f);
}
static void fprintf_json_int(FILE * f, const char * key, int64_t val) {
    fprintf(f, "\"%s\":%lld", key, (long long)val);
}
static void fprintf_json_dbl(FILE * f, const char * key, double val) {
    fprintf(f, "\"%s\":%.2f", key, val);
}
static void fprintf_json_bool(FILE * f, const char * key, bool val) {
    fprintf(f, "\"%s\":%s", key, val ? "true" : "false");
}

// ============================================================
// Prompt generation
// ============================================================

static std::string generate_prompt(int target_tokens, const llama_vocab * vocab) {
    // Generate a prompt of approximately target_tokens by tokenizing
    // a long text and truncating to the desired token count.
    const char * base =
        "The quick brown fox jumps over the lazy dog. "
        "Artificial intelligence and machine learning are transforming the world. "
        "Large language models can process natural language, generate creative content, "
        "and assist with complex reasoning tasks across many domains. "
        "The transformer architecture, introduced in 2017, uses self-attention mechanisms "
        "to capture long-range dependencies in sequential data. "
        "Modern GPUs accelerate the matrix multiplications at the core of deep learning. "
        "Software optimization techniques like kernel fusion, quantization, and memory "
        "management are critical for efficient inference deployment. ";
    // Repeat base text until we have enough characters
    std::string text;
    while ((int)text.size() < target_tokens * 6) { // ~6 chars per token
        text += base;
    }
    // Tokenize and truncate
    std::vector<llama_token> tokens(target_tokens * 2 + 128);
    int n = llama_tokenize(vocab, text.c_str(), (int)text.size(), tokens.data(),
                           (int)tokens.size(), true, false);
    if (n <= 0) return std::string(base, 256); // fallback
    if (n > target_tokens) n = target_tokens;

    // Detokenize back to get the prompt text
    std::string prompt;
    prompt.reserve(n * 8);
    for (int i = 0; i < n; i++) {
        char buf[64];
        int len = llama_token_to_piece(vocab, tokens[i], buf, sizeof(buf), 0, false);
        if (len > 0) prompt.append(buf, len);
    }
    return prompt;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    bench_config cfg;
    cfg.model_path = argv[1];

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0)           { print_usage(argv[0]); return 0; }
        else if (strcmp(argv[i], "--expert-prefetch") == 0) { cfg.expert_prefetch = true; }
        else if (strcmp(argv[i], "--adaptive") == 0)       { cfg.adaptive = true; }
        else if (strcmp(argv[i], "--pinned") == 0)         { cfg.use_pinned_copy = true; }
        else if (strcmp(argv[i], "--swlp-verbose") == 0)   { cfg.verbose = true; }
        else if (parse_int_arg(argc, argv, i, "--ngl", cfg.n_gpu_layers)) {}
        else if (parse_int_arg(argc, argv, i, "--window", cfg.window_size)) {}
        else if (parse_int_arg(argc, argv, i, "--prefetch", cfg.prefetch_depth)) {}
        else if (parse_int_arg(argc, argv, i, "--expert-cache", cfg.expert_cache)) {}
        else if (strcmp(argv[i], "--swlp-auto") == 0)       { cfg.window_size = -1; }
        else if (parse_int_arg(argc, argv, i, "--ctx", cfg.n_ctx)) {}
        else if (parse_int_arg(argc, argv, i, "--threads", cfg.n_threads)) {}
        else if (parse_int_arg(argc, argv, i, "--batch", cfg.n_batch)) {}
        else if (parse_int_arg(argc, argv, i, "--warmup", cfg.n_warmup)) {}
        else if (parse_int_arg(argc, argv, i, "--iters", cfg.n_iters)) {}
        else if (parse_int_arg(argc, argv, i, "--gen", cfg.n_gen)) {}
        else if (parse_int_arg(argc, argv, i, "--prompt-tokens", cfg.n_prompt_tokens)) {}
        else if (parse_float_arg(argc, argv, i, "--swlp-alpha", cfg.ewma_alpha)) {
            cfg.alpha_auto = false;  // explicit alpha disables auto
        }
        else if (parse_int_arg(argc, argv, i, "--swlp-adapt-interval", cfg.adapt_interval)) {}
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            cfg.out_file = argv[++i];
        }
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            cfg.prompt = argv[++i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    // ============================================================
    // Init
    // ============================================================
    llama_backend_init();

    // Load model
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(cfg.model_path, mparams);
    if (!model) {
        fprintf(stderr, "ERROR: Failed to load model\n");
        llama_backend_free();
        return 1;
    }

    int n_layers = llama_model_n_layer(model);
    uint64_t model_size = llama_model_size(model);
    const auto & vocab = llama_model_get_vocab(model);

    // Clamp window
    // -1 = auto, passed through to llama_swlp_params for runtime computation
    int effective_window = cfg.window_size;
    if (effective_window > n_layers) effective_window = n_layers;
    if (effective_window < 0 && effective_window != -1) effective_window = 0;

    // Resolve prompt
    std::string resolved_prompt;
    if (cfg.n_prompt_tokens > 0) {
        resolved_prompt = generate_prompt(cfg.n_prompt_tokens, vocab);
        cfg.prompt = resolved_prompt.c_str();
    }
    // Note: cfg.prompt always valid -- either points to default literal or resolved_prompt

    // Print header
    fprintf(stderr, "SWLP Bench\n");
    fprintf(stderr, "  Model:      %s\n", cfg.model_path);
    fprintf(stderr, "  Layers:     %d\n", n_layers);
    fprintf(stderr, "  GPU layers: %d\n", cfg.n_gpu_layers);
    fprintf(stderr, "  SWLP:       %s\n", effective_window != 0 ? "ON" : "OFF");
    if (effective_window > 0) {
        fprintf(stderr, "  Window:     %d\n", effective_window);
        fprintf(stderr, "  Prefetch:   %d\n", cfg.prefetch_depth);
        fprintf(stderr, "  Adaptive:   %s\n", cfg.adaptive ? "ON" : "OFF");
        if (cfg.adaptive) {
            fprintf(stderr, "  EWMA Alpha: %.2f%s\n", cfg.ewma_alpha,
                cfg.alpha_auto ? " (auto)" : "");
            fprintf(stderr, "  Adapt Int:  %d%s\n", cfg.adapt_interval,
                cfg.adapt_interval == 0 ? " (auto)" : "");
        }
        fprintf(stderr, "  Pinned:     %s\n", cfg.use_pinned_copy ? "ON" : "OFF");
    }
    if (cfg.expert_cache > 0) {
        fprintf(stderr, "  ExpertCache: %d (NOTE: currently no-op, see docs-zh/llama.cpp-swlp综合测试报告v4.md)\n",
            cfg.expert_cache);
    }
    fprintf(stderr, "  Mode:       %s\n", cfg.n_gen > 0 ? "generation" : "prompt-processing");
    if (cfg.n_gen > 0) {
        fprintf(stderr, "  Gen tokens: %d\n", cfg.n_gen);
    }
    fprintf(stderr, "  Warmup:     %d\n", cfg.n_warmup);
    fprintf(stderr, "  Iters:      %d\n", cfg.n_iters);
    fprintf(stderr, "\n");

    // ============================================================
    // Context
    // ============================================================
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = cfg.n_ctx;
    cparams.n_threads = cfg.n_threads;
    cparams.n_batch   = cfg.n_batch;

    if (effective_window != 0) {  // 0=off, -1=auto, >0=explicit
        cparams.swlp.window_size      = effective_window;
        cparams.swlp.prefetch_depth   = cfg.prefetch_depth;
        cparams.swlp.expert_cache_size = cfg.expert_cache;
        cparams.swlp.expert_prefetch  = cfg.expert_prefetch;
        cparams.swlp.adaptive         = cfg.adaptive;
        cparams.swlp.use_pinned_copy  = cfg.use_pinned_copy;
        cparams.swlp.verbose          = cfg.verbose;
        cparams.swlp.ewma_alpha       = cfg.ewma_alpha;
        cparams.swlp.alpha_auto       = cfg.alpha_auto;
        cparams.swlp.adapt_interval   = cfg.adapt_interval;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "ERROR: Failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // ============================================================
    // Tokenize
    // ============================================================
    int prompt_chars = (int)strlen(cfg.prompt);
    std::vector<llama_token> tokens(prompt_chars + 256);
    int n_prompt = llama_tokenize(vocab, cfg.prompt, prompt_chars,
                                  tokens.data(), (int)tokens.size(), true, false);
    if (n_prompt <= 0) {
        fprintf(stderr, "ERROR: Tokenization failed\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    fprintf(stderr, "  Prompt tokens: %d\n\n", n_prompt);

    // ============================================================
    // Prompt processing (always)
    // ============================================================
    llama_batch pp_batch = llama_batch_get_one(tokens.data(), n_prompt);

    // Warmup (prompt processing)
    for (int i = 0; i < cfg.n_warmup; i++) {
        llama_decode(ctx, pp_batch);
    }

    // Benchmark prompt processing
    std::vector<double> pp_ms_list;
    pp_ms_list.reserve(cfg.n_iters);
    for (int i = 0; i < cfg.n_iters; i++) {
        int64_t t0 = ggml_time_us();
        int ret = llama_decode(ctx, pp_batch);
        int64_t t1 = ggml_time_us();
        if (ret != 0 && ret != 1) {
            fprintf(stderr, "ERROR: decode returned %d on iter %d\n", ret, i);
        }
        pp_ms_list.push_back((t1 - t0) / 1000.0);
    }

    latency_stats pp_stats = compute_stats(pp_ms_list, n_prompt);

    // ============================================================
    // Generation (if requested)
    // ============================================================
    std::vector<double> gen_ms_list;
    latency_stats gen_stats;

    if (cfg.n_gen > 0) {
        gen_ms_list.reserve(cfg.n_gen);

        int n_vocab_local = llama_vocab_n_tokens(llama_model_get_vocab(model));

        // Greedy sampling: argmax over logits
        auto sample_greedy = [&]() -> llama_token {
            float * logits = llama_get_logits(ctx);
            if (!logits) return 0;
            llama_token best_tok = 0;
            float best_val = -1e30f;
            for (int j = 0; j < n_vocab_local; j++) {
                if (logits[j] > best_val) {
                    best_val = logits[j];
                    best_tok = j;
                }
            }
            return best_tok;
        };

        // First token: warmup, not timed
        llama_token tok = sample_greedy();
        llama_batch gen_batch = llama_batch_get_one(&tok, 1);
        llama_decode(ctx, gen_batch);

        for (int i = 0; i < cfg.n_gen; i++) {
            tok = sample_greedy();
            gen_batch = llama_batch_get_one(&tok, 1);

            int64_t t0 = ggml_time_us();
            int ret = llama_decode(ctx, gen_batch);
            int64_t t1 = ggml_time_us();
            if (ret != 0 && ret != 1) {
                fprintf(stderr, "ERROR: gen decode returned %d at token %d\n", ret, i);
                break;
            }
            gen_ms_list.push_back((t1 - t0) / 1000.0);
        }
        gen_stats = compute_stats(gen_ms_list, 1);
    }

    // ============================================================
    // Output results
    // ============================================================
    FILE * out = cfg.out_file ? fopen(cfg.out_file, "w") : stdout;
    if (!out) {
        fprintf(stderr, "ERROR: Cannot open output file: %s\n", cfg.out_file);
        out = stdout;
    }

    fprintf(out, "{\n");
    fprintf(out, "  ");
    fprintf_json_str(out, "model_path", cfg.model_path);       fprintf(out, ",\n  ");
    fprintf_json_dbl(out, "model_size_mb", model_size / (1024.0*1024.0)); fprintf(out, ",\n  ");
    fprintf_json_int(out, "n_layers", n_layers);               fprintf(out, ",\n  ");
    fprintf_json_int(out, "n_gpu_layers", cfg.n_gpu_layers);   fprintf(out, ",\n  ");
    fprintf_json_int(out, "window_size", effective_window);    fprintf(out, ",\n  ");
    fprintf_json_int(out, "prefetch_depth", cfg.prefetch_depth); fprintf(out, ",\n  ");
    fprintf_json_int(out, "expert_cache", cfg.expert_cache);   fprintf(out, ",\n  ");
    fprintf_json_bool(out, "expert_prefetch", cfg.expert_prefetch); fprintf(out, ",\n  ");
    fprintf_json_bool(out, "adaptive", cfg.adaptive);          fprintf(out, ",\n  ");
    // Note: these are config values; auto-tuned values may differ (see SWLP verbose log)
    fprintf_json_dbl(out, "cfg_ewma_alpha", cfg.ewma_alpha);    fprintf(out, ",\n  ");
    fprintf_json_bool(out, "cfg_alpha_auto", cfg.alpha_auto);    fprintf(out, ",\n  ");
    fprintf_json_int(out, "cfg_adapt_interval", cfg.adapt_interval); fprintf(out, ",\n  ");
    fprintf_json_bool(out, "use_pinned_copy", cfg.use_pinned_copy); fprintf(out, ",\n  ");
    fprintf_json_int(out, "n_prompt_tokens", n_prompt);        fprintf(out, ",\n  ");
    fprintf_json_int(out, "n_iters", cfg.n_iters);             fprintf(out, ",\n  ");
    fprintf_json_int(out, "n_gen", cfg.n_gen);                 fprintf(out, ",\n  ");

    // Prompt processing stats
    fprintf_json_dbl(out, "pp_mean_ms", pp_stats.mean_ms);    fprintf(out, ",\n  ");
    fprintf_json_dbl(out, "pp_min_ms", pp_stats.min_ms);      fprintf(out, ",\n  ");
    fprintf_json_dbl(out, "pp_max_ms", pp_stats.max_ms);      fprintf(out, ",\n  ");
    fprintf_json_dbl(out, "pp_p50_ms", pp_stats.p50_ms);      fprintf(out, ",\n  ");
    fprintf_json_dbl(out, "pp_p95_ms", pp_stats.p95_ms);      fprintf(out, ",\n  ");
    fprintf_json_dbl(out, "pp_std_ms", pp_stats.std_ms);      fprintf(out, ",\n  ");
    fprintf_json_dbl(out, "pp_tps", pp_stats.tps);            fprintf(out, ",\n  ");

    // Per-iteration prompt processing times
    fprintf(out, "\"pp_iter_ms\":[");
    for (size_t i = 0; i < pp_ms_list.size(); i++) {
        if (i > 0) fprintf(out, ",");
        fprintf(out, "%.2f", pp_ms_list[i]);
    }
    fprintf(out, "]");

    // Generation stats
    if (cfg.n_gen > 0) {
        fprintf(out, ",\n  ");
        fprintf_json_dbl(out, "gen_mean_ms", gen_stats.mean_ms); fprintf(out, ",\n  ");
        fprintf_json_dbl(out, "gen_min_ms", gen_stats.min_ms);   fprintf(out, ",\n  ");
        fprintf_json_dbl(out, "gen_max_ms", gen_stats.max_ms);   fprintf(out, ",\n  ");
        fprintf_json_dbl(out, "gen_p50_ms", gen_stats.p50_ms);   fprintf(out, ",\n  ");
        fprintf_json_dbl(out, "gen_p95_ms", gen_stats.p95_ms);   fprintf(out, ",\n  ");
        fprintf_json_dbl(out, "gen_std_ms", gen_stats.std_ms);   fprintf(out, ",\n  ");
        fprintf_json_dbl(out, "gen_tps", gen_stats.tps);         fprintf(out, ",\n  ");

        fprintf(out, "\"gen_iter_ms\":[");
        for (size_t i = 0; i < gen_ms_list.size(); i++) {
            if (i > 0) fprintf(out, ",");
            fprintf(out, "%.2f", gen_ms_list[i]);
        }
        fprintf(out, "]");
    }

    fprintf(out, "\n}\n");

    if (cfg.out_file) {
        fclose(out);
        fprintf(stderr, "Results written to: %s\n", cfg.out_file);
    }

    // ============================================================
    // Cleanup
    // ============================================================
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
