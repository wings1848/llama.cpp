#include "llama-swlp-moe.h"
#include "llama-impl.h"
#include "llama-swlp-internal.h"

#include <algorithm>

std::vector<int> llama_swlp::predict_experts(int il, int top_k) const {
    if (!enabled || state->expert_cache_size <= 0) return {};
    if (il < 0 || il >= state->num_layers) return {};
    if (state->expert_count_per_layer[il] == 0) return {};
    if (state->activation_counts[il].empty()) return {};

    std::vector<std::pair<int64_t, int>> ranked;
    ranked.reserve(state->activation_counts[il].size());
    for (int e = 0; e < (int)state->activation_counts[il].size(); e++) {
        ranked.push_back({state->activation_counts[il][e], e});
    }
    int k = std::min(top_k, (int)ranked.size());
    std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(),
        [](const auto & a, const auto & b) { return a.first > b.first; });

    std::vector<int> result;
    for (int i = 0; i < std::min(top_k, (int)ranked.size()); i++) {
        result.push_back(ranked[i].second);
    }
    return result;
}

void llama_swlp::ensure_experts_cached(int il, const std::vector<int> & expert_ids) {
    if (!enabled || state->expert_cache_size <= 0) return;
    if (il < 0 || il >= state->num_layers) return;
    if (state->expert_count_per_layer[il] == 0) return;

    auto & cache = state->expert_in_cache[il];

    int cached_count = 0;
    for (int e = 0; e < (int)cache.size(); e++) {
        if (cache[e]) cached_count++;
    }

    for (int eid : expert_ids) {
        if (eid < 0 || eid >= (int)cache.size()) continue;

        if (!cache[eid]) {
            if (cached_count >= state->expert_cache_size) {
                // LRU-like eviction: evict non-active expert with lowest activation count
                int lru_idx = -1;
                int64_t lru_val = INT64_MAX;

                for (int e = 0; e < (int)cache.size(); e++) {
                    if (!cache[e]) continue;
                    bool is_active = false;
                    for (int aid : expert_ids) {
                        if (e == aid) { is_active = true; break; }
                    }
                    if (is_active) continue;

                    if (state->activation_counts[il][e] < lru_val) {
                        lru_val = state->activation_counts[il][e];
                        lru_idx = e;
                    }
                }

                if (lru_idx >= 0) {
                    cache[lru_idx] = false;
                    cached_count--;
                }
            }

            cache[eid] = true;
            cached_count++;
            state->expert_cache_misses++;
            state->expert_bytes_streamed += state->expert_sizes[il][eid];
        } else {
            state->expert_cache_hits++;
        }
    }
}

void llama_swlp::prepare_layer_experts(int il, int token_count) {
    // Disabled until per-expert tensor splitting is implemented.
    // See git history for the planned implementation.
    GGML_UNUSED(il);
    GGML_UNUSED(token_count);
}
