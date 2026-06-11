# Code Context

## Files Retrieved

1. `src/llama-graph.cpp` (lines 1409-1790) — Core `build_moe_ffn` function; the single entry point for all MoE graph construction
2. `src/llama-graph.h` (lines 580-630, 790-930) — `llm_graph_cb` type, `n_expert/n_expert_used` members, `build_moe_ffn` declarations
3. `src/llama-model.cpp` (lines 2666-2680) — `create_tensor_gate_up_exps` showing tensor shapes for stacked expert tensors
4. `src/llama-model.h` (lines 289-350, 430-445) — All `ffn_gate_*` / `ffn_*_exps` tensor fields on `llama_layer`
5. `src/models/llama.cpp` (lines 73-92, 195-225) — Canonical example of expert tensor creation and `build_moe_ffn` call site
6. `src/llama-arch.cpp` (lines 365-376) — Tensor name registry for all MoE tensors
7. `src/llama-swlp.cpp` (lines 95-130, 300-340, 990-1075) — `activation_counts` data structure, `predict_experts`, `ensure_experts_cached`, with FIXME acknowledging the missing hook
8. `src/llama-swlp.h` (lines 25-75) — `prepare_layer_experts`, `predict_experts` public API
9. `src/llama-context.cpp` (lines 2476-2478) — Where `prepare_layer_experts` is called during graph annotation callback

---

## Key Code

### Tensor shapes (from `src/models/llama.cpp` lines 79-87)

```cpp
// ffn_gate_inp: router weights — maps embeddings to expert scores
layer.ffn_gate_inp  = create_tensor(..., {n_embd, n_expert}, 0);
// ffn_gate_exps: per-expert gate projection — stacked along 3rd dim
layer.ffn_gate_exps = create_tensor(..., {n_embd, n_ff, n_expert}, TENSOR_NOT_REQUIRED);
// ffn_down_exps: per-expert down projection — stacked along 3rd dim
layer.ffn_down_exps = create_tensor(..., {n_ff, n_embd, n_expert}, 0);
// ffn_up_exps: per-expert up projection — stacked along 3rd dim
layer.ffn_up_exps   = create_tensor(..., {n_embd, n_ff, n_expert}, 0);
```

Merged gate+up variant (`src/llama-model.cpp` line 2670):
```cpp
layer.ffn_gate_up_exps = create_tensor(..., {n_embd, n_ff * 2, n_expert}, TENSOR_NOT_REQUIRED);
```

In ggml's column-major notation, `ne[0]` is the innermost (row) dimension. So `ffn_gate_inp` is `n_embd × n_expert`, and the 3D expert tensors are `n_embd × n_ff × n_expert`.

### Key tensor fields on `llama_layer` (`src/llama-model.h` lines 289-340)

```cpp
struct llama_layer {
    // ff MoE
    struct ggml_tensor * ffn_gate_inp      = nullptr;  // router: [n_embd, n_expert]
    struct ggml_tensor * ffn_gate_inp_s    = nullptr;  // gemma4 router scale
    struct ggml_tensor * ffn_gate_exps     = nullptr;  // [n_embd, n_ff, n_expert]
    struct ggml_tensor * ffn_down_exps     = nullptr;  // [n_ff, n_embd, n_expert]
    struct ggml_tensor * ffn_up_exps       = nullptr;  // [n_embd, n_ff, n_expert]
    struct ggml_tensor * ffn_gate_up_exps  = nullptr;  // merged: [n_embd, n_ff*2, n_expert]
    struct ggml_tensor * ffn_gate_inp_b    = nullptr;  // optional bias
    struct ggml_tensor * ffn_exp_probs_b   = nullptr;  // expert selection bias (DeepSeek V3)
    // ... per-expert scale tensors, shared expert tensors, bias tensors ...
};
```

### Core `build_moe_ffn` flow (`src/llama-graph.cpp` lines 1451-1790)

```
// 1. Compute router logits: [n_expert, n_tokens]
logits = build_lora_mm(gate_inp, cur);          // cb: "ffn_moe_logits"
if (gate_inp_b) logits = ggml_add(...);         // cb: "ffn_moe_logits_biased"

// 2. Activation function (softmax/sigmoid)
probs = ggml_soft_max/sigmoid(logits);          // cb: "ffn_moe_probs"

// 3. Optional: expert group routing (DeepSeek V3 style)
if (n_expert_groups > 1) { /* group selection masking */ }

// 4. Select top-k experts
selected_experts = ggml_argsort_top_k(probs, n_expert_used); // cb: "ffn_moe_topk"
                                                             // shape: [n_expert_used, n_tokens]

// 5. Extract gating weights
weights = ggml_get_rows(probs, selected_experts); // cb: "ffn_moe_weights"
                                                  // shape: [1, n_expert_used, n_tokens]

// 6. Per-expert FFN computation via mul_mat_id
//    (uses selected_experts indices to pick the right 2D slices from 3D expert tensors)
if (gate_up_exps) {
    gate_up = build_lora_mm_id(gate_up_exps, cur, selected_experts); // cb: "ffn_moe_gate_up"
    // split into gate and up halves
} else {
    up   = build_lora_mm_id(up_exps,   cur, selected_experts);       // cb: "ffn_moe_up"
    cur  = build_lora_mm_id(gate_exps, cur, selected_experts);       // cb: "ffn_moe_gate"
}

// 7. Activation (SiLU/GeLU/ReLU) and down-projection
cur = ggml_swiglu_split/silu/gelu/relu(cur, up);                   // cb: "ffn_moe_swiglu", etc.
experts = build_lora_mm_id(down_exps, cur, selected_experts);      // cb: "ffn_moe_down"

// 8. Weight and aggregate
experts = ggml_mul(experts, weights);                              // cb: "ffn_moe_weighted"
moe_out = experts[0] + experts[1] + ... + experts[n_expert_used-1];// cb: "ffn_moe_out"
```

### `build_lora_mm_id` (`src/llama-graph.cpp` lines 1104-1130)

```cpp
ggml_tensor * build_lora_mm_id(ggml_tensor * w, ggml_tensor * cur, ggml_tensor * ids) const {
    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
    // ... optional LoRA adapters ...
    return res;
}
```

`ggml_mul_mat_id` is the key operation — it uses the `ids` tensor (selected expert indices, shape `[n_expert_used, n_tokens]`) to select which 2D weight matrices from the 3D stacked expert tensor `w` to multiply with each column of `cur`.

### SWLP activation tracking (`src/llama-swlp.cpp` lines 100-130, 300-340, 990-1075)

```cpp
struct llama_swlp_state {
    std::vector<std::vector<int64_t>> activation_counts;  // [layer][expert_id] — NEVER UPDATED
    std::vector<std::vector<bool>>    expert_in_cache;     // boolean flags
    // ...
};
```

Initialized at `src/llama-swlp.cpp` line 315:
```cpp
state->activation_counts[il].resize(n_experts, 0);  // all zeros
```

Never updated after initialization (the FIXME at line 1063 confirms this).

```cpp
void llama_swlp::prepare_layer_experts(int il, int token_count) {
    // FIXME (P0): expert cache currently a no-op.
    // activation_counts are never updated (always zero), so predict_experts
    // returns meaningless results.
    // TODO: implement (1) activation tracking via expert gate hook,
    // (2) per-expert tensor split/migration, (3) true LRU eviction.
    GGML_UNUSED(il); GGML_UNUSED(token_count);
    return;  // early return, everything below is dead code
    // ... (dead code that would use activation_counts)
}
```

### Call site where `build_moe_ffn` is invoked (`src/models/llama.cpp` lines 203-221)

```cpp
cur = build_moe_ffn(cur,
        model.layers[il].ffn_gate_inp,
        model.layers[il].ffn_up_exps,
        model.layers[il].ffn_gate_exps,
        model.layers[il].ffn_down_exps,
        nullptr,                                    // exp_probs_b
        n_expert, n_expert_used,
        LLM_FFN_SILU, true,                         // norm_w
        hparams.expert_weights_scale,
        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
        il,
        nullptr, nullptr,                           // probs_in, gate_up_exps
        model.layers[il].ffn_up_exps_s,
        model.layers[il].ffn_gate_exps_s,
        model.layers[il].ffn_down_exps_s);
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    llama_context                          │
│                                                          │
│  [graph annotation callback]                             │
│    → calls swlp->prepare_layer(il)                       │
│    → calls swlp->prepare_layer_experts(il, n_tokens)     │
│         (currently NO-OP — activation_counts never set)   │
│                                                          │
│  [graph execution callback: llm_graph_cb]                │
│    → called for every node with name + layer index       │
│    → "ffn_moe_logits"  — pre-softmax router scores       │
│    → "ffn_moe_topk"    — selected expert indices         │
│    → "ffn_moe_weights" — gating weights                  │
│    → "ffn_moe_out"     — final MoE output                │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  llm_graph_context::build_moe_ffn                        │
│                                                          │
│  Step 1:  gate_inp × cur  →  logits [n_expert,n_tokens] │
│           (router: "who gets to compute?")                │
│  Step 2:  softmax/sigmoid →  probs                       │
│  Step 3:  top-k argsort   →  selected_experts            │
│                             [n_expert_used,n_tokens]      │
│  Step 4:  per-expert FFN via mul_mat_id                  │
│           (selected_experts picks slices from 3D tensors) │
│  Step 5:  weight, aggregate → moe_out                    │
└──────────────────────────────────────────────────────────┘

Stacked expert tensors layout (e.g., ffn_gate_exps):
  shape: [n_embd, n_ff, n_expert]
  → n_expert separate 2D weight matrices of size [n_embd, n_ff]
  → mul_mat_id(weight_3d, input, expert_ids) picks the right
    per-expert slice based on expert_ids
```

---

## Start Here

Open **`src/llama-graph.cpp`** around line **1451** where `build_moe_ffn` is defined. This is the single function that handles all MoE graph construction across all model architectures.

Then look at **`src/llama-context.cpp`** around line **2476** to see where `prepare_layer_experts` is called during graph annotation — this is where the activation tracking hook would be wired in.

---

## Hook Point Summary

### Best point to track which experts are activated

**Tensor:** `selected_experts` (expert indices, `int32_t`, shape `[n_expert_used, n_tokens]`)  
**Callback name:** `"ffn_moe_topk"`  
**File:** `src/llama-graph.cpp`, line 1558-1559

After top-k selection, `selected_experts` contains the indices of the experts that will compute for each token. This is the most direct hook for tracking which expert IDs are activated on each forward pass.

### Second hook point (raw scores before softmax)

**Tensor:** `logits` (expert scores, `f32`, shape `[n_expert, n_tokens]`)  
**Callback name:** `"ffn_moe_logits"`  
**File:** `src/llama-graph.cpp`, lines 1482-1483

This gives the raw gating scores before softmax — useful for tracking not just *which* experts were selected but their relative scores.

### Current SWLP gap

The `activation_counts[il]` vector is initialized to zero at `src/llama-swlp.cpp:315` but **never incremented**. The FIXME comment at line 1063 explicitly states: *"implement (1) activation tracking via expert gate hook"*. This is the missing piece — we need to:
1. Add a hook in `build_moe_ffn` (or in the graph annotation callback in `llama-context.cpp`) that reads `selected_experts`/`logits`
2. Increment `activation_counts[il][expert_id]` for each activated expert
3. Then `predict_experts` and `ensure_experts_cached` will work correctly
