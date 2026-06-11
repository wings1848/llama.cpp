#pragma once

// Helper to iterate over every tensor in a llama_layer.
// This is intentionally co-located with llama-swlp so readers see the coupling.
// IMPORTANT: when a new tensor is added to struct llama_layer in llama-model.h,
// it MUST be added here as well. The two files must stay in sync.

struct ggml_tensor;
struct llama_layer;

template<typename F>
static void for_each_tensor_in_layer(const llama_layer & layer, F && fn) {
    // normalization
    fn(layer.attn_norm);       fn(layer.attn_norm_b);
    fn(layer.attn_norm_2);     fn(layer.attn_norm_2_b);
    fn(layer.attn_q_norm);     fn(layer.attn_q_norm_b);
    fn(layer.attn_k_norm);     fn(layer.attn_k_norm_b);
    fn(layer.attn_out_norm);   fn(layer.attn_out_norm_b);
    fn(layer.attn_q_a_norm);   fn(layer.attn_kv_a_norm);
    fn(layer.attn_sub_norm);   fn(layer.attn_post_norm);
    fn(layer.ffn_sub_norm);    fn(layer.attn_norm_cross);
    fn(layer.attn_norm_enc);
    fn(layer.ssm_norm);         fn(layer.ssm_dt_norm);
    fn(layer.ssm_b_norm);       fn(layer.ssm_c_norm);

    // attention weights
    fn(layer.wq);       fn(layer.wk);       fn(layer.wv);       fn(layer.wo);
    fn(layer.wqkv);     fn(layer.wq_a);     fn(layer.wq_b);
    fn(layer.wkv_a_mqa); fn(layer.wkv_b);
    fn(layer.wk_b);     fn(layer.wv_b);     fn(layer.wqkv_b);   fn(layer.wo_b);
    fn(layer.wq_cross); fn(layer.wk_cross); fn(layer.wv_cross); fn(layer.wo_cross);
    fn(layer.wq_enc);   fn(layer.wk_enc);   fn(layer.wv_enc);   fn(layer.wo_enc);
    fn(layer.wqkv_gate);

    // relative position bias
    fn(layer.attn_rel_b); fn(layer.attn_rel_b_enc); fn(layer.attn_rel_b_cross);

    // ffn normalization
    fn(layer.ffn_norm);       fn(layer.ffn_norm_b);
    fn(layer.ffn_post_norm);  fn(layer.ffn_post_norm_1); fn(layer.ffn_post_norm_2);
    fn(layer.ffn_pre_norm_2);
    fn(layer.layer_out_norm); fn(layer.layer_out_norm_b);
    fn(layer.ffn_norm_exps);  fn(layer.ffn_norm_enc);

    // ff weights
    fn(layer.ffn_gate);     fn(layer.ffn_down);     fn(layer.ffn_up);
    fn(layer.ffn_gate_enc); fn(layer.ffn_down_enc); fn(layer.ffn_up_enc);

    // ff MoE
    fn(layer.ffn_gate_inp);     fn(layer.ffn_gate_inp_s);
    fn(layer.ffn_gate_exps);    fn(layer.ffn_down_exps);    fn(layer.ffn_up_exps);
    fn(layer.ffn_gate_up_exps);
    fn(layer.ffn_gate_inp_b);
    fn(layer.ffn_gate_exps_b);  fn(layer.ffn_down_exps_b);  fn(layer.ffn_up_exps_b);
    fn(layer.ffn_gate_up_exps_b);

    // ff MoE per-expert scales
    fn(layer.ffn_gate_exps_s); fn(layer.ffn_down_exps_s); fn(layer.ffn_up_exps_s);

    // ff MoE latent
    fn(layer.ffn_latent_down); fn(layer.ffn_latent_up);

    // ff shared expert
    fn(layer.ffn_gate_inp_shexp); fn(layer.ffn_gate_shexp);
    fn(layer.ffn_down_shexp);    fn(layer.ffn_up_shexp);

    // ff adjugate experts
    fn(layer.ffn_gate_chexps); fn(layer.ffn_down_chexps); fn(layer.ffn_up_chexps);

    // ff bias
    fn(layer.ffn_gate_b); fn(layer.ffn_down_b); fn(layer.ffn_up_b);
    fn(layer.ffn_act);    fn(layer.ffn_exp_probs_b);

    // mamba
    fn(layer.ssm_in);    fn(layer.ssm_x);   fn(layer.ssm_dt);   fn(layer.ssm_out);
    fn(layer.ssm_conv1d); fn(layer.ssm_a);  fn(layer.ssm_d);
    fn(layer.ssm_conv1d_b); fn(layer.ssm_dt_b);
    fn(layer.ssm_beta_alpha); fn(layer.ssm_alpha);

    // rwkv time-mix
    fn(layer.time_mix_w1);         fn(layer.time_mix_w2);
    fn(layer.time_mix_lerp_x);     fn(layer.time_mix_lerp_w);
    fn(layer.time_mix_lerp_k);     fn(layer.time_mix_lerp_v);
    fn(layer.time_mix_lerp_r);     fn(layer.time_mix_lerp_g);
    fn(layer.time_mix_lerp_fused);
    fn(layer.time_mix_first);      fn(layer.time_mix_decay);
    fn(layer.time_mix_decay_w1);   fn(layer.time_mix_decay_w2);
    fn(layer.time_mix_key);        fn(layer.time_mix_key_b);
    fn(layer.time_mix_value);      fn(layer.time_mix_value_b);
    fn(layer.time_mix_receptance); fn(layer.time_mix_receptance_b);
    fn(layer.time_mix_gate);
    fn(layer.time_mix_w0); fn(layer.time_mix_a0); fn(layer.time_mix_a1); fn(layer.time_mix_a2);
    fn(layer.time_mix_v0); fn(layer.time_mix_v1); fn(layer.time_mix_v2);
    fn(layer.time_mix_g1); fn(layer.time_mix_g2);
    fn(layer.time_mix_k_k); fn(layer.time_mix_k_a); fn(layer.time_mix_r_k);
    fn(layer.time_mix_ln); fn(layer.time_mix_ln_b); fn(layer.time_mix_output);
    fn(layer.channel_mix_lerp_k); fn(layer.channel_mix_lerp_r);
    fn(layer.channel_mix_key); fn(layer.channel_mix_receptance); fn(layer.channel_mix_value);

    // rope
    fn(layer.rope_long); fn(layer.rope_short); fn(layer.rope_freqs);

    // bitnet scales
    fn(layer.wq_s); fn(layer.wk_s); fn(layer.wv_s); fn(layer.wo_s);
    fn(layer.wqkv_s); fn(layer.wqkv_gate_s);
    fn(layer.ffn_gate_s); fn(layer.ffn_up_s); fn(layer.ffn_down_s);
    fn(layer.ffn_gate_shexp_s); fn(layer.ffn_up_shexp_s); fn(layer.ffn_down_shexp_s);
    fn(layer.ssm_in_s); fn(layer.ssm_out_s); fn(layer.ssm_alpha_s); fn(layer.ssm_beta_s);

    // input scales
    fn(layer.wq_in_s); fn(layer.wk_in_s); fn(layer.wv_in_s); fn(layer.wo_in_s);
    fn(layer.wqkv_in_s); fn(layer.wqkv_gate_in_s);
    fn(layer.ffn_gate_in_s); fn(layer.ffn_up_in_s); fn(layer.ffn_down_in_s);
    fn(layer.ffn_gate_exps_in_s); fn(layer.ffn_down_exps_in_s); fn(layer.ffn_up_exps_in_s);
    fn(layer.ffn_gate_shexp_in_s); fn(layer.ffn_up_shexp_in_s); fn(layer.ffn_down_shexp_in_s);
    fn(layer.ssm_in_in_s); fn(layer.ssm_out_in_s); fn(layer.ssm_alpha_in_s); fn(layer.ssm_beta_in_s);

    // altup & laurel
    fn(layer.per_layer_inp_gate); fn(layer.per_layer_proj); fn(layer.per_layer_post_norm);
    fn(layer.altup_correct_coef); fn(layer.altup_correct_scale);
    fn(layer.altup_predict_coef); fn(layer.altup_router); fn(layer.altup_router_norm);
    fn(layer.laurel_l); fn(layer.laurel_r); fn(layer.laurel_post_norm);

    // openai-moe / cogvlm / apertus
    fn(layer.attn_sinks);
    fn(layer.visexp_attn_wqkv); fn(layer.visexp_attn_wo);
    fn(layer.visexp_ffn_gate);  fn(layer.visexp_ffn_down);  fn(layer.visexp_ffn_up);
    fn(layer.ffn_act_alpha_n); fn(layer.ffn_act_alpha_p);
    fn(layer.ffn_act_beta);    fn(layer.ffn_act_eps);

    // Kimi KDA
    fn(layer.ssm_q_conv); fn(layer.ssm_k_conv); fn(layer.ssm_v_conv);
    fn(layer.ssm_f_a);    fn(layer.ssm_f_b);    fn(layer.ssm_beta);
    fn(layer.ssm_g_a);    fn(layer.ssm_g_b);    fn(layer.ssm_o_norm);

    // DSA
    fn(layer.indexer_k_norm); fn(layer.indexer_k_norm_b);
    fn(layer.indexer_proj);   fn(layer.indexer_attn_k);   fn(layer.indexer_attn_q_b);

    // misc
    fn(layer.out_scale);

    // posnet sub-struct
    fn(layer.posnet.norm1); fn(layer.posnet.norm1_b);
    fn(layer.posnet.conv1); fn(layer.posnet.conv1_b);
    fn(layer.posnet.norm2); fn(layer.posnet.norm2_b);
    fn(layer.posnet.conv2); fn(layer.posnet.conv2_b);
    fn(layer.posnet.attn_norm); fn(layer.posnet.attn_norm_b);
    fn(layer.posnet.attn_q); fn(layer.posnet.attn_q_b);
    fn(layer.posnet.attn_k); fn(layer.posnet.attn_k_b);
    fn(layer.posnet.attn_v); fn(layer.posnet.attn_v_b);
    fn(layer.posnet.attn_o); fn(layer.posnet.attn_o_b);
    fn(layer.posnet.norm); fn(layer.posnet.norm_b);

    // convnext sub-struct
    fn(layer.convnext.dw); fn(layer.convnext.dw_b);
    fn(layer.convnext.norm); fn(layer.convnext.norm_b);
    fn(layer.convnext.pw1); fn(layer.convnext.pw1_b);
    fn(layer.convnext.pw2); fn(layer.convnext.pw2_b);
    fn(layer.convnext.gamma);

    // shortconv sub-struct
    fn(layer.shortconv.in_proj); fn(layer.shortconv.conv); fn(layer.shortconv.out_proj);

    // nextn sub-struct
    fn(layer.nextn.eh_proj); fn(layer.nextn.eh_proj_s); fn(layer.nextn.eh_proj_in_s);
    fn(layer.nextn.embed_tokens); fn(layer.nextn.enorm); fn(layer.nextn.hnorm);
    fn(layer.nextn.shared_head_head); fn(layer.nextn.shared_head_head_s);
    fn(layer.nextn.shared_head_head_in_s); fn(layer.nextn.shared_head_norm);
}

// Non-layer tensors shared across all layers
struct llama_model;
template<typename F>
static void for_each_tensor_in_model_non_layer(const llama_model & model, F && fn) {
    fn(model.tok_embd);   fn(model.type_embd);   fn(model.pos_embd);
    fn(model.tok_norm);   fn(model.tok_norm_b);
    fn(model.output_norm); fn(model.output_norm_b); fn(model.output);
    fn(model.output_b);    fn(model.output_norm_enc);
    fn(model.output_s);    fn(model.output_in_s);
    fn(model.cls); fn(model.cls_b); fn(model.cls_out);
    fn(model.cls_out_b); fn(model.cls_norm);
    fn(model.conv1d); fn(model.conv1d_b);
    fn(model.altup_proj); fn(model.altup_unembd_proj);
    fn(model.per_layer_tok_embd); fn(model.per_layer_model_proj);
    fn(model.per_layer_proj_norm);
    fn(model.dense_2_out_layers); fn(model.dense_2_out_layers_b);
    fn(model.dense_3_out_layers);
    fn(model.nextn_proj_pre); fn(model.nextn_proj_post);
}
