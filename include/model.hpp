#pragma once
#include "cli.hpp"
#include "gguf.hpp"
#include "ops.hpp"
#include "quantize.hpp"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// This layer is based on the TiyLlama model layers
struct TransformerLayer {
  Tensor attn_norm, ffn_down, ffn_gate, ffn_up, ffn_norm, attn_k, attn_output,
      attn_q, attn_v;
};

struct ModelConfig {
  int hidden_size;
  int num_layers;
  int num_heads;
  int num_kv_heads;
  int intermediate_size;
  int context_length;
  float rope_theta;
  float rms_norm_eps;
  int head_dim;
  int vocab_size;
  bool tie_embeddings;
};

struct Model {
  ModelConfig config;
  GGUFFile gguf;
  Tensor output, token_embd, output_norm;
  std::vector<TransformerLayer> layers;

  struct {
    std::vector<float> x, normed, q, k, v, attn_out, attn_heads, scores, gate,
        up, mlp_out, logits;
  } buf;

  void allocate_buffer();
};

struct KVCache {
  std::vector<float> data;
  int num_layers;
  int max_seq_len;
  int kv_heads;
  int head_dim;

  int layer_size;
  int kv_dim;

  void allocate(const ModelConfig &config);

  float *k_at(int layer, int pos) {
    return &data[layer * layer_size + pos * kv_dim];
  }
  float *v_at(int layer, int pos) {
    return &data[layer * layer_size + pos * kv_dim + kv_dim * max_seq_len];
  }
};

Model load_model(const std::string &path);
void embed_token(float *out, const Tensor &embedding, int token_id,
                 int hidden_size);
void attention(float *out, const float *x, const TransformerLayer &layer,
               Model &model, KVCache &cache, int layer_idx, int pos);
void mlp(float *out, const float *x, const TransformerLayer &layer,
         Model &model);
void transformer(float *x, const TransformerLayer &layer, Model &model,
                 KVCache &cache, int layer_idx, int pos);
void forward(float *logits, int token_id, Model &model, KVCache &cache,
             int pos);
int argmax(const float *logits, int vocab_size);
int sample(const float *logits, int vocab_size, float temp, float top_p);
void generate(Model &model, KVCache &cache,
              const std::vector<int> &prompt_tokens, const CliArgs &args);
