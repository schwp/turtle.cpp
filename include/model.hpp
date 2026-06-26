#pragma once
#include "gguf.hpp"
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
};

Model load_model(const std::string &path);
