#include "model.hpp"
#include <cstdio>

// This config loading function is based on the GGUF Llama models metadata for
// now, but will be adapted to support other models in the future.
static ModelConfig
config_from_gguf(const GGUFFile &gguf,
                 const std::unordered_map<std::string, Tensor> &tensors) {
  ModelConfig c{};

  c.hidden_size = gguf.metadata_u32.at("llama.embedding_length");
  c.num_layers = gguf.metadata_u32.at("llama.block_count");
  c.num_heads = gguf.metadata_u32.at("llama.attention.head_count");
  c.num_kv_heads = gguf.metadata_u32.at("llama.attention.head_count_kv");
  c.intermediate_size = gguf.metadata_u32.at("llama.feed_forward_length");
  c.context_length = gguf.metadata_u32.at("llama.context_length");
  c.rope_theta = gguf.metadata_f32.at("llama.rope.freq_base");
  c.rms_norm_eps =
      gguf.metadata_f32.at("llama.attention.layer_norm_rms_epsilon");

  c.head_dim = c.hidden_size / c.num_heads;
  c.vocab_size = tensors.at("token_embd.weight").dimensions[1];
  c.tie_embeddings = tensors.count("output.weight") == 0;

  return c;
}

static TransformerLayer
load_layer(const std::unordered_map<std::string, Tensor> &tensors,
           int layer_idx) {
  auto get = [&](const std::string &suffix) -> Tensor {
    std::string name = "blk." + std::to_string(layer_idx) + "." + suffix;
    auto it = tensors.find(name);
    if (it == tensors.end())
      throw std::runtime_error("missing tensor: " + name);
    return it->second;
  };

  TransformerLayer layer;
  layer.attn_norm = get("attn_norm.weight");
  layer.attn_q = get("attn_q.weight");
  layer.attn_k = get("attn_k.weight");
  layer.attn_v = get("attn_v.weight");
  layer.attn_output = get("attn_output.weight");
  layer.ffn_norm = get("ffn_norm.weight");
  layer.ffn_gate = get("ffn_gate.weight");
  layer.ffn_up = get("ffn_up.weight");
  layer.ffn_down = get("ffn_down.weight");

  return layer;
}

Model load_model(const std::string &path) {
  GGUFFile gguf = parse_gguf_config(path);

  Model model;
  model.gguf = parse_gguf_config(path);
  auto tensors = load_tensors(gguf);
  model.config = config_from_gguf(gguf, tensors);

  model.token_embd = tensors.at("token_embd.weight");
  model.output_norm = tensors.at("output_norm.weight");

  if (model.config.tie_embeddings) {
    model.output = model.token_embd;
    printf("Tied embeddings: output.weight = token_embd.weight\n");
  } else {
    model.output = tensors.at("output.weight");
  }

  model.layers.resize(model.config.num_layers);
  for (int i = 0; i < model.config.num_layers; i++)
    model.layers[i] = load_layer(tensors, i);

  return model;
}
