#include "model.hpp"
#include <cmath>
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

void Model::allocate_buffer() {
  buf.x.resize(config.hidden_size);
  buf.normed.resize(config.hidden_size);
  buf.q.resize(config.num_heads * config.head_dim);
  buf.k.resize(config.num_kv_heads * config.head_dim);
  buf.v.resize(config.num_kv_heads * config.head_dim);
  buf.attn_out.resize(config.hidden_size);
  buf.attn_heads.resize(config.hidden_size);
  buf.scores.resize(config.context_length);
  buf.gate.resize(config.intermediate_size);
  buf.up.resize(config.intermediate_size);
  buf.mlp_out.resize(config.hidden_size);
  buf.logits.resize(config.vocab_size);
}

void KVCache::allocate(const ModelConfig &config) {
  num_layers = config.num_layers;
  max_seq_len = config.context_length;
  kv_heads = config.num_kv_heads;
  head_dim = config.head_dim;

  kv_dim = kv_heads * head_dim;
  layer_size = max_seq_len * kv_dim * 2;

  data.resize(layer_size * num_layers, 0.0f);
}

Model load_model(const std::string &path) {
  Model model;
  model.gguf = parse_gguf_config(path);

  auto tensors = load_tensors(model.gguf);

  model.config = config_from_gguf(model.gguf, tensors);

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

void embed_token(float *out, const Tensor &embedding, int token_id,
                 int hidden_size) {
  size_t offset = token_id * row_bytes(embedding.type, hidden_size);
  const void *row = static_cast<const uint8_t *>(embedding.data) + offset;
  dequantize_row(row, out, embedding.type, hidden_size);
}

void attention(float *out, const float *x, const TransformerLayer &layer,
               Model &model, KVCache &cache, int layer_idx, int pos) {
  float *Q = model.buf.q.data();
  float *K = model.buf.k.data();
  float *V = model.buf.v.data();
  float *scores = model.buf.scores.data();
  float *attn_heads = model.buf.attn_heads.data();

  matmul(Q, x, layer.attn_q.data, layer.attn_q.type, model.config.hidden_size,
         model.config.num_heads * model.config.head_dim);
  matmul(K, x, layer.attn_k.data, layer.attn_k.type, model.config.hidden_size,
         model.config.num_kv_heads * model.config.head_dim);
  matmul(V, x, layer.attn_v.data, layer.attn_v.type, model.config.hidden_size,
         model.config.num_kv_heads * model.config.head_dim);

  rope(Q, K, model.config.head_dim, model.config.num_heads,
       model.config.num_kv_heads, pos, model.config.rope_theta);

  memcpy(cache.k_at(layer_idx, pos), K,
         model.config.num_kv_heads * model.config.head_dim * sizeof(float));
  memcpy(cache.v_at(layer_idx, pos), V,
         model.config.num_kv_heads * model.config.head_dim * sizeof(float));

  float s = 1.0f / sqrtf(static_cast<float>(model.config.head_dim));
  for (int i = 0; i < model.config.num_heads; i++) {
    int kv_h = i / (model.config.num_heads / model.config.num_kv_heads);
    float *q_head = Q + i * model.config.head_dim;
    float *head_out = attn_heads + i * model.config.head_dim;

    for (int j = 0; j <= pos; j++) {
      float *k_i = cache.k_at(layer_idx, j) + kv_h * model.config.head_dim;
      float score = 0.0f;
      for (int d = 0; d < model.config.head_dim; d++)
        score += q_head[d] * k_i[d];
      scores[j] = score * s;
    }

    softmax(scores, pos + 1);

    for (int j = 0; j < model.config.head_dim; j++)
      head_out[j] = 0.0f;

    for (int j = 0; j <= pos; j++) {
      float *v_j = cache.v_at(layer_idx, j) + kv_h * model.config.head_dim;
      float w = scores[j];
      for (int k = 0; k < model.config.head_dim; k++)
        head_out[k] += w * v_j[k];
    }
  }

  matmul(out, attn_heads, layer.attn_output.data, layer.attn_output.type,
         model.config.hidden_size, model.config.hidden_size);
}

void mlp(float *out, const float *x, const TransformerLayer &layer,
         Model &model) {
  float *gate = model.buf.gate.data();
  float *up = model.buf.up.data();

  matmul(gate, x, layer.ffn_gate.data, layer.ffn_gate.type,
         model.config.hidden_size, model.config.intermediate_size);
  matmul(up, x, layer.ffn_up.data, layer.ffn_up.type, model.config.hidden_size,
         model.config.intermediate_size);

  silu(gate, model.config.intermediate_size);
  mul(gate, gate, up, model.config.intermediate_size);

  matmul(out, gate, layer.ffn_down.data, layer.ffn_down.type,
         model.config.intermediate_size, model.config.hidden_size);
}

void transformer(float *x, const TransformerLayer &layer, Model &model,
                 KVCache &cache, int layer_idx, int pos) {
  float *normed = model.buf.normed.data();
  float *attn_out = model.buf.attn_out.data();
  float *mlp_out = model.buf.mlp_out.data();

  rmsnorm(normed, x, static_cast<const float *>(layer.attn_norm.data),
          model.config.hidden_size, model.config.rms_norm_eps);

  attention(attn_out, normed, layer, model, cache, layer_idx, pos);

  add(x, x, attn_out, model.config.hidden_size);

  rmsnorm(normed, x, static_cast<const float *>(layer.ffn_norm.data),
          model.config.hidden_size, model.config.rms_norm_eps);

  mlp(mlp_out, normed, layer, model);

  add(x, x, mlp_out, model.config.hidden_size);
}

void forward(float *logits, int token_id, Model &model, KVCache &cache,
             int pos) {
  float *x = model.buf.x.data();

  embed_token(x, model.token_embd, token_id, model.config.hidden_size);

  for (int i = 0; i < model.config.num_layers; i++)
    transformer(x, model.layers[i], model, cache, i, pos);

  rmsnorm(x, x, static_cast<const float *>(model.output_norm.data),
          model.config.hidden_size, model.config.rms_norm_eps);

  matmul(logits, x, model.output.data, model.output.type,
         model.config.hidden_size, model.config.vocab_size);
}

int argmax(const float *logits, int vocab_size) {
  int max_idx = 0;
  float max_val = logits[0];

  for (int i = 1; i < vocab_size; i++) {
    if (logits[i] > max_val) {
      max_val = logits[i];
      max_idx = i;
    }
  }

  return max_idx;
}

static std::string decode_token(const Model &model, int token_id) {
  std::string word =
      model.gguf.metadata_str_arr.at("tokenizer.ggml.tokens")[token_id];

  size_t pos_sp;
  while ((pos_sp = word.find("▁")) != std::string::npos)
    word.replace(pos_sp, 3, " ");

  return word;
}

void generate(Model &model, KVCache &cache,
              const std::vector<int> &prompt_tokens, int max_tokens) {
  float *logits = model.buf.logits.data();
  int pos = 0;

  for (int token : prompt_tokens)
    forward(logits, token, model, cache, pos++);

  int next_token = argmax(logits, model.config.vocab_size);

  int eos_token = 2;

  for (int i = 0; i < max_tokens; i++) {
    printf("%s", decode_token(model, next_token).c_str());

    if (next_token == eos_token)
      break;

    forward(logits, next_token, model, cache, pos++);
    next_token = argmax(logits, model.config.vocab_size);
  }

  printf("\n");
}
