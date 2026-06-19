#include "quantize.hpp"
#include <cstdint>
#include <cstdio>

const TypeInfo &get_type_info(GGMLType type) {
  static const TypeInfo table[] = {
      {"F32", 1, 4},    {"F16", 1, 2},   {"Q4_0", 32, 18},
      {"Q4_1", 32, 20}, {nullptr, 0, 0}, {nullptr, 0, 0},
      {nullptr, 0, 0},  {nullptr, 0, 0}, {"Q8_0", 32, 34},
  };
  static const TypeInfo unknown = {"unknown", 0, 0};
  static constexpr int table_size = sizeof(table) / sizeof(table[0]);

  if (type >= table_size || table[type].name == nullptr)
    return unknown;
  return table[type];
}

static size_t tensor_bytes(GGMLType type, size_t n_elements) {
  const TypeInfo &info = get_type_info(type);
  if (info.block_size == 0)
    return 0;

  size_t n_blocks = (n_elements + info.block_size - 1) / info.block_size;
  return n_blocks * info.type_size;
}

static size_t row_bytes(GGMLType type, int n) { return tensor_bytes(type, n); }

static void dequantize_q4_0(const BlockQ4_0 &block, float *out) {
  float s = fp16_to_fp32(block.scale);

  for (int i = 0; i < 16; i++) {
    uint8_t byte = block.data[i];

    int low = (byte & 0x0F) - 8;
    int high = (byte >> 4) - 8;

    out[2 * i] = low * s;
    out[2 * i + 1] = high * s;
  }
}

static void dequantize_q4_1(const BlockQ4_1 &block, float *out) {
  float s = fp16_to_fp32(block.scale);
  float m = fp16_to_fp32(block.min);

  for (int i = 0; i < 16; i++) {
    uint8_t byte = block.data[i];

    int low = byte & 0x0F;
    int high = byte >> 4;

    out[2 * i] = low * s + m;
    out[2 * i + 1] = high * s + m;
  }
}

static void dequantize_q8_0(const BlockQ8_0 &block, float *out) {
  float s = fp16_to_fp32(block.scale);

  for (int i = 0; i < 32; i++)
    out[i] = block.data[i] * s;
}

static float vec_dot_f32_impl(const float *x, const void *w, int n) {
  const float *w_dot = static_cast<const float *>(w);
  float sum = 0.0f;

  for (int i = 0; i < n; i++)
    sum += x[i] * w_dot[i];

  return sum;
}

static float vec_dot_f16_impl(const float *x, const void *w, int n) {
  const uint16_t *w_dot = static_cast<const uint16_t *>(w);
  float sum = 0.0f;

  for (int i = 0; i < n; i++)
    sum += x[i] * fp16_to_fp32(w_dot[i]);

  return sum;
}

static float vec_dot_q4_0_impl(const float *x, const void *w, int n) {
  const BlockQ4_0 *w_dot = static_cast<const BlockQ4_0 *>(w);
  int nb_blocks = n / 32;
  float sum = 0.0f;

  for (int i = 0; i < nb_blocks; i++) {
    float s = fp16_to_fp32(w_dot[i].scale);

    for (int j = 0; j < 16; j++) {
      uint8_t byte = w_dot[i].data[j];

      int low = (byte & 0x0F) - 8;
      int high = (byte >> 4) - 8;

      sum += x[i * 32 + 2 * j] * low * s;
      sum += x[i * 32 + 2 * j + 1] * high * s;
    }
  }

  return sum;
}

static float vec_dot_q4_1_impl(const float *x, const void *w, int n) {
  const BlockQ4_1 *w_dot = static_cast<const BlockQ4_1 *>(w);
  int nb_blocks = n / 32;
  float sum = 0.0f;

  for (int i = 0; i < nb_blocks; i++) {
    float s = fp16_to_fp32(w_dot[i].scale);
    float m = fp16_to_fp32(w_dot[i].min);

    for (int j = 0; j < 16; j++) {
      uint8_t byte = w_dot[i].data[j];

      int low = (byte & 0x0F) - 8;
      int high = (byte >> 4) - 8;

      sum += x[i * 32 + 2 * j] * (low * s + m);
      sum += x[i * 32 + 2 * j + 1] * (high * s + m);
    }
  }

  return sum;
}

static float vec_dot_q8_0_impl(const float *x, const void *w, int n) {
  const BlockQ8_0 *w_dot = static_cast<const BlockQ8_0 *>(w);
  int nb_blocks = n / 32;
  float s = 0.0f;

  for (int i = 0; i < nb_blocks; i++) {
    float scale = fp16_to_fp32(w_dot[i].scale);

    for (int j = 0; j < 32; j++) {
      s += x[i * 32 + j] * (w_dot[i].data[j] * scale);
    }
  }

  return s;
}

float vec_dot(const float *x, const void *w, GGMLType type, int n) {
  switch (type) {
  case GGML_TYPE_F32:
    return vec_dot_f32_impl(x, w, n);
  case GGML_TYPE_F16:
    return vec_dot_f16_impl(x, w, n);
  case GGML_TYPE_Q4_0:
    return vec_dot_q4_0_impl(x, w, n);
  case GGML_TYPE_Q4_1:
    return vec_dot_q4_1_impl(x, w, n);
  case GGML_TYPE_Q8_0:
    return vec_dot_q8_0_impl(x, w, n);
  case GGML_TYPE_Q6_K:
  default:
    fprintf(stderr, "vec_dot: unsupported type %u", type);
    return 0.0f;
  }
}

void matmul(float *out, const float *x, const void *w, GGMLType type,
            int in_features, int out_features) {
  size_t rows = row_bytes(type, in_features);

  for (int i = 0; i < out_features; i++) {
    const void *row = static_cast<const uint8_t *>(w) + i * rows;
    out[i] = vec_dot(x, row, type, in_features);
  }
}

std::unordered_map<std::string, Tensor> load_tensors(GGUFFile const &gguf) {
  std::unordered_map<std::string, Tensor> tensors;

  for (const GGUFTensorInfo &info : gguf.tensors) {
    Tensor t;
    t.type = info.type;
    t.n_dimensions = info.n_dimensions;

    size_t n_elements = 1;
    for (uint32_t i = 0; i < info.n_dimensions; i++)
      n_elements *= info.dimensions[i];
    t.n_elements = n_elements;

    t.data = gguf.mapped_ptr + gguf.tensor_offset + info.offset;

    tensors[info.name] = t;
  }

  return tensors;
}
