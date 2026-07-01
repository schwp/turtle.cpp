#include "quantize.hpp"
#include <cstdint>
#include <cstdio>
#include <immintrin.h>
#include <omp.h>

const TypeInfo &get_type_info(GGMLType type) {
  static const TypeInfo table[] = {
      {"F32", 1, 4},    {"F16", 1, 2},   {"Q4_0", 32, 18},   {"Q4_1", 32, 20},
      {nullptr, 0, 0},  {nullptr, 0, 0}, {nullptr, 0, 0},    {nullptr, 0, 0},
      {"Q8_0", 32, 34}, {nullptr, 0, 0}, {nullptr, 0, 0},    {nullptr, 0, 0},
      {nullptr, 0, 0},  {nullptr, 0, 0}, {"Q6_K", 256, 210},
  };
  static const TypeInfo unknown = {"unknown", 0, 0};
  static constexpr int table_size = sizeof(table) / sizeof(table[0]);

  if (type >= table_size || table[type].name == nullptr)
    return unknown;
  return table[type];
}

size_t tensor_bytes(GGMLType type, size_t n_elements) {
  const TypeInfo &info = get_type_info(type);
  if (info.block_size == 0)
    return 0;

  size_t n_blocks = (n_elements + info.block_size - 1) / info.block_size;
  return n_blocks * info.type_size;
}

size_t row_bytes(GGMLType type, int n) { return tensor_bytes(type, n); }

static void dequantize_q4_0(const BlockQ4_0 &block, float *out) {
  float s = fp16_to_fp32(block.scale);

  for (int i = 0; i < 16; i++) {
    uint8_t byte = block.data[i];

    int low = (byte & 0x0F) - 8;
    int high = (byte >> 4) - 8;

    out[i] = low * s;
    out[i + 16] = high * s;
  }
}

static void dequantize_q4_1(const BlockQ4_1 &block, float *out) {
  float s = fp16_to_fp32(block.scale);
  float m = fp16_to_fp32(block.min);

  for (int i = 0; i < 16; i++) {
    uint8_t byte = block.data[i];

    int low = byte & 0x0F;
    int high = byte >> 4;

    out[i] = low * s + m;
    out[i + 16] = high * s + m;
  }
}

static void dequantize_q6_k(const BlockQ6_K &block, float *out) {
  float super_scale = fp16_to_fp32(block.d);

  for (int half = 0; half < 256; half += 128) {
    for (int l = 0; l < 32; l++) {
      int is = half / 16 + l / 16;

      int8_t q1 =
          static_cast<int8_t>((block.ql[half / 2 + l + 0] & 0xF) |
                              (((block.qh[half / 4 + l] >> 0) & 3) << 4)) -
          32;
      int8_t q2 =
          static_cast<int8_t>((block.ql[half / 2 + l + 32] & 0xF) |
                              (((block.qh[half / 4 + l] >> 2) & 3) << 4)) -
          32;
      int8_t q3 =
          static_cast<int8_t>((block.ql[half / 2 + l + 0] >> 4) |
                              (((block.qh[half / 4 + l] >> 4) & 3) << 4)) -
          32;
      int8_t q4 =
          static_cast<int8_t>((block.ql[half / 2 + l + 32] >> 4) |
                              (((block.qh[half / 4 + l] >> 6) & 3) << 4)) -
          32;

      out[half + l + 0] = super_scale * block.scales[is + 0] * q1;
      out[half + l + 32] = super_scale * block.scales[is + 2] * q2;
      out[half + l + 64] = super_scale * block.scales[is + 4] * q3;
      out[half + l + 96] = super_scale * block.scales[is + 6] * q4;
    }
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

      sum += x[i * 32 + j] * low * s;
      sum += x[i * 32 + j + 16] * high * s;
    }
  }

  return sum;
}

static float vec_dot_q4_1_impl(const float *x, const void *w, int n) {
  const BlockQ4_1 *w_dot = static_cast<const BlockQ4_1 *>(w);
  int nb_blocks = n / 32;

#ifdef __AVX2__
  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < nb_blocks; i++) {
    float scale = fp16_to_fp32(w_dot[i].scale);
    __m256 vscale = _mm256_set1_ps(scale);

    for (int j = 0; j < 16; j += 8) {
      float dq[8];
      for (int k = 0; k < 8; k++) {
        int lo = (w_dot[i].data[j + k] & 0x0F) - 8;
        dq[k] = lo * scale;
      }
      __m256 vw = _mm256_loadu_ps(dq);
      __m256 vx = _mm256_loadu_ps(x + i * 32 + j);
      acc = _mm256_fmadd_ps(vw, vx, acc);
    }

    for (int j = 0; j < 16; j += 8) {
      float dq[8];
      for (int k = 0; k < 8; k++) {
        int hi = (w_dot[i].data[j + k] >> 4) - 8;
        dq[k] = hi * scale;
      }
      __m256 vw = _mm256_loadu_ps(dq);
      __m256 vx = _mm256_loadu_ps(x + i * 32 + j + 16);
      acc = _mm256_fmadd_ps(vw, vx, acc);
    }
  }

  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 sum128 = _mm_add_ps(lo, hi);

  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);

  return _mm_cvtss_f32(sum128);
#else
  float sum = 0.0f;

  for (int i = 0; i < nb_blocks; i++) {
    float s = fp16_to_fp32(w_dot[i].scale);
    float m = fp16_to_fp32(w_dot[i].min);

    for (int j = 0; j < 16; j++) {
      uint8_t byte = w_dot[i].data[j];

      int low = (byte & 0x0F) - 8;
      int high = (byte >> 4) - 8;

      sum += x[i * 32 + j] * (low * s + m);
      sum += x[i * 32 + j + 16] * (high * s + m);
    }
  }

  return sum;
#endif
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

static float vec_dot_q6_k_impl(const float *x, const void *w, int n) {
  const BlockQ6_K *blocks = (const BlockQ6_K *)w;
  int n_blocks = n / 256;
  float sum = 0.0f;

  for (int b = 0; b < n_blocks; b++) {
    float block_out[256];

    dequantize_q6_k(blocks[b], block_out);

    int base = b * 256;
    for (int i = 0; i < 256; i++) {
      sum += x[base + i] * block_out[i];
    }
  }

  return sum;
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
    return vec_dot_q6_k_impl(x, w, n);
  default:
    fprintf(stderr, "vec_dot: unsupported type %u", type);
    return 0.0f;
  }
}

void matmul(float *out, const float *x, const void *w, GGMLType type,
            int in_features, int out_features) {
  size_t rows = row_bytes(type, in_features);

#pragma omp parallel for
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
    for (uint32_t i = 0; i < info.n_dimensions; i++) {
      n_elements *= info.dimensions[i];
      t.dimensions[i] = info.dimensions[i];
    }
    t.n_elements = n_elements;

    t.data = gguf.mapped_ptr + gguf.tensor_offset + info.offset;

    tensors[info.name] = t;
  }

  return tensors;
}

void dequantize_row(const void *data, float *out, GGMLType type, int n) {
  switch (type) {
  case GGML_TYPE_F32:
    memcpy(out, data, n * sizeof(float));
    break;

  case GGML_TYPE_F16: {
    const uint16_t *src = (const uint16_t *)data;
    for (int i = 0; i < n; i++)
      out[i] = fp16_to_fp32(src[i]);
    break;
  }

  case GGML_TYPE_Q4_0: {
    const BlockQ4_0 *blocks = (const BlockQ4_0 *)data;
    int n_blocks = n / 32;
    for (int b = 0; b < n_blocks; b++)
      dequantize_q4_0(blocks[b], out + b * 32);
    break;
  }

  case GGML_TYPE_Q4_1: {
    const BlockQ4_1 *blocks = (const BlockQ4_1 *)data;
    int n_blocks = n / 32;
    for (int b = 0; b < n_blocks; b++)
      dequantize_q4_1(blocks[b], out + b * 32);
    break;
  }

  case GGML_TYPE_Q8_0: {
    const BlockQ8_0 *blocks = (const BlockQ8_0 *)data;
    int n_blocks = n / 32;
    for (int b = 0; b < n_blocks; b++)
      dequantize_q8_0(blocks[b], out + b * 32);
    break;
  }

  case GGML_TYPE_Q6_K: {
    const BlockQ6_K *blocks = (const BlockQ6_K *)data;
    int n_blocks = n / 256;
    for (int b = 0; b < n_blocks; b++)
      dequantize_q6_k(blocks[b], out + b * 256);
    break;
  }

  default:
    fprintf(stderr, "dequantize_row: unsupported type %u\n", type);
    break;
  }
}
