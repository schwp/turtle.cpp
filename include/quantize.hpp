#pragma once
#include "gguf.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>

struct Tensor {
  const void *data;
  uint32_t n_dimensions;
  uint64_t dimensions[4];
  GGMLType type;
  size_t n_elements;
};

struct BlockQ4_0 {
  uint16_t scale;
  uint8_t data[16];
};

struct BlockQ4_1 {
  uint16_t scale;
  uint16_t min;
  uint8_t data[16];
};

struct BlockQ8_0 {
  uint16_t scale;
  int8_t data[32];
};

struct TypeInfo {
  const char *name;
  size_t block_size;
  size_t type_size;
};

static inline float fp16_to_fp32(uint16_t fp16) {
  uint16_t s = (fp16 & 0x8000) >> 15;
  uint16_t e = (fp16 & 0x7C00) >> 10;
  uint16_t m = fp16 & 0x03FF;

  uint32_t f32_s = static_cast<uint32_t>(s) << 31;
  uint32_t f32_e = 0;
  uint32_t f32_m = 0;

  if (e == 0) {
    if (m != 0) {
      uint32_t expo = 113;
      while ((m & 0x0400) == 0) {
        m <<= 1;
        expo--;
      }

      m &= 0x03FF;

      f32_e = expo << 23;
      f32_m = static_cast<uint32_t>(m) << 13;
    }
  } else if (e == 0x1F) {
    f32_e = 0xFF << 23;
    f32_m = static_cast<uint32_t>(m) << 13;
  } else {
    f32_e = static_cast<uint32_t>(e - 15 + 127) << 23;
    f32_m = static_cast<uint32_t>(m) << 13;
  }

  uint32_t fp32 = f32_s | f32_e | f32_m;

  float result;
  memcpy(&result, &fp32, sizeof(float));
  return result;
}

float vec_dot(const float *x, const void *w, GGMLType type, int n);
void matmul();
std::unordered_map<std::string, Tensor> load_tensors(const GGUFFile &gguf);

void dequantize_row(const void *data, float *out, GGMLType type, int n);
