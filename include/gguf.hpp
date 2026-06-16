#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

enum GGUFValueType : uint32_t {
  GGUF_TYPE_UINT8 = 0,
  GGUF_TYPE_INT8 = 1,
  GGUF_TYPE_UINT16 = 2,
  GGUF_TYPE_INT16 = 3,
  GGUF_TYPE_UINT32 = 4,
  GGUF_TYPE_INT32 = 5,
  GGUF_TYPE_FLOAT32 = 6,
  GGUF_TYPE_BOOL = 7,
  GGUF_TYPE_STRING = 8,
  GGUF_TYPE_ARRAY = 9,
  GGUF_TYPE_UINT64 = 10,
  GGUF_TYPE_INT64 = 11,
  GGUF_TYPE_FLOAT64 = 12,
};

enum GGMLType : uint32_t {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_1 = 3,
  GGML_TYPE_Q8_0 = 8,
};

struct GGUFFile {
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;

  std::unordered_map<std::string, std::string> metadata_str;
  std::unordered_map<std::string, uint32_t> metadata_u32;
  std::unordered_map<std::string, uint64_t> metadata_u64;
  std::unordered_map<std::string, float> metadata_f32;
  std::unordered_map<std::string, std::vector<std::string>> metadata_str_arr;
};

GGUFFile parse_gguf_config(const std::string &path);
