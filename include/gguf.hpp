#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <sys/mman.h>
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
  GGML_TYPE_Q6_K = 14,
};

struct GGUFTensorInfo {
  std::string name;
  uint32_t n_dimensions;
  uint64_t dimensions[4];
  GGMLType type;
  uint64_t offset;
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

  std::vector<GGUFTensorInfo> tensors;

  size_t tensor_offset = 0;
  const uint8_t *mapped_ptr = nullptr;
  size_t file_size = 0;

  GGUFFile() = default;

  ~GGUFFile() {
    if (mapped_ptr) {
      munmap((void *)mapped_ptr, file_size);
      mapped_ptr = nullptr;
    }
  }

  GGUFFile(GGUFFile &&other) noexcept
      : version(other.version), tensor_count(other.tensor_count),
        metadata_kv_count(other.metadata_kv_count),
        metadata_str(std::move(other.metadata_str)),
        metadata_u32(std::move(other.metadata_u32)),
        metadata_u64(std::move(other.metadata_u64)),
        metadata_f32(std::move(other.metadata_f32)),
        metadata_str_arr(std::move(other.metadata_str_arr)),
        tensors(std::move(other.tensors)), tensor_offset(other.tensor_offset),
        mapped_ptr(other.mapped_ptr), file_size(other.file_size) {
    other.mapped_ptr = nullptr;
    other.file_size = 0;
  }

  GGUFFile &operator=(GGUFFile &&other) noexcept {
    if (this != &other) {
      if (mapped_ptr)
        munmap((void *)mapped_ptr, file_size);

      version = other.version;
      tensor_count = other.tensor_count;
      metadata_kv_count = other.metadata_kv_count;
      metadata_str = std::move(other.metadata_str);
      metadata_u32 = std::move(other.metadata_u32);
      metadata_u64 = std::move(other.metadata_u64);
      metadata_f32 = std::move(other.metadata_f32);
      metadata_str_arr = std::move(other.metadata_str_arr);
      tensors = std::move(other.tensors);
      tensor_offset = other.tensor_offset;
      mapped_ptr = other.mapped_ptr;
      file_size = other.file_size;

      other.mapped_ptr = nullptr;
      other.file_size = 0;
    }
    return *this;
  }

  GGUFFile(const GGUFFile &) = delete;
  GGUFFile &operator=(const GGUFFile &) = delete;
};

GGUFFile parse_gguf_config(const std::string &path);
