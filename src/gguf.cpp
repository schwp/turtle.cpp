#include "gguf.hpp"
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct Parser {
  const uint8_t *data;
  size_t pos, offset;

  template <typename T> T read() {
    T val;
    memcpy(&val, data + pos, sizeof(T));
    pos += sizeof(T);
    return val;
  }

  std::string read_string() {
    uint64_t len = read<uint64_t>();
    std::string s(reinterpret_cast<const char *>(data + pos), len);
    pos += len;
    return s;
  }
};

static void parse_metadata(GGUFFile &gguf, Parser &parser) {
  for (uint64_t i = 0; i < gguf.metadata_kv_count; i++) {
    std::string key = parser.read_string();
    uint32_t type = parser.read<uint32_t>();

    switch (type) {
    case GGUF_TYPE_UINT32:
      gguf.metadata_u32[key] = parser.read<uint32_t>();
      break;
    case GGUF_TYPE_INT32:
      gguf.metadata_u32[key] = parser.read<int32_t>();
      break;
    case GGUF_TYPE_FLOAT32:
      gguf.metadata_f32[key] = parser.read<float>();
      break;
    case GGUF_TYPE_UINT64:
      gguf.metadata_u64[key] = parser.read<uint64_t>();
      break;
    case GGUF_TYPE_BOOL:
      gguf.metadata_u32[key] = parser.read<uint8_t>(); // bool is 1 byte
      break;
    case GGUF_TYPE_STRING:
      gguf.metadata_str[key] = parser.read_string();
      break;
    case GGUF_TYPE_ARRAY: {
      uint32_t arr_type = parser.read<uint32_t>();
      uint64_t arr_len = parser.read<uint64_t>();

      if (arr_type == GGUF_TYPE_STRING) {
        std::vector<std::string> arr;
        arr.reserve(arr_len);
        for (uint64_t j = 0; j < arr_len; j++)
          arr.push_back(parser.read_string());
        gguf.metadata_str_arr[key] = std::move(arr);
      } else {
        size_t elem_size = 0;
        switch (arr_type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
          elem_size = 1;
          break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
          elem_size = 2;
          break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
          elem_size = 4;
          break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
          elem_size = 8;
          break;
        }
        parser.pos += arr_len * elem_size;
      }
      break;
    }

    default:
      printf("  WARNING: unknown type %u for key '%s'\n", type, key.c_str());
      break;
    }
  }
}

static void parse_tensor(GGUFFile &gguf, Parser &parser) {
  // TODO: Parse tensor and add them to the GGUFFile object
}

GGUFFile parse_gguf_config(const std::string &path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd == -1)
    throw std::runtime_error("Error opening the file " + path);

  struct stat st;
  fstat(fd, &st);
  const uint8_t *content =
      (const uint8_t *)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);

  Parser parser{content, 0, 0};
  GGUFFile gguf;

  uint32_t magic = parser.read<uint32_t>();
  if (magic != 0x46554747)
    throw std::runtime_error("Not a GGUF file");

  gguf.version = parser.read<uint32_t>();
  gguf.tensor_count = parser.read<uint64_t>();
  gguf.metadata_kv_count = parser.read<uint64_t>();

  parse_metadata(gguf, parser);
  parse_tensors(gguf, parser);

  return gguf;
}
