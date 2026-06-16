#include "gguf.hpp"
#include <cstdio>

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <path/to/model.gguf>\n", argv[0]);
    return 1;
  }

  GGUFFile gguf = parse_gguf_config(argv[1]);
  printf("GGUF v%u - %lu tensors, %lu kv pairs\n", gguf.version,
         gguf.tensor_count, gguf.metadata_kv_count);

  for (auto &[k, v] : gguf.metadata_u32)
    printf("  %-40s = %u\n", k.c_str(), v);
  for (auto &[k, v] : gguf.metadata_f32)
    printf("  %-40s = %.6f\n", k.c_str(), v);
  for (auto &[k, v] : gguf.metadata_str)
    printf("  %-40s = %s\n", k.c_str(), v.c_str());
}
