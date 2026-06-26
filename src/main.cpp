#include "model.hpp"
#include <cstdio>

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <model.gguf>\n", argv[0]);
    return 1;
  }

  Model model = load_model(argv[1]);

  auto &l = model.layers[0];
  printf("Layer 0 tensor types:\n");
  printf("  attn_norm:   %s\n", get_type_info(l.attn_norm.type).name);
  printf("  attn_q:      %s\n", get_type_info(l.attn_q.type).name);
  printf("  attn_k:      %s\n", get_type_info(l.attn_k.type).name);
  printf("  ffn_gate:    %s\n", get_type_info(l.ffn_gate.type).name);
  printf("  output:      %s\n", get_type_info(model.output.type).name);

  return 0;
}
