#include "model.hpp"
#include <cstdio>

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <model.gguf>\n", argv[0]);
    return 1;
  }

  Model model = load_model(argv[1]);
  model.allocate_buffer();

  KVCache cache;
  cache.allocate(model.config);

  // Hardcoded prompt: "The capital of France is"
  std::vector<int> prompt = {1, 450, 7483, 310, 3444, 338};

  printf("Generating...\n");
  generate(model, cache, prompt, 50);

  return 0;
}
