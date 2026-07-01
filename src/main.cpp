#include "cli.hpp"
#include "model.hpp"
#include <cstdio>

int main(int argc, char **argv) {
  CliArgs args = parse_args(argc, argv);

  Model model = load_model(args.model_path);
  model.allocate_buffer();

  KVCache cache;
  cache.allocate(model.config);

  printf("Threads: %d\n", args.threads);
  printf("Sampling: %s\n", args.greedy ? "greedy" : "top-p");
  if (!args.greedy)
    printf("Temperature: %.2f, Top-p: %.2f\n", args.temperature, args.top_p);

  // Hardcoded prompt: "The capital of France is"
  std::vector<int> prompt = {1, 450, 7483, 310, 3444, 338};

  printf("Prompt: \"%s\"\n", args.prompt.c_str());
  generate(model, cache, prompt, args);

  return 0;
}
