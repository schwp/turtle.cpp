#include "cli.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void print_usage(const char *prog) {
  printf("Usage: %s <model.gguf> [options]\n\n", prog);
  printf("Options:\n");
  printf("  --prompt \"text\"      Input prompt (default: \"The capital of "
         "France is\")\n");
  printf("  --max-tokens N       Maximum tokens to generate (default: 50)\n");
  printf("  --temperature F      Sampling temperature (default: 1.0)\n");
  printf("  --top-p F            Top-p sampling threshold (default: 0.9)\n");
  printf("  --greedy             Use greedy decoding (default)\n");
  printf("  --sample             Use temperature + top-p sampling\n");
  printf("  --threads N          Number of threads (default: 4)\n");
}

CliArgs parse_args(int argc, char **argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    exit(1);
  }

  CliArgs args;
  args.model_path = argv[1];

  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
      args.prompt = argv[++i];
    } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
      args.max_tokens = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      args.temperature = atof(argv[++i]);
    } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
      args.top_p = atof(argv[++i]);
    } else if (strcmp(argv[i], "--greedy") == 0) {
      args.greedy = true;
    } else if (strcmp(argv[i], "--sample") == 0) {
      args.greedy = false;
    } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      args.threads = atoi(argv[++i]);
    } else {
      printf("Unknown option: %s\n", argv[i]);
      print_usage(argv[0]);
      exit(1);
    }
  }

  return args;
}
