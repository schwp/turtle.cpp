#pragma once
#include <string>

struct CliArgs {
  std::string model_path;
  std::string prompt = "The capital of France is";
  int max_tokens = 50;
  float temperature = 1.0f;
  float top_p = 0.9f;
  int threads = 4;
  bool greedy = true;
};

CliArgs parse_args(int argc, char **argv);
