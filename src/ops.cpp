#include "ops.hpp"
#include <cfloat>
#include <cmath>

void rmsnorm(float *out, const float *x, const float *weight, int n,
             float eps) {
  double sum = 0.0f;
  for (int i = 0; i < n; i++)
    sum += static_cast<double>(x[i]) * static_cast<double>(x[i]);

  float scale = 1.0f / sqrtf(static_cast<float>((sum / n)) + eps);

  for (int i = 0; i < n; i++)
    out[i] = x[i] * scale * weight[i];
}

void softmax(float *x, int n) {
  float max_x = -FLT_MAX, sum = 0.0f;
  for (int i = 0; i < n; i++)
    max_x = fmaxf(max_x, x[i]);

  for (int i = 0; i < n; i++)
    sum += expf(x[i] - max_x);

  for (int i = 0; i < n; i++)
    x[i] = expf(x[i] - max_x) / sum;
}

void silu(float *x, int n) {
  for (int i = 0; i < n; i++)
    x[i] = x[i] / (1.0f + expf(-x[i]));
}

void mul(float *out, const float *a, const float *b, int n) {
  for (int i = 0; i < n; i++)
    out[i] = a[i] * b[i];
}

void add(float *out, const float *a, const float *b, int n) {
  for (int i = 0; i < n; i++)
    out[i] = a[i] + b[i];
}
