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

static void rope_rotation(float *row, int position, int n, int d, int theta) {
  for (int i = 0; i < n; i++) {
    float freq =
        1.0f / powf(theta, static_cast<float>(2 * i) / static_cast<float>(d));
    float angle = position * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float q0 = row[i], q1 = row[i + n];
    row[i] = q0 * cos_a - q1 * sin_a;
    row[i + n] = q0 * sin_a + q1 * cos_a;
  }
}

void rope(float *q, float *k, int head_dim, int n_heads, int n_kv_heads,
          int pos, int theta) {
  int mid = head_dim / 2;

  for (int h = 0; h < n_heads; h++) {
    float *q_values = q + h * head_dim;
    rope_rotation(q_values, pos, mid, head_dim, theta);
  }

  for (int h = 0; h < n_kv_heads; h++) {
    float *k_values = k + h * head_dim;
    rope_rotation(k_values, pos, mid, head_dim, theta);
  }
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
