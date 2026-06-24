#pragma once

void rmsnorm(float *out, const float *x, const float *weight, int n, float eps);
void rope(float *q, float *k, int head_dim, int n_heads, int n_kv_heads,
          int pos, int theta);
void softmax(float *x, int n);
void silu(float *x, int n);
void mul(float *out, const float *a, const float *b, int n);
void add(float *out, const float *a, const float *b, int n);
