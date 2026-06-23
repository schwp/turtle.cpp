#pragma once

void rmsnorm(float *out, const float *x, const float *weight, int n, float eps);
void softmax(float *x, int n);
void silu(float *x, int n);
void mul(float *out, const float *a, const float *b, int n);
void add(float *out, const float *a, const float *b, int n);
