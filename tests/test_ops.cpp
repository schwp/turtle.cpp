#include "ops.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>

TEST(OpsTest, SoftmaxMatchingPyTorch) {
  std::vector<float> arr = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f};
  int n = arr.size();

  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor tensor = torch::from_blob(arr.data(), {n}, options);

  torch::Tensor res = torch::softmax(tensor, 0);
  softmax(arr.data(), n);

  torch::Tensor my_output_tensor = torch::from_blob(arr.data(), {n}, options);

  bool is_close = torch::allclose(res, my_output_tensor);
  EXPECT_TRUE(is_close) << "Softmax vectors don't match PyTorch reference";
}

TEST(OpsTest, SiLUMatchingPyTorch) {
  std::vector<float> arr = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f};
  int n = arr.size();

  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor tensor = torch::from_blob(arr.data(), {n}, options);

  torch::Tensor res = torch::silu(tensor);
  silu(arr.data(), n);

  torch::Tensor my_output_tensor = torch::from_blob(arr.data(), {n}, options);

  bool is_close = torch::allclose(res, my_output_tensor);
  EXPECT_TRUE(is_close) << "SiLU vectors don't match PyTorch reference";
}

TEST(OpsTest, RMSNormMatchingPyTorch) {
  std::vector<float> in_data = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f};
  std::vector<float> weight_data = {0.5f, 1.0f, 1.5f, 2.0f, 1.0f, 0.5f};
  int n = in_data.size();
  float eps = 1e-5f;

  std::vector<float> out_data(n, 0.0f);

  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor th_input = torch::from_blob(in_data.data(), {n}, options);
  torch::Tensor th_weight = torch::from_blob(weight_data.data(), {n}, options);

  torch::Tensor th_variance = th_input.pow(2).mean(0, true);
  torch::Tensor th_output =
      th_input * torch::rsqrt(th_variance + eps) * th_weight;

  rmsnorm(out_data.data(), in_data.data(), weight_data.data(), n, eps);

  torch::Tensor my_output_tensor =
      torch::from_blob(out_data.data(), {n}, options);

  bool is_close = torch::allclose(th_output, my_output_tensor, 1e-4, 1e-5);

  EXPECT_TRUE(is_close) << "RMSNorm vectors don't match PyTorch reference";
}

TEST(OpsTest, RoPEMatchingPyTorch) {
  int head_dim = 64;
  int n_heads = 4;
  int n_kv_heads = 2;
  float theta = 10000.0f;
  int pos = 7;

  int q_size = n_heads * head_dim;
  int k_size = n_kv_heads * head_dim;

  std::vector<float> q(q_size);
  std::vector<float> k(k_size);
  for (int i = 0; i < q_size; i++)
    q[i] = 0.01f * (i + 1);
  for (int i = 0; i < k_size; i++)
    k[i] = 0.02f * (i + 1);

  auto options = torch::TensorOptions().dtype(torch::kFloat32);

  int half = head_dim / 2;
  torch::Tensor freq_indices = torch::arange(0, half, options);
  torch::Tensor freqs = 1.0 / torch::pow(theta, 2.0 * freq_indices / head_dim);

  torch::Tensor angles = freqs * pos;
  torch::Tensor cos_a = torch::cos(angles);
  torch::Tensor sin_a = torch::sin(angles);

  torch::Tensor q_tensor =
      torch::from_blob(q.data(), {n_heads, head_dim}, options).clone();
  torch::Tensor q_first = q_tensor.slice(1, 0, half);
  torch::Tensor q_second = q_tensor.slice(1, half, head_dim);

  torch::Tensor q_rot_first = q_first * cos_a - q_second * sin_a;
  torch::Tensor q_rot_second = q_first * sin_a + q_second * cos_a;
  torch::Tensor q_ref = torch::cat({q_rot_first, q_rot_second}, 1);

  torch::Tensor k_tensor =
      torch::from_blob(k.data(), {n_kv_heads, head_dim}, options).clone();
  torch::Tensor k_first = k_tensor.slice(1, 0, half);
  torch::Tensor k_second = k_tensor.slice(1, half, head_dim);

  torch::Tensor k_rot_first = k_first * cos_a - k_second * sin_a;
  torch::Tensor k_rot_second = k_first * sin_a + k_second * cos_a;
  torch::Tensor k_ref = torch::cat({k_rot_first, k_rot_second}, 1);

  rope(q.data(), k.data(), head_dim, n_heads, n_kv_heads, pos, theta);

  torch::Tensor q_out =
      torch::from_blob(q.data(), {n_heads, head_dim}, options);
  torch::Tensor k_out =
      torch::from_blob(k.data(), {n_kv_heads, head_dim}, options);

  bool q_close = torch::allclose(q_ref, q_out, 1e-4, 1e-5);
  bool k_close = torch::allclose(k_ref, k_out, 1e-4, 1e-5);

  EXPECT_TRUE(q_close) << "Q vectors don't match PyTorch reference";
  EXPECT_TRUE(k_close) << "K vectors don't match PyTorch reference";
}

TEST(OpsTest, RoPEPosition0IsIdentity) {
  int head_dim = 64, n_heads = 2, n_kv_heads = 1;

  std::vector<float> q(n_heads * head_dim);
  std::vector<float> k(n_kv_heads * head_dim);
  for (int i = 0; i < (int)q.size(); i++)
    q[i] = 0.1f * (i + 1);
  for (int i = 0; i < (int)k.size(); i++)
    k[i] = 0.2f * (i + 1);

  std::vector<float> q_orig = q;
  std::vector<float> k_orig = k;

  rope(q.data(), k.data(), head_dim, n_heads, n_kv_heads, 0, 10000.0f);

  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor q_before =
      torch::from_blob(q_orig.data(), {(int)q.size()}, options);
  torch::Tensor q_after = torch::from_blob(q.data(), {(int)q.size()}, options);
  torch::Tensor k_before =
      torch::from_blob(k_orig.data(), {(int)k.size()}, options);
  torch::Tensor k_after = torch::from_blob(k.data(), {(int)k.size()}, options);

  EXPECT_TRUE(torch::allclose(q_before, q_after, 1e-6))
      << "RoPE at position 0 should be identity for Q";
  EXPECT_TRUE(torch::allclose(k_before, k_after, 1e-6))
      << "RoPE at position 0 should be identity for K";
}

TEST(OpsTest, RoPEPreservesMagnitude) {
  int head_dim = 64, n_heads = 2, n_kv_heads = 1;

  std::vector<float> q(n_heads * head_dim);
  for (int i = 0; i < (int)q.size(); i++)
    q[i] = 0.1f * (i + 1);

  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor q_tensor =
      torch::from_blob(q.data(), {n_heads, head_dim}, options).clone();
  float norm_before = q_tensor.norm().item<float>();

  rope(q.data(), q.data(), head_dim, n_heads, n_kv_heads, 42, 10000.0f);

  torch::Tensor q_after =
      torch::from_blob(q.data(), {n_heads, head_dim}, options);
  float norm_after = q_after.norm().item<float>();

  EXPECT_NEAR(norm_before, norm_after, 1e-3)
      << "RoPE should preserve vector magnitude";
}
