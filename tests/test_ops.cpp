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
  EXPECT_TRUE(is_close);
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
  EXPECT_TRUE(is_close);
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

  EXPECT_TRUE(is_close);
}
