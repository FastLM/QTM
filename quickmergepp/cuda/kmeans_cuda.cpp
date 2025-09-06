#include <torch/extension.h>

torch::Tensor assign_cuda(torch::Tensor x, torch::Tensor centroids);
torch::Tensor update_cuda(torch::Tensor x, torch::Tensor weights, torch::Tensor labels, int64_t k);

torch::Tensor assign(torch::Tensor x, torch::Tensor centroids) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(centroids.is_cuda(), "centroids must be CUDA");
  return assign_cuda(x, centroids);
}

torch::Tensor update(torch::Tensor x, torch::Tensor weights, torch::Tensor labels, int64_t k) {
  TORCH_CHECK(x.is_cuda() && weights.is_cuda() && labels.is_cuda(), "inputs must be CUDA");
  return update_cuda(x, weights, labels, k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("assign", &assign, "KMeans Assign (CUDA)");
  m.def("update", &update, "KMeans Update (CUDA)");
}


