#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void assign_kernel(const scalar_t* __restrict__ x, const scalar_t* __restrict__ c,
                              int64_t* __restrict__ labels, int n, int d, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const scalar_t* xi = x + i * d;
  scalar_t best = 1e38;
  int best_j = 0;
  for (int j = 0; j < k; ++j) {
    const scalar_t* cj = c + j * d;
    scalar_t dist = 0;
    for (int p = 0; p < d; ++p) {
      scalar_t diff = xi[p] - cj[p];
      dist += diff * diff;
    }
    if (dist < best) { best = dist; best_j = j; }
  }
  labels[i] = best_j;
}

template <typename scalar_t>
__global__ void update_kernel(const scalar_t* __restrict__ x, const scalar_t* __restrict__ w,
                              const int64_t* __restrict__ labels, scalar_t* __restrict__ c,
                              scalar_t* __restrict__ denom, int n, int d, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int64_t lab = labels[i];
  const scalar_t* xi = x + i * d;
  scalar_t wi = w[i];
  atomicAdd(denom + lab, wi);
  for (int p = 0; p < d; ++p) {
    atomicAdd(c + lab * d + p, wi * xi[p]);
  }
}

torch::Tensor assign_cuda(torch::Tensor x, torch::Tensor centroids) {
  auto n = x.size(0);
  auto d = x.size(1);
  auto k = centroids.size(0);
  auto labels = torch::empty({n}, x.options().dtype(torch::kInt64));
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "assign_cuda", ([&] {
    assign_kernel<scalar_t><<<blocks, threads>>>(
      x.data_ptr<scalar_t>(), centroids.data_ptr<scalar_t>(), labels.data_ptr<int64_t>(), n, d, k);
  }));
  return labels;
}

torch::Tensor update_cuda(torch::Tensor x, torch::Tensor weights, torch::Tensor labels, int64_t k) {
  auto n = x.size(0);
  auto d = x.size(1);
  auto centroids = torch::zeros({k, d}, x.options());
  auto denom = torch::zeros({k}, x.options());

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "update_cuda", ([&] {
    update_kernel<scalar_t><<<blocks, threads>>>(
      x.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(), labels.data_ptr<int64_t>(),
      centroids.data_ptr<scalar_t>(), denom.data_ptr<scalar_t>(), n, d, (int)k);
  }));

  denom = denom.clamp_min(1e-12);
  centroids = centroids / denom.view({k, 1});
  return centroids;
}


