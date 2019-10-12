#include "ATen/Context.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/cuda/PinnedMemoryAllocator.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"

#include "ATen/native/LinearAlgebraUtils.h"
#include "ATen/native/cuda/MiscUtils.h"

#include "THC.h" // for USE_MAGMA

#ifdef USE_MAGMA
#include <magma.h>
#include <magma_types.h>
#endif

namespace at {
namespace native {

#ifdef USE_MAGMA
template<class scalar_t>
void magmaGesvBatched(
    magma_int_t n, magma_int_t nrhs, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, scalar_t** dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array, magma_int_t batch_count, const MAGMAQueue& magma_queue) {
  AT_ERROR("gesv only takes float or double Tensors");
}

template<class scalar_t>
void magmaGetrfBatched(
    magma_int_t m, magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
  AT_ERROR("getrf only takes float or double Tensors");
}

template<class scalar_t>
void magmaGetriBatched(
    magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, scalar_t** dinvA_array, magma_int_t lddia,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  AT_ERROR("getri only takes float or double Tensors");
}

template<class scalar_t>
void magmaPotrsBatched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, scalar_t** dA_array, magma_int_t ldda,
    scalar_t** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  AT_ERROR("potrs only takes float or double Tensors");
}

template<class scalar_t>
void magmaCholeskyBatched(
    magma_uplo_t uplo, magma_int_t n, scalar_t** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
  AT_ERROR("cholesky only takes float or double Tensors");
}

template<>
void magmaGesvBatched<double>(
    magma_int_t n, magma_int_t nrhs, double** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, double** dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array, magma_int_t batch_count, const MAGMAQueue& magma_queue) {
  magma_dgesv_batched(n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, dinfo_array, batch_count, magma_queue.get_queue());
}

template<>
void magmaGesvBatched<float>(
    magma_int_t n, magma_int_t nrhs, float** dA_array, magma_int_t ldda,
    magma_int_t** dipiv_array, float** dB_array, magma_int_t lddb,
    magma_int_t* dinfo_array, magma_int_t batch_count, const MAGMAQueue& magma_queue) {
  magma_sgesv_batched(n, nrhs, dA_array, ldda, dipiv_array, dB_array, lddb, dinfo_array, batch_count, magma_queue.get_queue());
}

template<>
void magmaGetrfBatched<double>(
    magma_int_t m, magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
    magma_dgetrf_batched(m, n, dA_array, ldda, ipiv_array, info_array, batchsize, magma_queue.get_queue());
}

template<>
void magmaGetrfBatched<float>(
    magma_int_t m, magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, magma_int_t* info_array, magma_int_t batchsize,
    const MAGMAQueue& magma_queue) {
    magma_sgetrf_batched(m, n, dA_array, ldda, ipiv_array, info_array, batchsize, magma_queue.get_queue());
}

template<>
void magmaGetriBatched<double>(
    magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, double** dinvA_array, magma_int_t lddia,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
    magma_dgetri_outofplace_batched(n, dA_array, ldda, ipiv_array, dinvA_array, lddia, info_array, batchsize, magma_queue.get_queue());
}

template<>
void magmaGetriBatched<float>(
    magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t** ipiv_array, float** dinvA_array, magma_int_t lddia,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
    magma_sgetri_outofplace_batched(n, dA_array, ldda, ipiv_array, dinvA_array, lddia, info_array, batchsize, magma_queue.get_queue());
}

template<>
void magmaPotrsBatched<double>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, double** dA_array, magma_int_t ldda,
    double** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
    info = magma_dpotrs_batched(uplo, n, nrhs, dA_array, ldda, dB_array, lddb, batchsize, magma_queue.get_queue());
}

template<>
void magmaPotrsBatched<float>(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs, float** dA_array, magma_int_t ldda,
    float** dB_array, magma_int_t lddb, magma_int_t& info, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
    info = magma_spotrs_batched(uplo, n, nrhs, dA_array, ldda, dB_array, lddb, batchsize, magma_queue.get_queue());
}

template<>
void magmaCholeskyBatched<double>(
    magma_uplo_t uplo, magma_int_t n, double** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
    magma_dpotrf_batched(uplo, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
}

template<>
void magmaCholeskyBatched<float>(
    magma_uplo_t uplo, magma_int_t n, float** dA_array, magma_int_t ldda,
    magma_int_t* info_array, magma_int_t batchsize, const MAGMAQueue& magma_queue) {
    magma_spotrf_batched(uplo, n, dA_array, ldda, info_array, batchsize, magma_queue.get_queue());
}
#endif

#define ALLOCATE_ARRAY(name, type, size, dummy_tensor) \
  auto storage_##name = pin_memory<type>(size, dummy_tensor); \
  name = static_cast<type*>(storage_##name.data());

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ gesv ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_gesv(Tensor& b, Tensor& A, std::vector<int64_t>& infos) {
#ifndef USE_MAGMA
AT_ERROR("gesv: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  auto A_data = A.data<scalar_t>();
  auto b_data = b.data<scalar_t>();
  auto A_mat_stride = matrixStride(A);
  auto b_mat_stride = matrixStride(b);

  magma_int_t batch_size = magma_int_cast(batchCount(A), "batchCount");
  magma_int_t n = magma_int_cast(A.size(-2), "A.size(-2)");
  magma_int_t nrhs = magma_int_cast(b.size(-1), "b.size(-1)");

  magma_int_t* info_array;
  magma_int_t* ipiv_data;
  magma_int_t** ipiv_array;
  scalar_t** A_array;
  scalar_t** b_array;

  ALLOCATE_ARRAY(info_array, magma_int_t, batch_size, b);
  ALLOCATE_ARRAY(ipiv_data, magma_int_t, batch_size * n, b);
  ALLOCATE_ARRAY(ipiv_array, magma_int_t*, batch_size, b);
  ALLOCATE_ARRAY(A_array, scalar_t*, batch_size, b);
  ALLOCATE_ARRAY(b_array, scalar_t*, batch_size, b);

  // Set up the created arrays
  for (int64_t i = 0; i < batch_size; i++) {
    A_array[i] = &A_data[i * A_mat_stride];
    b_array[i] = &b_data[i * b_mat_stride];
    ipiv_array[i] = &ipiv_data[i * n];
  }

  MAGMAQueue magma_queue(b.get_device());
  magmaGesvBatched<scalar_t>(
      n, nrhs, A_array, n, ipiv_array, b_array, n,
      info_array, batch_size, magma_queue);

  for (int64_t i = 0; i < batch_size; i++) {
    infos[i] = info_array[i];
  }
#endif
}

std::tuple<Tensor, Tensor> _gesv_helper_cuda(const Tensor& self, const Tensor& A) {
  std::vector<int64_t> infos(batchCount(self), 0);
  auto self_working_copy = cloneBatchedColumnMajor(self);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "gesv", [&]{
    apply_gesv<scalar_t>(self_working_copy, A_working_copy, infos);
  });
  batchCheckErrors(infos, "gesv");
  return std::tuple<Tensor, Tensor>(self_working_copy, A_working_copy);
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ inverse ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_inverse(Tensor &self, Tensor &self_inv, std::vector<int64_t>& infos) {
#ifndef USE_MAGMA
AT_ERROR("inverse: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  auto self_data = self.data<scalar_t>();
  auto self_mat_stride = matrixStride(self);
  auto self_inv_data = self_inv.data<scalar_t>();
  auto self_inv_mat_stride = matrixStride(self_inv);

  magma_int_t batch_size = magma_int_cast(batchCount(self), "batchCount");
  magma_int_t n = magma_int_cast(self.size(-2), "self.size(-2)");

  magma_int_t* info_array;
  magma_int_t* ipiv_data;
  magma_int_t** ipiv_array;
  scalar_t** self_array;
  scalar_t** self_inv_array;

  ALLOCATE_ARRAY(info_array, magma_int_t, batch_size, self);
  ALLOCATE_ARRAY(ipiv_data, magma_int_t, batch_size * n, self);
  ALLOCATE_ARRAY(ipiv_array, magma_int_t*, batch_size, self);
  ALLOCATE_ARRAY(self_array, scalar_t*, batch_size, self);
  ALLOCATE_ARRAY(self_inv_array, scalar_t*, batch_size, self_inv);

  // Set up the created arrays
  for (int64_t i = 0; i < batch_size; i++) {
    self_array[i] = &self_data[i * self_mat_stride];
    self_inv_array[i] = &self_inv_data[i * self_inv_mat_stride];
    ipiv_array[i] = &ipiv_data[i * n];
  }

  MAGMAQueue magma_queue(self.get_device());
  magmaGetrfBatched<scalar_t>(
    n, n, self_array, n, ipiv_array, info_array,
    batch_size, magma_queue);

  magmaGetriBatched<scalar_t>(
    n, self_array, n, ipiv_array, self_inv_array,
    n, info_array, batch_size, magma_queue);

  for (int64_t i = 0; i < batch_size; i++) {
    infos[i] = info_array[i];
  }
#endif
}

// Because this is out-of-place inverse, the predefined macros will
// not work
Tensor _inverse_helper_cuda(const Tensor& self) {
  std::vector<int64_t> infos(batchCount(self), 0);
  auto self_working_copy = cloneBatchedColumnMajor(self);
  auto self_inv_working_copy = cloneBatchedColumnMajor(self);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "inverse", [&]{
    apply_inverse<scalar_t>(
      self_working_copy, self_inv_working_copy, infos);
  });
  batchCheckErrors(infos, "inverse");
  return self_inv_working_copy;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ potrs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_potrs(Tensor& b, Tensor& A, bool upper, int64_t& info) {
#ifndef USE_MAGMA
AT_ERROR("potrs: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;

  auto A_data = A.data<scalar_t>();
  auto b_data = b.data<scalar_t>();
  auto A_mat_stride = matrixStride(A);
  auto b_mat_stride = matrixStride(b);

  magma_int_t batch_size = magma_int_cast(batchCount(A), "batchCount");
  magma_int_t n = magma_int_cast(A.size(-2), "A.size(-2)");
  magma_int_t nrhs = magma_int_cast(b.size(-1), "b.size(-1)");

  magma_int_t info_tmp;
  magma_int_t* ipiv_data;
  magma_int_t** ipiv_array;
  scalar_t** A_array;
  scalar_t** b_array;

  ALLOCATE_ARRAY(ipiv_data, magma_int_t, batch_size * n, b);
  ALLOCATE_ARRAY(ipiv_array, magma_int_t*, batch_size, b);
  ALLOCATE_ARRAY(A_array, scalar_t*, batch_size, b);
  ALLOCATE_ARRAY(b_array, scalar_t*, batch_size, b);

  // Set up the created arrays
  for (int64_t i = 0; i < batch_size; i++) {
    A_array[i] = &A_data[i * A_mat_stride];
    b_array[i] = &b_data[i * b_mat_stride];
    ipiv_array[i] = &ipiv_data[i * n];
  }

  MAGMAQueue magma_queue(b.get_device());
  magmaPotrsBatched<scalar_t>(
      uplo, n, nrhs, A_array, n, b_array, n,
      info_tmp, batch_size, magma_queue);

  info = info_tmp;
#endif
}

Tensor _potrs_helper_cuda(const Tensor& self, const Tensor& A, bool upper) {
  int64_t info = 0;
  auto self_working_copy = cloneBatchedColumnMajor(self);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "potrs", [&]{
    apply_potrs<scalar_t>(self_working_copy, A_working_copy, upper, info);
  });
  AT_CHECK(info == 0, "MAGMA potrs : invalid argument: ", -info);
  return self_working_copy;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_cholesky(Tensor& self, bool upper, std::vector<int64_t>& infos) {
#ifndef USE_MAGMA
AT_ERROR("cholesky: MAGMA library not found in "
    "compilation. Please rebuild with MAGMA.");
#else
  magma_uplo_t uplo = upper ? MagmaUpper : MagmaLower;

  auto self_data = self.data<scalar_t>();
  auto self_mat_stride = matrixStride(self);

  magma_int_t batch_size = magma_int_cast(batchCount(self), "batchCount");
  magma_int_t n = magma_int_cast(self.size(-2), "self.size(-2)");

  magma_int_t* info_array;
  scalar_t** self_array;

  ALLOCATE_ARRAY(info_array, magma_int_t, batch_size, self);
  ALLOCATE_ARRAY(self_array, scalar_t*, batch_size, self);

  // Set up the created arrays
  for (int64_t i = 0; i < batch_size; i++) {
    self_array[i] = &self_data[i * self_mat_stride];
  }

  MAGMAQueue magma_queue(self.get_device());
  magmaCholeskyBatched<scalar_t>(
    uplo, n, self_array, n, info_array,
    batch_size, magma_queue);

  for (int64_t i = 0; i < batch_size; i++) {
    infos[i] = info_array[i];
  }
#endif
}

Tensor _cholesky_helper_cuda(const Tensor& self, bool upper) {
  std::vector<int64_t> infos(batchCount(self), 0);
  Tensor self_working_copy;
  if (upper) {
    self_working_copy = cloneBatchedColumnMajor(self.transpose(-1, -2));
  } else {
    self_working_copy = cloneBatchedColumnMajor(self);
  }

  AT_DISPATCH_FLOATING_TYPES(self.type(), "cholesky", [&]{
    apply_cholesky<scalar_t>(self_working_copy, false, infos);
  });
  batchCheckErrors(infos, "cholesky");
  if (upper) {
    return self_working_copy.transpose(-1, -2);
  } else {
    return self_working_copy;
  }
}

template <typename scalar_t, bool upper>
__global__
void triu_tril_kernel(
    scalar_t* result, scalar_t* self, int64_t k, int64_t N,
    int64_t res_batch_stride, int64_t res_row_stride, int64_t res_col_stride,
    int64_t self_batch_stride, int64_t self_row_stride, int64_t self_col_stride, int64_t self_ncol) {
  int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_idx >= N) {
    return;
  }

  int64_t self_batch_idx = blockIdx.y;
  int64_t row = linear_idx / self_ncol;
  int64_t col = linear_idx % self_ncol;

  bool mask = upper ? (col - row >= k) : (col - row <= k);

  // Now compute the offset for the self and result tensor
  int64_t res_offset = self_batch_idx * res_batch_stride + row * res_row_stride + col * res_col_stride;
  int64_t self_offset = self_batch_idx * self_batch_stride + row * self_row_stride + col * self_col_stride;
  result[res_offset] = mask ? self[self_offset] : scalar_t(0);
}

template <bool upper>
Tensor& triu_tril_cuda_template(Tensor& result, const Tensor& self, int64_t k, const char* name) {
  int64_t n_batches = batchCount(self), mat_size = self.size(-1) * self.size(-2),
          res_batch_stride = result.dim() > 2 ? result.stride(-3) : 1,
          res_row_stride = result.stride(-2), res_col_stride = result.stride(-1),
          self_batch_stride = self.dim() > 2 ? self.stride(-3) : 1,
          self_row_stride = self.stride(-2), self_col_stride = self.stride(-1);
  dim3 dim_block = cuda::getApplyBlock();
  dim3 dim_grid((mat_size + dim_block.x - 1) / dim_block.x, n_batches);
  AT_DISPATCH_ALL_TYPES_AND_HALF(self.type(), name, [&]{
    triu_tril_kernel<scalar_t, upper>
      <<<dim_grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        result.data<scalar_t>(), self.data<scalar_t>(), k, mat_size,
        res_batch_stride, res_row_stride, res_col_stride,
        self_batch_stride, self_row_stride, self_col_stride, self.size(-1));
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return result;
}

Tensor& tril_cuda_(Tensor &self, int64_t k) {
  if (!checkTrilTriuBatchContiguous(self)) self = self.contiguous();
  return tril_cuda_out(self, self, k);
}

Tensor& tril_cuda_out(Tensor &result, const Tensor& self, int64_t k) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return result;
  }
  Tensor self_c = checkTrilTriuBatchContiguous(self) ? self : self.contiguous();
  return triu_tril_cuda_template<false>(result, self_c, k, "tril");
}

Tensor& triu_cuda_(Tensor &self, int64_t k) {
  if (!checkTrilTriuBatchContiguous(self)) self = self.contiguous();
  return triu_cuda_out(self, self, k);
}

Tensor& triu_cuda_out(Tensor &result, const Tensor& self, int64_t k) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return result;
  }
  Tensor self_c = checkTrilTriuBatchContiguous(self) ? self : self.contiguous();
  return triu_tril_cuda_template<true>(result, self_c, k, "triu");
}

}}  // namespace at::native

#undef ALLOCATE_ARRAY
