#pragma once

#include <ATen/ATen.h>
#include "THC/THC.h"
#include "miopen-wrapper.h"
#include "ATen/miopen/Handle.h"

namespace at { namespace native {

inline void setMIOpenStreamToCurrent() {
  // NB: Due to in-place HIPify, getCurrentCUDAStream actually means
  // getCurrentHIPStream
  MIOPEN_CHECK(miopenSetStream(getMiopenHandle(), at::cuda::getCurrentCUDAStream()));
}

// This function makes tensors which have zero stride contiguous, by
// setting the strides to 1.
inline Tensor contiguousIfZeroInStrides(const Tensor& t) {
  for (auto s : t.strides()) {
    if (s == 0) return t.contiguous();
  }
  return t;
}

}}
