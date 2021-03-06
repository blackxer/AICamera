#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

using forward_fn = void (*)(Tensor&, const Tensor&);

DECLARE_DISPATCH(forward_fn, copy_kernel);

} // namespace native
} // namespace at
