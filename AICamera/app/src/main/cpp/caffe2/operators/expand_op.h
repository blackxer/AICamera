#ifndef CAFFE2_OPERATORS_REDUCE_OPS_H_
#define CAFFE2_OPERATORS_REDUCE_OPS_H_

#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename InputTypes, class Context>
class ExpandOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ExpandOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(0));
  }
 template <typename T>
  bool DoRunWithType() {
    const auto& X = Input(0);
    const auto& Y_shape_tensor = Input(1);
    std::vector<int64_t> shape_dims(Y_shape_tensor.numel());
    context_.template CopyToCPU<int64_t>(
        Y_shape_tensor.numel(),
        Y_shape_tensor.template data<int64_t>(),
        shape_dims.data());

    const int ndim = shape_dims.size();
    const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
    std::vector<int> Y_dims;
    Y_dims.reserve(std::max(ndim, X.dim()));
    // ndim, X.ndim() might equal to 0
    for (int i = ndim - 1, j = X.dim() - 1; i >= 0 || j >= 0; --i, --j) {
      const int shape_x = (j >= 0 ? X_dims[j] : 1);
      // In PyTorch expand treats -1 as a special value to indicate
      // preserving the size of that dimension.
      const int shape_y = ((i >= 0 && shape_dims[i] > 0) ? shape_dims[i] : 1);

      CAFFE_ENFORCE(
          shape_x == 1 || shape_y == 1 || shape_x == shape_y,
          "Dimensions format invalid.");
      Y_dims.push_back(std::max(shape_x, shape_y));
    }
    std::reverse(Y_dims.begin(), Y_dims.end());
    // TODO: remove when the function in math are changed to use vector<int64_t>
    std::vector<int64_t> Y_dims_int64;
    std::copy(Y_dims.begin(), Y_dims.end(), std::back_inserter(Y_dims_int64));
    auto* Y = Output(0, Y_dims_int64, at::dtype<T>());
    math::Broadcast<T, Context>(
        X_dims.size(),
        X_dims.data(),
        Y_dims.size(),
        Y_dims.data(),
        T(1),
        X.template data<T>(),
        Y->template mutable_data<T>(),
        &context_);
    return true;
  }

};

template <typename InputTypes, class Context>
class ExpandGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ExpandGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& dY = Input(0);
    const auto& X = Input(1);
    auto* dX = Output(0);
    const int ndim = dY.dim();
    const std::vector<int> dX_dims(X.sizes().cbegin(), X.sizes().cend());
    const std::vector<int> dY_dims(dY.sizes().cbegin(), dY.sizes().cend());
    dX->ResizeLike(X);
    std::vector<int> axes;
    const int offset = ndim - X.dim();
    for (int i = 0; i < ndim; i++) {
      if (i < offset || dX_dims[i - offset] == 1) {
        axes.push_back(i);
      }
    }
    math::ReduceSum<T, Context>(
        dY_dims.size(),
        dY_dims.data(),
        axes.size(),
        axes.data(),
        T(1),
        dY.template data<T>(),
        dX->template mutable_data<T>(),
        &context_);
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_REDUCE_OPS_H_
