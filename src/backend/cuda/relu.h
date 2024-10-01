#pragma once

#include "src/basics/tensor.h"
#include "utils.h"

namespace MyTorch::Backend::CUDA {

	Tensor relu_forward(const Tensor& input);

	Tensor relu_backward(const Tensor& grad_output, const Tensor& input);
}