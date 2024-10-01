#pragma once

#include "src/basics/tensor.h"
#include "utils.h"

namespace MyTorch::Backend::CUDA {
	Tensor sigmoid_forward(const Tensor& input);

	Tensor sigmoid_backward(const Tensor& grad_output, const Tensor& input);
}