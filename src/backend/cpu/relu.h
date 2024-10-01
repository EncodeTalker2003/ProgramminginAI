#pragma once

#include "src/basics/tensor.h"

namespace MyTorch::Backend::CPU {
	Tensor relu_forward(const Tensor& input);

	Tensor relu_backward(const Tensor& grad_output, const Tensor& input);
}