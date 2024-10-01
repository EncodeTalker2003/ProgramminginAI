#pragma once

#include "src/basics/tensor.h"

namespace MyTorch::Backend::CPU {
	Tensor sigmoid_forward(const Tensor& input);

	Tensor sigmoid_backward(const Tensor& grad_output, const Tensor& input);
}