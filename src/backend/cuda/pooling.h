#pragma once

#include "src/basics/tensor.h"
#include "utils.h"


namespace MyTorch::Backend::CUDA{
	std::pair<Tensor, Tensor> pool_forward(const Tensor &input, int pool_size);

	Tensor pool_backward(const Tensor &grad_output, const Tensor &mask, int pool_size);
}