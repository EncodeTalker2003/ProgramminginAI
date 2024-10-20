#pragma once

#include "src/basics/tensor.h"
#include "utils.h"
#include <vector>

namespace MyTorch::Backend::CUDA {
	// [N,C] --softmax--> [N,C] --cross_entropy--> [N] in [0,1]^N
	Tensor softmax_and_CELoss_forward(const Tensor &input, const Tensor &truth);

	Tensor softmax_and_CELoss_backward(const Tensor &grad_output, const Tensor &prob, const Tensor &truth);
}