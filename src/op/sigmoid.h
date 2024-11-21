#pragma once

#include "utils.h"
#include "context.h"

namespace MyTorch{
	Tensor sigmoid_forward_manual(const std::vector<Tensor> inputs, OpContext &cxt, void* args);

	Tensor sigmoid_backward_manual(const Tensor& grad_output, OpContext &cxt);

	Tensor sigmoid_forward(const Tensor& input);
}