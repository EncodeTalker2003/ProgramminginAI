#pragma once

#include "utils.h"
#include "context.h"

namespace MyTorch{
	Tensor relu_forward_manual(const std::vector<Tensor> &inputs, OpContext &cxt, void* args);

	Tensor relu_backward_manual(const Tensor& grad_output, OpContext &cxt);

	Tensor relu_forward(const Tensor &input);
}