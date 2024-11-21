#pragma once

#include "utils.h"
#include "context.h"

namespace MyTorch{
	Tensor conv_forward_manual(const std::vector<Tensor> inputs, OpContext &cxt, void* args);

	std::pair<Tensor, Tensor> conv_backward_manual(const Tensor& grad_output, OpContext &cxt);

	Tensor conv_forward(const Tensor &input, const Tensor &kernel);
}