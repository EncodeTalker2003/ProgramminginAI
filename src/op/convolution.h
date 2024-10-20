#pragma once

#include "utils.h"
#include "context.h"

namespace MyTorch{
	Tensor conv_forward_manual(const Tensor& input, OpContext &cxt);

	std::pair<Tensor, Tensor> conv_backward_manual(const Tensor& grad_output, OpContext &cxt);
}