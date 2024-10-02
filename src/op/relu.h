#pragma once

#include "utils.h"
#include "context.h"

namespace MyTorch{
	Tensor relu_forward_manual(const Tensor& input, OpContext &cxt);

	Tensor relu_backward_manual(const Tensor& grad_output, OpContext &cxt);
}