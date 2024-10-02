#pragma once

#include "utils.h"
#include "context.h"

namespace MyTorch{
	Tensor sigmoid_forward_manual(const Tensor& input, OpContext &cxt);

	Tensor sigmoid_backward_manual(const Tensor& grad_output, OpContext &cxt);
}