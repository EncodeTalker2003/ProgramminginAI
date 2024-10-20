#pragma once

#include "utils.h"
#include "context.h"

namespace MyTorch{
	Tensor pooling_forward_manual(const Tensor& input, int64_t pool_size, OpContext &cxt);

	Tensor pooling_backward_manual(const Tensor& grad_output, OpContext &cxt);
}