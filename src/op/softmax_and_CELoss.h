#pragma once

#include "utils.h"
#include "context.h"

namespace MyTorch{
	Tensor softmax_and_CELoss_forward_manual(const Tensor& input, const Tensor &truth, OpContext &cxt);

	Tensor softmax_and_CELoss_backward_manual(const Tensor& grad_output, OpContext &cxt);
}