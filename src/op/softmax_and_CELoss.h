#pragma once

#include "utils.h"
#include "context.h"

namespace MyTorch{
	Tensor softmax_and_CELoss_forward_manual(const std::vector<Tensor> inputs, OpContext &cxt, void *args);

	Tensor softmax_and_CELoss_backward_manual(const Tensor& grad_output, OpContext &cxt);

	Tensor softmax_and_CELoss_forward(const Tensor& input, const Tensor &truth);
}