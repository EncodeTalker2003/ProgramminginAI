#pragma once

#include "utils.h"
#include "context.h"
#include <vector>

namespace MyTorch{
	Tensor matmul_forward_manual(const Tensor& a, const Tensor& b, OpContext &cxt);

	std::vector<Tensor> matmul_backward_manual(const Tensor& grad_output, OpContext &cxt);
}