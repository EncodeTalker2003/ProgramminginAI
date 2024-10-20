#pragma once

#include "utils.h"
#include "context.h"
#include <vector>

namespace MyTorch{
	/*
		Input:
			a: Tensor of shape (n, m)
			b: Tensor of shape (m, p)
		Output: c
			c: Tensor of shape (n, p) where c = a @ b
	*/
	Tensor matmul_forward_manual(const Tensor& a, const Tensor& b, OpContext &cxt);

	/*
		Input:
			dc: Tensor of shape (n, p)
		Output: (da, db)
			da: Tensor of shape (n, m) where da = dc @ b^T
			db: Tensor of shape (m, p) where db = a^T @ dc
		*/
	std::pair<Tensor, Tensor> matmul_backward_manual(const Tensor& grad_output, OpContext &cxt);
}