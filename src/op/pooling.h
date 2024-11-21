#pragma once

#include "utils.h"
#include "context.h"

namespace MyTorch{

	struct Pooling_args {
		int64_t pool_size;
	};

	Tensor pooling_forward_manual(const std::vector<Tensor> &inputs, OpContext &cxt, void* args);

	Tensor pooling_backward_manual(const Tensor& grad_output, OpContext &cxt);

	Tensor max_pool(const Tensor &input, int64_t pool_size);
}