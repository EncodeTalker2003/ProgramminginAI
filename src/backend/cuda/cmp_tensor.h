#pragma once

#include "src/basics/tensor.h"
#include "utils.h"

namespace MyTorch::Backend::CUDA {
	#define ABS_ERR_THRESHOLD 1e-3
	#define REL_ERR_THRESHOLD 1e-2
	bool cmp_tensor(const Tensor& input1, const Tensor& input2);
}