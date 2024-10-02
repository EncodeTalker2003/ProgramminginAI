#pragma once

#include "src/basics/tensor.h"

namespace MyTorch::Backend::CPU {
	bool cmp_tensor(const Tensor& input1, const Tensor& input2);
}