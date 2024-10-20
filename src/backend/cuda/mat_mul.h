#pragma once

#include "src/basics/tensor.h"
#include "utils.h"

namespace MyTorch::Backend::CUDA {

	Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_a = false, bool transpose_b = false);

	Tensor matmul_batch(const Tensor& a, const Tensor& b, bool transpose_a = false, bool transpose_b = false);
}