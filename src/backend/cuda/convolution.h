#pragma once

#include "src/basics/tensor.h"
#include "utils.h"

namespace MyTorch::Backend::CUDA {

	// Turn [N, C_in, h, w] to [N, h * w, C_in * kh * kw] so that it could be multiplied by the kernel vector [C_in * kh * kw]
	Tensor im2col(const Tensor &input, const int64_t kh, const int64_t kw);

	// Turn grad [N, h * w, C_in * kh * kw] to [N , C_in, h, w]
	Tensor col2im(const Tensor &input, const int64_t c_in, const int64_t h, const int64_t w, const int64_t kh, const int64_t kw);

}