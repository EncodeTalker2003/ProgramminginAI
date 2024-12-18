#pragma once

#include "src/basics/tensor.h"
#include "utils.h"


namespace MyTorch::Backend::CUDA {

	Tensor transpose(const Tensor &input, int axis1, int axis2);
}