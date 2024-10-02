#include "relu.h"
#include "src/backend/cpu/relu.h"
#include "src/backend/cuda/relu.h"

namespace MyTorch{
	Tensor relu_forward_manual(const Tensor& input, OpContext &cxt) {
		Tensor res = DISPATCH_TO_BACKEND(input.device.device_type, relu_forward(input));
		cxt.push_back(input);
		return res;
	}

	Tensor relu_backward_manual(const Tensor& grad_output, OpContext &cxt) {
		Tensor input = cxt.pop_back();
		Tensor res = DISPATCH_TO_BACKEND(grad_output.device.device_type, relu_backward(grad_output, input));
		return res;
	}
}