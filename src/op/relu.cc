#include "relu.h"
#include "src/backend/cpu/relu.h"
#include "src/backend/cuda/relu.h"

namespace MyTorch{
	Tensor relu_forward_manual(const std::vector<Tensor> &inputs, OpContext &cxt, void* args) {
		Tensor res = DISPATCH_TO_BACKEND(inputs[0].device.device_type, relu_forward(inputs[0]));
		cxt.push_back(inputs[0]);
		return res;
	}

	Tensor relu_backward_manual(const Tensor& grad_output, OpContext &cxt) {
		Tensor input = cxt.pop_back();
		Tensor res = DISPATCH_TO_BACKEND(grad_output.device.device_type, relu_backward(grad_output, input));
		return res;
	}

	Tensor relu_forward(const Tensor &input) {
		OpContext cxt;
		return relu_forward_manual({input}, cxt, NULL);
	}
}