#include "sigmoid.h"
#include "src/backend/cpu/sigmoid.h"
#include "src/backend/cuda/sigmoid.h"

namespace MyTorch{
	Tensor sigmoid_forward_manual(const std::vector<Tensor> inputs, OpContext &cxt, void* args) {
		Tensor res = DISPATCH_TO_BACKEND(inputs[0].device.device_type, sigmoid_forward(inputs[0]));
		cxt.push_back(res);
		return res;
	}

	Tensor sigmoid_backward_manual(const Tensor& grad_output, OpContext &cxt) {
		Tensor output = cxt.pop_back();
		Tensor res = DISPATCH_TO_BACKEND(grad_output.device.device_type, sigmoid_backward(grad_output, output));
		return res;
	}

	Tensor sigmoid_forward(const Tensor& input) {
		OpContext cxt;
		return sigmoid_forward_manual({input}, cxt, NULL);
	}
}