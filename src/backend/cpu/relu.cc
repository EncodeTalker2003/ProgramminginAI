#include "relu.h"

namespace MyTorch::Backend::CPU {
	float relu_forward_cpu(float input) {
		return input > 0 ? input : 0;
	}

	Tensor relu_forward(const Tensor &input) {
		Tensor output(input.shape, input.device);
		int64_t tot = input.numel();
		float* output_ptr = (float*)output.data_ptr();
		float* input_ptr = (float*)input.data_ptr();
		for (int64_t i = 0; i < tot; i++) {
			output_ptr[i] = relu_forward_cpu(input_ptr[i]);
		}
		return output;
	}

	float relu_backward_cpu(float grad_output, float input) {
		return input > 0 ? grad_output : 0;
	}

	Tensor relu_backward(const Tensor &grad_output, const Tensor &input) {
		Tensor grad_input(input.shape, input.device);
		int64_t tot = input.numel();
		float* grad_input_ptr = (float*)grad_input.data_ptr();
		float* grad_output_ptr = (float*)grad_output.data_ptr();
		float* input_ptr = (float*)input.data_ptr();
		for (int64_t i = 0; i < tot; i++) {
			grad_input_ptr[i] = relu_backward_cpu(grad_output_ptr[i], input_ptr[i]);
		}
		return grad_input;
	}
}