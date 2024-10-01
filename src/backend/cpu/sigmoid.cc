#include "sigmoid.h"
#include <cmath>

namespace MyTorch::Backend::CPU {
	float sigmoid_forward_cpu(float input) {
		return 1 / (1 + std::exp(-input));
	}

	Tensor sigmoid_forward(const Tensor &input) {
		Tensor output(input.shape, input.device);
		int64_t tot = input.numel();
		float* output_ptr = (float*)output.data_ptr();
		float* input_ptr = (float*)input.data_ptr();
		for (int64_t i = 0; i < tot; i++) {
			output_ptr[i] = sigmoid_forward_cpu(input_ptr[i]);
		}
		return output;
	}

	float sigmoid_backward_cpu(float grad_output, float output) {
		return grad_output * output * (1 - output);
	}

	Tensor sigmoid_backward(const Tensor &grad_output, const Tensor &output) {
		Tensor grad_input(output.shape, output.device);
		int64_t tot = output.numel();
		float* grad_input_ptr = (float*)grad_input.data_ptr();
		float* grad_output_ptr = (float*)grad_output.data_ptr();
		float* output_ptr = (float*)output.data_ptr();
		for (int64_t i = 0; i < tot; i++) {
			grad_input_ptr[i] = sigmoid_backward_cpu(grad_output_ptr[i], output_ptr[i]);
		}
		return grad_input;
	}
}