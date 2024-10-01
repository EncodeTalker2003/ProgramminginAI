#include "sigmoid.h"
#include <cmath>

namespace MyTorch::Backend::CUDA {
	__global__ void sigmoid_forward_kernel(float* output, float* input, int64_t n) {
		CUDA_KERNEL_LOOP(i, n) {
			output[i] = 1.0 / (1.0 + std::exp(-input[i]));
		}
	}

	Tensor sigmoid_forward(const Tensor &input) {
		Tensor output(input.shape, input.device);
		int64_t tot = input.numel();
		int block_size = kCudaThreadsNum;
		int grid_size = CudaGetBlocks(tot);
		float* output_ptr = (float*)output.data_ptr();
		float* input_ptr = (float*)input.data_ptr();
		sigmoid_forward_kernel<<<grid_size, block_size>>>(output_ptr, input_ptr, tot);
	}

	__global__ void sigmoid_backward_kernel(float* grad_input, float* grad_output, float* output, int64_t n) {
		CUDA_KERNEL_LOOP(i, n) {
			grad_input[i] = grad_output[i] * output[i] * (1.0 - output[i]);
		}
	}

	Tensor sigmoid_backward(const Tensor &grad_output, const Tensor &output) {
		Tensor grad_input(output.shape, output.device);
		int64_t tot = output.numel();
		int block_size = kCudaThreadsNum;
		int grid_size = CudaGetBlocks(tot);
		float* grad_input_ptr = (float*)grad_input.data_ptr();
		float* grad_output_ptr = (float*)grad_output.data_ptr();
		float* output_ptr = (float*)output.data_ptr();
		sigmoid_backward_kernel<<<grid_size, block_size>>>(grad_input_ptr, grad_output_ptr, output_ptr, tot);
		return grad_input;
	}
}