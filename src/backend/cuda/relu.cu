#include "relu.h"

#include <cuda_runtime.h>

namespace MyTorch::Backend::CUDA {
	__global__ void relu_forward_kernel(float* input, float* output, int64_t n) {
		CUDA_KERNEL_LOOP(i, n) {
			output[i] = input[i] > 0.0 ? input[i] : 0.0;
		}
	}

	Tensor relu_forward(const Tensor &input) {
		Tensor output(input.shape, input.device);
		int64_t tot = input.numel();
		int block_size = kCudaThreadsNum;
		int grid_size = CudaGetBlocks(tot);
		float* input_ptr = (float*)input.data_ptr();
		float* output_ptr = (float*)output.data_ptr();
		relu_forward_kernel<<<grid_size, block_size>>>(input_ptr, output_ptr, tot);
		return output;
	}

	__global__ void relu_backward_kernel(float* grad_input, float* grad_output, float* input, int64_t n) {
		CUDA_KERNEL_LOOP(i, n) {
			grad_input[i] = input[i] > 0.0 ? grad_output[i] : 0.0;
		}
	}

	Tensor relu_backward(const Tensor &grad_output, const Tensor &input) {
		Tensor grad_input(input.shape, input.device);
		int64_t tot = input.numel();
		int block_size = kCudaThreadsNum;
		int grid_size = CudaGetBlocks(tot);
		float* grad_input_ptr = (float*)grad_input.data_ptr();
		float* grad_output_ptr = (float*)grad_output.data_ptr();
		float* input_ptr = (float*)input.data_ptr();
		relu_backward_kernel<<<grid_size, block_size>>>(grad_input_ptr, grad_output_ptr, input_ptr, tot);
		return grad_input;
	}
}