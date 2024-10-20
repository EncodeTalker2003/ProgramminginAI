#include "softmax_and_CELoss.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

namespace MyTorch::Backend::CUDA {

	__global__ void softmax_and_CELoss_forward_kernel(
		const float* input, 
		const int32_t* truth, 
		float* prob, 
		float* loss, 
		const int64_t batch_size,
		const int64_t num_classes) {
		int64_t batch_id = blockIdx.x;
		const float* cur_input = input + batch_id * num_classes;
		float max_val = cur_input[threadIdx.x];
		for (int64_t i = threadIdx.x + blockDim.x; i < num_classes; i += blockDim.x) {
			max_val = max(max_val, cur_input[i]);
		}
		
	}
	
	Tensor softmax_and_CELoss_forward(const Tensor &input, const std::vector<int> truth) {
		if (input.dim() != 2) {
			LOG_ERROR("softmax_and_CELoss_forward: input should be 2D");
		}
		if (input.shape[1] != (int64_t)truth.size()) {
			LOG_ERROR("softmax_and_CELoss_forward: input and truth should have the same number of classes");
		}
		int64_t batch_size = input.shape[0];
		int64_t num_classes = input.shape[1];
		Tensor prob({batch_size, num_classes}, input.device);
		float loss = 0.0;

		dim3 blocks(batch_size);
		dim3 threads(std::min(num_classes, (int64_t)kCudaThreadsNum));
		softmax_and_CELoss_forward_kernel<<<blocks, threads>>>(
			(const float*)input.data_ptr(), 
			(const int*)truth.data(), 
			(float*)prob.data_ptr(), 
			&loss, 
			batch_size,
			num_classes);
	}

	Tensor softmax_and_CELoss_backward(const Tensor &grad_output, const Tensor &prob, const std::vector<int> truth);
}