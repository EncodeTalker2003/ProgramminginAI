#include "softmax_and_CELoss.h"
#include "reduce.cuh"
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
		const int64_t num_classes
	) {
		int64_t batch_id = blockIdx.x;
		const float* cur_input = input + batch_id * num_classes;
		float local_max_val = cur_input[threadIdx.x];
		for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
			local_max_val = max(local_max_val, cur_input[i]);
		}
		float global_max_val = block_reduce_max_broadcast(local_max_val);

		float sum = 0.0;
		for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
			float tmp = std::exp(cur_input[i] - global_max_val);
			prob[batch_id * num_classes + i] = tmp;
			sum += tmp;
		}
		float global_sum = block_reduce_sum_broadcast(sum);

		for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
			//printf("prob[%ld]: %f	global_sum: %f\n", batch_id * num_classes + i, prob[batch_id * num_classes + i],global_sum);
			prob[batch_id * num_classes + i] /= global_sum;
		}

		__syncthreads();
		if (threadIdx.x == 0) {
			//printf("batch_id: %ld	truth: %d\n", batch_id, truth[batch_id]);
			loss[batch_id] = -std::log(prob[batch_id * num_classes + truth[batch_id]]);
		}
	}
	
	std::pair<Tensor, Tensor> softmax_and_CELoss_forward(const Tensor &input, const Tensor &truth) {
		if (input.dim() != 2) {
			LOG_FATAL("softmax_and_CELoss_forward: input should be 2D");
		}
		if (truth.dim() != 1) {
			LOG_FATAL("softmax_and_CELoss_forward: truth should be 1D");
		}
		if (input.shape[0] != truth.shape[0]) {
			LOG_FATAL("softmax_and_CELoss_forward: input and truth should have the same number of data");
		}
		int64_t batch_size = input.shape[0];
		int64_t num_classes = input.shape[1];
		Tensor prob({batch_size, num_classes}, input.device);
		Tensor loss({batch_size}, input.device);

		dim3 blocks(batch_size);
		dim3 threads(std::min(num_classes, (int64_t)kCudaThreadsNum));
		softmax_and_CELoss_forward_kernel<<<blocks, threads>>>(
			(const float*)input.data_ptr(), 
			(const int32_t*)truth.data_ptr(), 
			(float*)prob.data_ptr(), 
			(float*)loss.data_ptr(), 
			batch_size,
			num_classes);
		//prob.print();
		return std::make_pair(prob, loss);
	}

	__global__ void softmax_and_CELoss_backward_kernel(
		const float* grad_output, 
		const float* prob, 
		const int32_t* truth, 
		float* grad_input, 
		const int64_t batch_size,
		const int64_t num_classes
	) {
		int64_t batch_id = blockIdx.x;
		const float* cur_prob = prob + batch_id * num_classes;
		float* cur_grad_input = grad_input + batch_id * num_classes;
		float cur_grad_output = grad_output[batch_id];
		for (int64_t i = threadIdx.x; i < num_classes; i += blockDim.x) {
			cur_grad_input[i] = cur_prob[i] * cur_grad_output;
		}
		if (threadIdx.x == 0) {
			cur_grad_input[truth[batch_id]] -= cur_grad_output;
		}
	}

	Tensor softmax_and_CELoss_backward(const Tensor &grad_output, const Tensor &prob, const Tensor truth) {
		if (grad_output.dim() != 1) {
			LOG_FATAL("softmax_and_CELoss_backward: grad_output should be 1D");
		}
		if (prob.dim() != 2) {
			LOG_FATAL("softmax_and_CELoss_backward: prob should be 2D");
		}
		if (truth.dim() != 1) {
			LOG_FATAL("softmax_and_CELoss_backward: truth should be 1D");
		}
		if (grad_output.shape[0] != prob.shape[0]) {
			LOG_FATAL("softmax_and_CELoss_backward: grad_output and prob should have the same number of data");
		}
		int64_t batch_size = prob.shape[0];
		int64_t num_classes = prob.shape[1];
		Tensor grad_input(prob.shape, prob.device);
		dim3 blocks(batch_size);
		//printf("num_classes: %ld\n", num_classes);
		dim3 threads(std::min(num_classes, (int64_t)kCudaThreadsNum));
		//prob.print();
		softmax_and_CELoss_backward_kernel<<<blocks, threads>>>(
			(const float*)grad_output.data_ptr(), 
			(const float*)prob.data_ptr(), 
			(const int32_t*)truth.data_ptr(), 
			(float*)grad_input.data_ptr(), 
			batch_size,
			num_classes
		);
		return grad_input;
	}
}