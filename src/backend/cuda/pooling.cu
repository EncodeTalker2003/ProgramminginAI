#include "pooling.h"
#include <cuda_runtime.h>

namespace MyTorch::Backend::CUDA {

	__global__ void pool_forward_kernel(const float* input, float* output, int8_t* mask, int64_t h, int64_t w, int pool_size) {
		int64_t batch_id = blockIdx.x;
		int64_t pool_h_id = blockIdx.y;
		for (int64_t pool_w_id = threadIdx.x; pool_w_id < w / pool_size; pool_w_id += blockDim.x) {
			const float* st = input + 
				batch_id * h * w + 
				pool_h_id * pool_size * w + 
				pool_w_id * pool_size;
			float max_val = *st;
			int x_pos = 0, y_pos = 0;
			for (int64_t i = 0; i < pool_size; i++) {
				for (int64_t j = 0; j < pool_size; j++) {
					float cur_val = st[i * w + j];
					if (cur_val > max_val) {
						max_val = cur_val;
						x_pos = i;
						y_pos = j;
					}
				}
			}
			int64_t cur_pos = 
				batch_id * (h / pool_size) * (w / pool_size) +
				pool_h_id * (w / pool_size) +
				pool_w_id;
			output[cur_pos] = max_val;
			cur_pos = 
				batch_id * h * w + 
				(pool_h_id * pool_size + x_pos) * w +
				(pool_w_id * pool_size + y_pos);
			mask[cur_pos] = 1;
		}
	}

	std::pair<Tensor, Tensor> pool_forward(const Tensor &input, int pool_size) {
		if (input.dim() < 2) {
			LOG_FATAL("pool_forward: input should be at least 2D");
		}
		int64_t h = input.shape[input.dim() - 2];
		int64_t w = input.shape[input.dim() - 1];
		if ((h % pool_size != 0) || (w % pool_size != 0)) {
			LOG_FATAL("pool_forward: input size should be divisible by pool_size");
		}
		int64_t batch_size = 1;
		std::vector<int64_t> output_shape;
		for (int i = 0; i < (int)input.dim() - 2; i++) {
			batch_size *= input.shape[i];
			output_shape.push_back(input.shape[i]);
		}
		output_shape.push_back(h / pool_size);
		output_shape.push_back(w / pool_size);
		Tensor output(output_shape, input.device);
		Tensor mask = Tensor::zeros(output_shape, input.device);
		
		dim3 blocks(batch_size, h / pool_size);
		dim3 threads(std::min(w / pool_size, (int64_t)kCudaThreadsNum));
		pool_forward_kernel<<<blocks, threads>>>((const float*)input.data_ptr(), (float*)output.data_ptr(),  (int8_t*)mask.data_ptr(), h, w, pool_size);
		return std::make_pair(output, mask);
	}

	__global__ void pool_backward_kernel(const float* grad_output, const int8_t* mask, float* grad_input, int64_t h, int64_t w, int pool_size) {
		int64_t batch_id = blockIdx.x;
		int64_t pool_h_id = blockIdx.y;
		for (int64_t pool_w_id = threadIdx.x; pool_w_id < w / pool_size; pool_w_id += blockDim.x) {
			float grad_val = grad_output[batch_id * (h / pool_size) * (w / pool_size) + pool_h_id * (w / pool_size) + pool_w_id];
			int64_t cur_pos = 
				batch_id * h * w + 
				pool_h_id * pool_size * w + 
				pool_w_id * pool_size;
			for (int64_t i = 0; i < pool_size; i++) {
				for (int64_t j = 0; j < pool_size; j++) {
					grad_input[cur_pos + i * w + j] = grad_val * mask[cur_pos + i * w + j];
				}
			}
		}
	}

	Tensor pool_backward(const Tensor &grad_output, const Tensor &mask, int pool_size) {
		if (grad_output.dim() < 2) {
			LOG_FATAL("pool_backward: grad_output should be at least 2D");
		}
		if (mask.dim() < 2) {
			LOG_FATAL("pool_backward: mask should be at least 2D");
		}
		if (grad_output.dim() != mask.dim()) {
			LOG_FATAL("pool_backward: grad_output and mask should have the same shape");
		}

		int64_t batch_size = 1;
		std::vector<int64_t> grad_input_shape;
		for (int i = 0; i < (int)grad_output.dim() - 2; i++) {
			if (grad_output.shape[i] != mask.shape[i]) {
				LOG_FATAL("pool_backward: grad_output and mask should have the same shape");
			}
			batch_size *= grad_output.shape[i];
			grad_input_shape.push_back(grad_output.shape[i]);
		}
		int64_t h = grad_output.shape[grad_output.dim() - 2] * pool_size;
		int64_t w = grad_output.shape[grad_output.dim() - 1] * pool_size;
		grad_input_shape.push_back(h);
		grad_input_shape.push_back(w);
		Tensor grad_input(grad_input_shape, grad_output.device);

		dim3 blocks(batch_size, h);
		dim3 threads(std::min(w, (int64_t)kCudaThreadsNum));
		pool_backward_kernel<<<blocks, threads>>>((const float*)grad_output.data_ptr(), (const int8_t*)mask.data_ptr(), (float*)grad_input.data_ptr(), h, w, pool_size);
		return grad_input;
	}
}