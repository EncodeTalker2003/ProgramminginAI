#include "convolution.h"
#include <cuda_runtime.h>

namespace MyTorch::Backend::CUDA {

	__global__ void im2col_kernel(const float* input, float* output, const int64_t n, const int64_t c_in, const int64_t h, const int64_t w, const int64_t kh, const int64_t kw) {
		int64_t n_id = blockIdx.x;
		int64_t c_id = blockIdx.y;
		int64_t h_id = blockIdx.z;
		for (int64_t w_id = threadIdx.x; w_id < w; w_id += blockDim.x) {
			float* cur_output = output + 
				n_id * (h * w) * (kh * kw * c_in) + 
				h_id * w * (kh * kw * c_in) + 
				w_id * (kh * kw * c_in) + 
				c_id * (kh * kw);
			const float* cur_input = input + 
				n_id * c_in * h * w + 
				c_id * h * w + 
				h_id * w + 
				w_id;
			for (int64_t kh_id = -(kh / 2); kh_id <= kh / 2; kh_id++) {
				for (int64_t kw_id = -(kw / 2); kw_id <= kw / 2; kw_id++) {
					int64_t cur_h_id = h_id + kh_id;
					int64_t cur_w_id = w_id + kw_id;
					*cur_output = (cur_h_id >= 0 && cur_h_id < h && cur_w_id >= 0 && cur_w_id < w) ? cur_input[kh_id * w + kw_id] : 0;
					cur_output++;
				}
			}
		}
	}


	Tensor im2col(const Tensor &input, const int64_t kh, const int64_t kw) {
		if (input.dim() != 4) {
			LOG_ERROR("im2col: input should be 4D");
		}
		if ((kh % 2 != 0) || (kw % 2 != 0)) {
			LOG_ERROR("im2col: kernel size should be odd");
		}
		int64_t n = input.shape[0];
		int64_t c_in = input.shape[1];
		int64_t h = input.shape[2];
		int64_t w = input.shape[3];
		Tensor output({n, h * w, c_in * kh * kw}, input.device);

		dim3 blocks(n, c_in, h);
		dim3 threads(std::min(w, (int64_t)kCudaThreadsNum));
		im2col_kernel<<<blocks, threads>>>((const float*)input.data_ptr(), (float*)output.data_ptr(), n, c_in, h, w, kh, kw);
		return output;
	}

	__global__ void col2im_kernel(const float* input, float* output, const int64_t n, const int64_t c_in, const int64_t h, const int64_t w, const int64_t kh, const int64_t kw) {
		int64_t n_id = blockIdx.x;
		int64_t c_id = blockIdx.y;
		int64_t h_id = blockIdx.z;
		for (int64_t w_id = threadIdx.x; w_id < w; w_id += blockDim.x) {
			float grad_sum = 0.0;
			for (int64_t kh_id = -(kh / 2); kh_id <= kh / 2; kh_id++) {
				for (int64_t kw_id = -(kw / 2); kw_id <= kw / 2; kw_id++) {
					int64_t h_center = h_id + kh_id;
					int64_t w_center = w_id + kw_id;
					if ((h_center >= 0) && (h_center < h) && (w_center >= 0) && (w_center < w)) {
						int64_t h_rel = kh / 2 - kh_id;
						int64_t w_rel = kw / 2 - kw_id;
						int64_t cur_pos = 
							n_id * (h * w) * (kh * kw * c_in) +
							h_center * w * (kh * kw * c_in) +
							w_center * (kh * kw * c_in) +
							c_id * (kh * kw) +
							h_rel * kw + w_rel;
						grad_sum += input[cur_pos];
					}
				}
			}
			int64_t cur_pos = 
				n_id * c_in * h * w +
				c_id * h * w +
				h_id * w + w_id;
			output[cur_pos] = grad_sum;
		}
	}

	Tensor col2im(const Tensor &input, const int64_t c_in, const int64_t h, const int64_t w, const int64_t kh, const int64_t kw) {
		if (input.dim() != 3) {
			LOG_ERROR("col2im: input should be 3D");
		}
		if ((kh % 2 != 0) || (kw % 2 != 0)) {
			LOG_ERROR("col2im: kernel size should be odd");
		}
		
		int64_t n = input.shape[0];
		Tensor output({n, c_in, h, w}, input.device);
		dim3 blocks(n, c_in, h);
		dim3 threads(std::min(w, (int64_t)kCudaThreadsNum));
		col2im_kernel<<<blocks, threads>>>((const float*)input.data_ptr(), (float*)output.data_ptr(), n, c_in, h, w, kh, kw);
		return output;
	}
}