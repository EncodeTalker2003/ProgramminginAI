#include "mat_transpose.h"
#include "src/basics/log.h"
#include <cuda_runtime.h>

namespace MyTorch::Backend::CUDA {
	const int64_t GRID_SIDE_SIZE = 16;
	const int64_t BLOCK_SIDE_SIZE = 32;

	__global__ void transpose_kernel(const float* input, float* output, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5) {
		for (int64_t i1 = blockIdx.x; i1 < d1; i1 += gridDim.x) {
			for (int64_t i2 = blockIdx.y; i2 < d2; i2 += gridDim.y) {
				for (int64_t i3 = blockIdx.z; i3 < d3; i3 += gridDim.z) {
					for (int64_t i4 = threadIdx.x; i4 < d4; i4 += blockDim.x) {
						for (int64_t i5 = threadIdx.y; i5 < d5; i5 += blockDim.y) {
							int64_t idx1 = i1 * d2 * d3 * d4 * d5 + i2 * d3 * d4 * d5 + i3 * d4 * d5 + i4 * d5 + i5;
							int64_t idx2 = i1 * d4 * d3 * d2 * d5 + i4 * d3 * d2 * d5 + i3 * d2 * d5 + i2 * d5 + i5;
							output[idx2] = input[idx1];
						}
					}
				}
			}
		}
		
	}

	Tensor transpose(const Tensor &input, int axis1, int axis2) {
		if ((axis1 < 0) || (axis1 >= input.dim()) || (axis2 < 0) || (axis2 >= input.dim())) {
			LOG_ERROR("mat_transpose: invalid axis");
		}
		int64_t d1 = 1;
		for (int i = 0; i < axis1; i++) {
			d1 *= input.shape[i];
		}
		int64_t d2 = input.shape[axis1];
		int64_t d3 = 1;
		for (int i = axis1 + 1; i < axis2; i++) {
			d3 *= input.shape[i];
		}
		int64_t d4 = input.shape[axis2];
		int64_t d5 = 1;
		for (int i = axis2 + 1; i < input.dim(); i++) {
			d5 *= input.shape[i];
		}
		std::vector<int64_t> new_shape = input.shape;
		std::swap(new_shape[axis1], new_shape[axis2]);
		Tensor output(new_shape, input.device);

		dim3 blocks(
			std::min(d1, GRID_SIDE_SIZE),
			std::min(d2, GRID_SIDE_SIZE),
			std::min(d3, GRID_SIDE_SIZE)
		);
		dim3 threads(
			std::min(d4, BLOCK_SIDE_SIZE),
			std::min(d5, BLOCK_SIDE_SIZE)
		);
		transpose_kernel<<<blocks, threads>>>(
			(const float*) input.data_ptr(),
			(float*) output.data_ptr(),
			d1, d2, d3, d4, d5
		);

	}
}