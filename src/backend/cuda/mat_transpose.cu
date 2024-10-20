#include "mat_transpose.h"
#include "src/basics/log.h"
#include <cuda_runtime.h>

namespace MyTorch::Backend::CUDA {
	const int BLOCK_SIZE = 16;

	__global__ void transpose_kernel(const float* input, float* output, int row, int col) {
		__shared__ float sdata[BLOCK_SIZE][BLOCK_SIZE + 1];

		int64_t x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		int64_t y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
		if ((x < row) && (y < col)) {
			sdata[threadIdx.y][threadIdx.x] = input[y * row + x];
		}
		__syncthreads();

		x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
		y = blockIdx.x * BLOCK_SIZE + threadIdx.y;
		if ((x < col) && (y < row)) {
			output[y * col + x] = sdata[threadIdx.x][threadIdx.y];
		}
	}

	Tensor transpose(const Tensor &input) {
		int dim = input.dim();
		if (dim != 2) {
			LOG_FATAL("Transpose: input must be 2D tensor");
		}
		int row = input.shape[0], col = input.shape[1];
		Tensor output({col, row}, input.device);
		dim3 blocks((col + BLOCK_SIZE - 1) / BLOCK_SIZE, (row + BLOCK_SIZE - 1) / BLOCK_SIZE);
		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
		transpose_kernel<<<blocks, threads>>>((const float*)input.data_ptr(), (float*)output.data_ptr(), row, col);
	}
}