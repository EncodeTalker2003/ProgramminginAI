#pragma once
#include "src/basics/log.h"

namespace MyTorch::Backend::CUDA {
	const int kCudaThreadsNum = 1024;
	inline int CudaGetBlocks(const int64_t N) {
		return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
	}
	// Define the grid stride looping
	#define CUDA_KERNEL_LOOP(i, n) \
		for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
			 i < (n); \
			 i += blockDim.x * gridDim.x)
}