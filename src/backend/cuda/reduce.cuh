#pragma once

// Reduction operations on CUDA

#include <cuda_runtime.h>
#include "utils.h"

namespace MyTorch::Backend::CUDA {
	#define WARP_SIZE 32
	#define FULL_MASK 0xffffffff
	
	__device__ __forceinline__ float warp_reduce_sum(float val) {
		#pragma unroll
		for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
			val += __shfl_xor_sync(FULL_MASK, val, offset);
		}
		return val;
	}

	__device__ __forceinline__ float block_reduce_sum(float val) {
		static __shared__ float sdata[32]; // 32 * 32 = 1024
		int lane = threadIdx.x % WARP_SIZE;
		int wid = threadIdx.x / WARP_SIZE;
		val = warp_reduce_sum(val);
		if (lane == 0) sdata[wid] = val;
		__syncthreads();

		if (wid == 0) {
			val = (threadIdx.x * WARP_SIZE < blockDim.x) ? sdata[lane] : 0;
			val = warp_reduce_sum(val);
		}
		return val;
	}

	__device__ __forceinline__ float block_reduce_sum_broadcast(float val) {
		static __shared__ float sdata[32]; // 32 * 32 = 1024
		int lane = threadIdx.x % WARP_SIZE;
		int wid = threadIdx.x / WARP_SIZE;
		val = warp_reduce_sum(val);
		if (lane == 0) sdata[wid] = val;
		__syncthreads();

		if (wid == 0) {
			val = (threadIdx.x * WARP_SIZE < blockDim.x ) ? sdata[lane] : 0;
			val = warp_reduce_sum(val);
			sdata[lane] = val;
		}
		__syncthreads();
		val = sdata[0];
		return val;
	}

	__device__ __forceinline__ float warp_reduce_max(float val) {
		#pragma unroll
		for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
			val = max(val, __shfl_xor_sync(FULL_MASK, val, offset));
		}
		return val;
	}

	__device__ __forceinline__ float block_reduce_max(float val) {
		static __shared__ float sdata[32]; // 32 * 32 = 1024
		int lane = threadIdx.x % WARP_SIZE;
		int wid = threadIdx.x / WARP_SIZE;
		val = warp_reduce_max(val);
		if (lane == 0) sdata[wid] = val;
		__syncthreads();

		if (wid == 0) {
			val = (threadIdx.x * WARP_SIZE < blockDim.x) ? sdata[lane] : -1e20;
			val = warp_reduce_max(val);
		}
		return val;
	}

	__device__ __forceinline__ float block_reduce_max_broadcast(float val) {
		static __shared__ float sdata[32]; // 32 * 32 = 1024
		int lane = threadIdx.x % WARP_SIZE;
		int wid = threadIdx.x / WARP_SIZE;
		val = warp_reduce_max(val);
		if (lane == 0) sdata[wid] = val;
		__syncthreads();

		if (wid == 0) {
			val = (threadIdx.x * WARP_SIZE < blockDim.x) ? sdata[lane] : -1e20;
			val = warp_reduce_max(val);
			sdata[lane] = val;
		}
		__syncthreads();
		val = sdata[0];
		return val;
	}

}