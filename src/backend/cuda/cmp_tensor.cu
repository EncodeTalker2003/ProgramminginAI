#include "cmp_tensor.h"
#include <cuda_runtime.h>

namespace MyTorch::Backend::CUDA {

	__device__ bool is_equal(float x, float y) {
		float abs_err = abs(x - y);
		float rel_err = abs_err / (max(abs(x), abs(y)) + 1e-6);
		if (abs_err < ABS_ERR_THRESHOLD || rel_err < REL_ERR_THRESHOLD) {
			return true;
		} else {
			return false;
		}
	}

	static __device__ int64_t neq_flag = 0;

	__global__ void cmp_tensor_kernel(float* input1, float* input2, int64_t n, int64_t tag) {
		CUDA_KERNEL_LOOP(i, n) {
			if (!is_equal(input1[i], input2[i])) {
				neq_flag = tag;
			}
		}
	}

	bool cmp_tensor(const Tensor& input1, const Tensor& input2) {
		static int tag = 1;
		tag += 1;
		if (input1.shape != input2.shape) {
			return false;
		}
		int64_t tot = input1.numel();
		float* input1_ptr = (float*)input1.data_ptr();
		float* input2_ptr = (float*)input2.data_ptr();
		int block_size = kCudaThreadsNum;
		int grid_size = CudaGetBlocks(tot);
		cmp_tensor_kernel<<<grid_size, block_size>>>(input1_ptr, input2_ptr, tot, tag);
		int64_t flag;
		cudaMemcpyFromSymbol(&flag, neq_flag, sizeof(int64_t), 0, cudaMemcpyDeviceToHost);
		if (flag == tag) {
			return false;
		} else {
			return true;
		}
	}
}