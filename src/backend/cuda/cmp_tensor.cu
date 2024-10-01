#include "cmp_tensor.h"

namespace MyTorch::Backend::CUDA {
	bool cmp_tensor(const Tensor& input1, const Tensor& input2) {
		if (input1.shape != input2.shape) {
			return false;
		}
		int64_t tot = input1.numel();
		float* input1_ptr = (float*)input1.data_ptr();
		float* input2_ptr = (float*)input2.data_ptr();
		for (int64_t i = 0; i < tot; i++) {
			if (input1_ptr[i] != input2_ptr[i]) {
				return false;
			}
		}
		return true;
	}
}