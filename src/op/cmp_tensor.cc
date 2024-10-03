#include "cmp_tensor.h"
#include "src/backend/cpu/cmp_tensor.h"
#include "src/backend/cuda/cmp_tensor.h"

namespace MyTorch{
	bool cmp_tensor(const Tensor& input1, const Tensor& input2) {
		if (input1.device.device_type != input2.device.device_type) {
			LOG_FATAL("Device type mismatch in cmp_tensor");
		}
		bool res = DISPATCH_TO_BACKEND(input1.device.device_type, cmp_tensor(input1, input2));
		return res;
	}
}