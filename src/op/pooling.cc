#include "pooling.h"
#include "src/backend/cuda/pooling.h"

namespace MyTorch{
	Tensor pooling_forward_manual(const std::vector<Tensor> &inputs, OpContext &cxt, void* args) {
		int64_t pool_size = ((Pooling_args*)args)->pool_size;
		auto [output, mask] = MyTorch::Backend::CUDA::pool_forward(inputs[0], pool_size);
		cxt.push_back(mask);
		cxt.save_pool_size(pool_size);
		return output;
	}

	Tensor pooling_backward_manual(const Tensor& grad_output, OpContext &cxt) {
		Tensor mask = cxt.pop_back();
		int64_t pool_size = cxt.get_pool_size();
		return MyTorch::Backend::CUDA::pool_backward(grad_output, mask, pool_size);
	}

	Tensor max_pool(const Tensor &input, int64_t pool_size) {
		Pooling_args args;
		args.pool_size = pool_size;
		OpContext cxt;
		return MyTorch::pooling_forward_manual({input}, cxt, &args);
	}
}