#include "mat_mul.h"
#include "src/backend/cuda/mat_mul.h"
#include "src/basics/log.h"

namespace MyTorch{

	// C(m,n) = A(m * k) * B(k * n)
	// dL/dA = B^T * dL/dC
	// dL/dB = dL/dC * A^T
	Tensor matmul_forward_manual(const std::vector<Tensor> &inputs, OpContext &cxt, void* args) {
		Tensor a = inputs[0];
		Tensor b = inputs[1];
		if (a.device != b.device) {
			LOG_FATAL("matmul_forward: Tensors should be on the same device");
		}
		if ((a.dim() != 2) || (b.dim() != 2)) {
			LOG_FATAL("matmul_forward: Tensors should be 2D");
		}
		cxt.push_back(a);
		cxt.push_back(b);
		return MyTorch::Backend::CUDA::matmul(a, b, false, false);
	}

	std::pair<Tensor, Tensor> matmul_backward_manual(const Tensor& grad_output, OpContext &cxt) {
		Tensor b = cxt.pop_back();
		Tensor a = cxt.pop_back();
		Tensor grad_a = MyTorch::Backend::CUDA::matmul(grad_output, b, false, true);
		Tensor grad_b = MyTorch::Backend::CUDA::matmul(a, grad_output, true, false);
		return std::make_pair(grad_a, grad_b);
	}

	Tensor matmul_forward(const Tensor &a, const Tensor &b) {
		OpContext cxt;
		return matmul_forward_manual({a, b}, cxt, NULL);
	}
}