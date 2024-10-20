#include "mat_mul.h"
#include "src/backend/cuda/mat_mul.h"
#include "src/basics/log.h"

namespace MyTorch{

	// C(m,n) = A(m * k) * B(k * n)
	// dL/dA = B^T * dL/dC
	// dL/dB = dL/dC * A^T
	Tensor matmul_forward_manual(const Tensor& a, const Tensor& b, OpContext &cxt) {
		if (a.device != b.device) {
			LOG_ERROR("matmul_forward: Tensors should be on the same device");
		}
		if ((a.dim() != 2) || (b.dim() != 2)) {
			LOG_ERROR("matmul_forward: Tensors should be 2D");
		}
		cxt.push_back(a);
		cxt.push_back(b);
		return MyTorch::Backend::CUDA::matmul(a, b, false, false);
	}

	std::pair<Tensor, Tensor> matmul_backward_manual(const Tensor& grad_output, OpContext &cxt) {
		Tensor b = cxt.pop_back();
		Tensor a = cxt.pop_back();
		Tensor grad_a = MyTorch::Backend::CUDA::matmul(b, grad_output, false, true);
		Tensor grad_b = MyTorch::Backend::CUDA::matmul( grad_output, a, true, false);
		return std::make_pair(grad_a, grad_b);
	}
}