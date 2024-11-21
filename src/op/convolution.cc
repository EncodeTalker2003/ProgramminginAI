#include "convolution.h"
#include "src/backend/cuda/convolution.h"
#include "src/backend/cuda/mat_mul.h"
#include "src/backend/cuda/mat_transpose.h"

namespace MyTorch{
	Tensor conv_forward_manual(const std::vector<Tensor> inputs, OpContext &cxt, void* args) {
		Tensor image = inputs[0];
		Tensor kernel = inputs[1];
		if (image.dim() != 4) {
			LOG_FATAL("Convolution forward: Input image must be 4D tensor.");
		}
		if (kernel.dim() != 4) {
			LOG_FATAL("Convolution forward: Kernel must be 4D tensor.");
		}
		int64_t n = image.shape[0];
		int64_t c_in = image.shape[1];
		int64_t h = image.shape[2];
		int64_t w = image.shape[3];
		int64_t c_out = kernel.shape[0];
		int64_t kh = kernel.shape[2];
		int64_t kw = kernel.shape[3];
		
		if (c_in != kernel.shape[1]) {
			LOG_FATAL("Convolution forward: Input channel must be equal to kernel channel.");
		}
		if ((kh % 2 == 0) || (kw % 2 == 0)) {
			LOG_FATAL("Convolution forward: Kernel size must be odd.");
		}
		Tensor im2col_ret = MyTorch::Backend::CUDA::im2col(image, kh, kw);
		Tensor im2col_ret_reshaped = im2col_ret.reshape({n * h * w, c_in * kh * kw});
		Tensor kernel_reshaped = kernel.reshape({c_out, c_in * kh * kw});
		// kernel * im2col^T => [c_out, n, h, w]
		// im2col * kernel^T => [n, h, w, c_out] 
		Tensor prod = MyTorch::Backend::CUDA::matmul(kernel_reshaped, im2col_ret_reshaped, false, true).reshape({c_out, n, h, w});
		Tensor conv = MyTorch::Backend::CUDA::transpose(prod, 0, 1);
		cxt.push_back(kernel);
		cxt.push_back(im2col_ret); 
		return conv;
	}

	std::pair<Tensor, Tensor> conv_backward_manual(const Tensor& grad_output, OpContext &cxt) {
		Tensor im2col_ret = cxt.pop_back();
		Tensor kernel = cxt.pop_back();
		int64_t n = im2col_ret.shape[0];
		int64_t h_w = im2col_ret.shape[1];
		int64_t c_in_kh_kw = im2col_ret.shape[2];
		int64_t c_out = kernel.shape[0];
		int64_t c_in = kernel.shape[1];
		int64_t kh = kernel.shape[2];
		int64_t kw = kernel.shape[3];
		int64_t h = grad_output.shape[2];
		int64_t w = grad_output.shape[3];
		if ((h * w != h_w) || (c_in * kh * kw != c_in_kh_kw)) {
			LOG_FATAL("Convolution backward: Shape mismatch.");
		}
		Tensor grad_prod = MyTorch::Backend::CUDA::transpose(grad_output, 0, 1).reshape({c_out, n * h * w});
		Tensor im2col_ret_grad = MyTorch::Backend::CUDA::matmul_batch(
			grad_prod,
			kernel.reshape({c_out, c_in * kh * kw}),
			true, false
		).reshape({n, h * w, c_in * kh * kw});

		Tensor kernel_grad = MyTorch::Backend::CUDA::matmul_batch(
			grad_prod,
			im2col_ret.reshape({n * h * w, c_in * kh * kw}),
			false, false
		).reshape({c_out, c_in, kh, kw});

		Tensor img_grad = MyTorch::Backend::CUDA::col2im(im2col_ret_grad, c_in, h, w, kh, kw);

		return std::make_pair(img_grad, kernel_grad);
	}

	Tensor conv_forward(const Tensor &input, const Tensor &kernel) {
		OpContext cxt;
		return conv_forward_manual({input, kernel}, cxt, NULL);
	}
}