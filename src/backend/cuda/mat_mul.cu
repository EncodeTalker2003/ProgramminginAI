#include "mat_mul.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace MyTorch::Backend::CUDA {
	cublasHandle_t handle;

	class CublasHandleHolder {
		CublasHandleHolder() {
			cublasCreate(&handle);
		}

		~CublasHandleHolder() {
			cublasDestroy(handle);
		}
	}_cublas_handle_holder;

	void gemm_inter(
		int rowa,
		int colb,
		int cola,
		const Tensor &a,
		const Tensor &b,
		Tensor &c,
		bool transpose_a,
		bool transpose_b,
		int batch_cnt,
		int64_t stride_a,
		int64_t stride_b,
		int64_t stride_c
	) {
		float alpha = 1.0, beta = 0.0;
		cublasStatus_t status = cublasGemmStridedBatchedEx(
			handle,
			transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
			transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
			colb,
			rowa,
			cola,
			&alpha,
			(const float*)b.data_ptr(),
			CUDA_R_32F,
			transpose_b ? cola : colb,
			stride_b,
			(const float*)a.data_ptr(),
			CUDA_R_32F,
			transpose_a ? rowa : cola,
			stride_a,
			&beta,
			(float*)c.data_ptr(),
			CUDA_R_32F,
			colb,
			stride_c,
			batch_cnt,
			CUDA_R_32F,
			CUBLAS_GEMM_DEFAULT
		);
		if (status != CUBLAS_STATUS_SUCCESS) {
			LOG_FATAL("gemm_inter failed: %d", status);
		}
	}

	Tensor matmul(const Tensor& a, const Tensor& b, bool transpose_a = false, bool transpose_b = false) {
		if ((a.dim() != 2) && (b.dim() != 2)) {
			LOG_FATAL("matmul: Only support 2D tensor");
		}
		int rowa = transpose_a? a.shape[1] : a.shape[0];
		int cola = transpose_a? a.shape[0] : a.shape[1];
		int rowb = transpose_b? b.shape[1] : b.shape[0];
		int colb = transpose_b? b.shape[0] : b.shape[1];
		if (cola != rowb) {
			LOG_FATAL("matmul: Dimension mismatch");
		}
		Tensor c({rowa, colb}, a.device);
		gemm_inter(
			rowa,
			colb,
			cola,
			a,
			b,
			c,
			transpose_a,
			transpose_b,
			1,
			1ll * rowa * colb,
			1ll * rowb * colb,
			1ll * rowa * colb
		);
		return c;
	}

	Tensor matmul_batch(const Tensor& a, const Tensor& b, bool transpose_a = false, bool transpose_b = false) {
		if ((a.dim() == 2) && (b.dim() == 2)) {
			return matmul(a, b, transpose_a, transpose_b);
		} else if ((a.dim() == 2) && (b.dim() == 3)) {
			int batch_cnt = b.shape[0];
			int rowa = transpose_a? a.shape[1] : a.shape[0];
			int cola = transpose_a? a.shape[0] : a.shape[1];
			int rowb = transpose_b? b.shape[2] : b.shape[1];
			int colb = transpose_b? b.shape[1] : b.shape[2];
			if (cola != rowb) {
				LOG_FATAL("matmul_batch: Dimension mismatch");
			}
			Tensor c({batch_cnt, rowa, colb}, a.device);
			gemm_inter(
				rowa,
				colb,
				cola,
				a,
				b,
				c,
				transpose_a,
				transpose_b,
				batch_cnt,
				0,
				1ll * rowb * colb,
				1ll * rowa * colb
			); 
			return c;
		} else if ((a.dim() == 3) && (b.dim() == 2)) {
			int batch_cnt = a.shape[0];
			int rowa = transpose_a? a.shape[2] : a.shape[1];
			int cola = transpose_a? a.shape[1] : a.shape[2];
			int rowb = transpose_b? b.shape[1] : b.shape[0];
			int colb = transpose_b? b.shape[0] : b.shape[1];
			if (cola != rowb) {
				LOG_FATAL("matmul_batch: Dimension mismatch");
			}
			Tensor c({batch_cnt, rowa, colb}, a.device);
			gemm_inter(
				rowa,
				colb,
				cola,
				a,
				b,
				c,
				transpose_a,
				transpose_b,
				batch_cnt,
				1ll * rowa * colb,
				0,
				1ll * rowa * colb
			);
			return c;
		} else if ((a.dim() == 3) && (b.dim() == 3)) {
			int batch_cnt = a.shape[0];
			int rowa = transpose_a? a.shape[2] : a.shape[1];
			int cola = transpose_a? a.shape[1] : a.shape[2];
			int rowb = transpose_b? b.shape[2] : b.shape[1];
			int colb = transpose_b? b.shape[1] : b.shape[2];
			if (cola != rowb) {
				LOG_FATAL("matmul_batch: Dimension mismatch");
			}
			Tensor c({batch_cnt, rowa, colb}, a.device);
			gemm_inter(
				rowa,
				colb,
				cola,
				a,
				b,
				c,
				transpose_a,
				transpose_b,
				batch_cnt,
				1ll * rowa * colb,
				1ll * rowb * colb,
				1ll * rowa * colb
			);
			return c;
		} else {
			LOG_FATAL("matmul_batch: Not valid dimension for input");
		}
	}
}