add_library(mytorch_backend_cuda STATIC
	relu.cu
	sigmoid.cu
	cmp_tensor.cu
	mat_mul.cu
	mat_transpose.cu
	convolution.cu
	pooling.cu
	softmax_and_CELoss.cu
	reduce.cuh
)

target_link_libraries(mytorch_backend_cuda PUBLIC  
	mytorch_basics 
	CUDA::cudart
	CUDA::cublas)
