add_library(mytorch_op STATIC
	context.cc
	relu.cc
	sigmoid.cc
	cmp_tensor.cc
	mat_mul.cc
	convolution.cc
	pooling.cc
	softmax_and_CELoss.cc
)

target_link_libraries(mytorch_op PUBLIC 
	mytorch_basics 
	CUDA::cudart 
	mytorch_backend_cpu 
	mytorch_backend_cuda
)
