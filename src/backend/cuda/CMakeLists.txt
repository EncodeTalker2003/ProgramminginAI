add_library(mytorch_backend_cuda STATIC
	relu.cu
	sigmoid.cu
	cmp_tensor.cu
)

target_link_libraries(mytorch_backend_cuda PUBLIC  mytorch_basics CUDA::cudart)
