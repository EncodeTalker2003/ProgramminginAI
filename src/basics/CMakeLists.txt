add_library(mytorch_basics STATIC
	device.cc
	mem.cc
	tensor.cc
)

target_link_libraries(mytorch_basics PUBLIC CUDA::cudart)
