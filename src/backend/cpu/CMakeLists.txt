add_library(mytorch_backend_cpu STATIC
	relu.cc
	sigmoid.cc
	cmp_tensor.cc
)

target_link_libraries(mytorch_backend_cpu PUBLIC mytorch_basics)
