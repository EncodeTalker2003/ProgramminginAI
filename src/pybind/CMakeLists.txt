set(PYMODULE_NAME "MyTorch")

pybind11_add_module(${PYMODULE_NAME} MODULE
	pybind.cc
	basics.cc
	op.cc
)

target_link_libraries(${PYMODULE_NAME} PUBLIC
	CUDA::cudart 
	mytorch_op
	mytorch_backend_cuda
	mytorch_basics
)

target_compile_options(${PYMODULE_NAME} PRIVATE
	-flto=auto	
)
