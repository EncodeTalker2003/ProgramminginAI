#include "op.h"
using namespace pybind11::literals; 

#include "src/op/relu.h"
#include "src/op/sigmoid.h"
#include "src/op/mat_mul.h"
#include "src/op/convolution.h"
#include "src/op/pooling.h"
#include "src/op/softmax_and_CELoss.h"

void init_op(pybind11::module& m) {
	auto op_m = m.def_submodule("op");

	op_m.def("relu_forward", MyTorch::relu_forward, "input"_a);

	op_m.def("sigmoid_forward", MyTorch::sigmoid_forward, "input"_a);

	op_m.def("matmul_forward", MyTorch::matmul_forward, "a"_a, "b"_a);

	op_m.def("conv_forward", MyTorch::conv_forward, "input"_a, "kernel"_a);

	op_m.def("pooling_forward", MyTorch::max_pool, "input"_a, "pool_size"_a);

	op_m.def("softmax_and_CELoss_forward", MyTorch::softmax_and_CELoss_forward, "input"_a, "truth"_a);
}