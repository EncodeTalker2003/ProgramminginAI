#include <pybind11/pybind11.h>
using namespace pybind11::literals;

#include "basics.h"
#include "op.h"

PYBIND11_MODULE(MyTorch, m) {
	init_basics(m);
	init_op(m);
}
