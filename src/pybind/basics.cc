#include "basics.h"
#include "src/basics/tensor.h"
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace pybind11::literals;
using MyTorch::Tensor;

void init_basics(pybind11::module& m) {
	pybind11::class_<MyTorch::Device>(m, "Device")
		.def("to_string", &MyTorch::Device::to_string)
		.def("switch_to_this_device", &MyTorch::Device::switch_to_this_device)
		.def_static("cpu", &MyTorch::Device::cpu)
		.def_static("cuda", &MyTorch::Device::cuda, "device_index"_a = 0)
		.def("__eq__", &MyTorch::Device::operator==)
		.def("__ne__", &MyTorch::Device::operator!=);

	pybind11::enum_<MyTorch::data_t>(m, "data_t")
		.value("float32", MyTorch::data_t::FLOAT32)
		.value("int32", MyTorch::data_t::INT32);

	pybind11::class_<MyTorch::Tensor>(m, "Tensor")
		.def_readwrite("Device", &Tensor::device)
		.def_readwrite("offset", &Tensor::offset)
		.def_readwrite("shape", &Tensor::shape)
		.def_readwrite("strides", &Tensor::strides)
		.def_readwrite("data_type", &Tensor::data_type)
		.def("numel", &Tensor::numel)
		.def("dim", &Tensor::dim)
		.def("reahspe", &Tensor::reshape, "new_shape"_a)
		.def("data_ptr", &Tensor::data_ptr, pybind11::return_value_policy::reference)
		.def("__repr__", &Tensor::to_string, "lim"_a = 16)
		.def("print", &Tensor::print, "lim"_a = 16)
		.def("to_string", &Tensor::to_string, "lim"_a = 16)
		.def("__str__", &Tensor::to_string, "lim"_a = 16)
		.def("to", &Tensor::to, "device"_a)
		.def("cpu", &Tensor::cpu)
		.def("cuda", &Tensor::cuda, "index"_a = 0)
		.def_static("zeros", &Tensor::zeros, "shape"_a, "device"_a)
		.def_static("randu", &Tensor::randu, "shape"_a, "device"_a, "lo"_a = 0.0, "hi"_a = 1.0)
		.def_static("from_data", &Tensor::from_data, "data"_a, "data_type"_a, "shape"_a, "device"_a)
		.def("__eq__", &Tensor::operator==)
		.def(pybind11::init([](const std::vector<float> data, MyTorch::data_t dtype, const std::vector<int64_t> shape, MyTorch::Device device) {
			std::vector<float> data_flat = std::vector<float>(data.begin(), data.end());
			return new Tensor(Tensor::from_data(data_flat, dtype, shape, device));
		}))
		.def(pybind11::init([](const std::vector<int32_t> data, MyTorch::data_t dtype, const std::vector<int64_t> shape, MyTorch::Device device) {
			std::vector<int32_t> data_flat = std::vector<int32_t>(data.begin(), data.end());
			return new Tensor(Tensor::from_int_data(data, dtype, shape, device));
		}))
		.def(pybind11::init([](pybind11::array_t<float> data, MyTorch::data_t dtype, MyTorch::Device device) {
			std::vector<int64_t> shape;
			int64_t numel = data.size();
			for (int i = 0; i < data.ndim(); i++) {
				shape.push_back(data.shape(i));
			}
			pybind11::array_t<float> data_1d = data.reshape({numel});
			std::vector<float> data_vec;
			for (int i = 0; i < numel; i++) {
				data_vec.push_back(data_1d.at(i));
			}
			return new Tensor(Tensor::from_data(data_vec, dtype, shape, device));
		}));
}