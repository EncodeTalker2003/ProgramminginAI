#pragma once

#include "mem.h"
#include "device.h"
#include "log.h"
#include <vector>
#include <string>

namespace MyTorch {
	enum class data_t {
		FLOAT32,
		INT32
	};

	class Tensor {
	private:
		Tensor(const Device &dev, const MemData &data, const std::vector<int64_t> &siz, const int64_t &off, const data_t dt);
	
	public: 
	
		Device device;
		int64_t offset;
		std::vector<int64_t> shape;
		std::vector<int64_t> strides;
		MemData mem_data;
		data_t data_type;
		

		Tensor(const Tensor &other) = default;
		// Create a empty tensor
		Tensor(const std::vector<int64_t> &shape, const Device &device, data_t data_type = data_t::FLOAT32);
		// Create a tensor with the given data
		static Tensor from_data(const std::vector<float> &data, const data_t data_type, const std::vector<int64_t> &shape, const Device &device);
		static Tensor from_int_data(const std::vector<int32_t> &data, const data_t data_type, const std::vector<int64_t> &shape, const Device &device);
		// Create a tensor with all 0s
		static Tensor zeros(const std::vector<int64_t> &shape, const Device &device);
		// Create a tensor with random values in [lo, hi]
		static Tensor randu(const std::vector<int64_t> &shape, const Device &device, float lo = 0.0, float hi = 1.0);
		// Set the value on `pos` to be `val` use memcpy
		void set_pos_data(int pos, float val);
		// Reshape the tensor
		Tensor reshape(const std::vector<int64_t> &new_shape) const;

		// Return the number of elements in the tensor
		int64_t numel() const;
		// Return the number of dimensions of the tensor
		int dim() const;
		// Return the offset of the element at the given position
		int64_t get_elem_offset(const std::vector<int64_t> &pos) const;
		// Return the address of the data
		void* data_ptr() const;
		// Return the address of the element at the given position
		void* get_elem_ptr(const std::vector<int64_t> &pos) const;
		// Return the element at the given position
		Tensor get_elem(const std::vector<int64_t> &pos) const;
		// Print the tensor to console 
		std::string to_string(int lim = 16) const;
		void print(int lim = 16) const;

		// Return a new tensor with the same data but on the given device
		Tensor to(const Device &device) const;
		// Return a new tensor with the same data but on the CPU
		Tensor cpu() const;
		// Return a new tensor with the same data but on the CUDA device
		Tensor cuda(int index = 0) const;

		bool operator == (const Tensor &other) const;
	};
}