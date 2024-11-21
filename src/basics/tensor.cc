#include "tensor.h"
#include "src/op/cmp_tensor.h"
#include <functional>

namespace MyTorch{
	Tensor::Tensor(const Device &dev, const MemData &data, const std::vector<int64_t> &siz, const int64_t &off, const data_t dt) : device(dev), offset(off), shape(siz), mem_data(data), data_type(dt) {
		strides.resize(shape.size());
		if (!shape.empty()) {
			strides[shape.size() - 1] = 1;
			for (int i = shape.size() - 2; i >= 0; i--) {
				strides[i] = strides[i + 1] * shape[i + 1];
			}
		}
	}

	int64_t Tensor::numel() const {
		int64_t res = 1;
		for (auto x: shape) {
			res *= x;
		}
		return res;
	}

	Tensor::Tensor(const std::vector<int64_t> &siz, const Device &dev, const data_t dt) : device(dev), offset(0), shape(siz), mem_data(dev, numel() * sizeof(float)), data_type(dt) {
		strides.resize(shape.size());
		if (!shape.empty()) {
			strides[shape.size() - 1] = 1;
			for (int i = shape.size() - 2; i >= 0; i--) {
				strides[i] = strides[i + 1] * shape[i + 1];
			}
		}
	}

	Tensor Tensor::from_data(const std::vector<float> &data, const data_t data_type, const std::vector<int64_t> &shape, const Device &device) {
		if (data_type == data_t::FLOAT32) {
			Tensor res(shape, Device::cpu());
			int tot = res.numel();
			if (tot != (int)data.size()) {
				LOG_FATAL("Size mismatch in Tensor::from_data");
			}
			float* ptr = (float*)res.data_ptr();
			for (int i = 0; i < tot; i++) {
				ptr[i] = data[i];
			}
			return res.to(device);
		} else {
			Tensor res(shape, Device::cpu(), data_t::INT32);
			int tot = res.numel();
			if (tot != (int)data.size()) {
				LOG_FATAL("Size mismatch in Tensor::from_data");
			}
			int32_t* ptr = (int32_t*)res.data_ptr();
			for (int i = 0; i < tot; i++) {
				ptr[i] = data[i];
			}
			return res.to(device);
		}
	}

	Tensor Tensor::from_int_data(const std::vector<int32_t> &data, const data_t data_type, const std::vector<int64_t> &shape, const Device &device) {
		if (data_type != data_t::INT32) {
			LOG_FATAL("Data type mismatch in Tensor::from_int_data");
		}
		Tensor res(shape, Device::cpu(), data_t::INT32);
		int tot = res.numel();
		if (tot != (int)data.size()) {
			LOG_FATAL("Size mismatch in Tensor::from_data");
		}
		int32_t* ptr = (int32_t*)res.data_ptr();
		for (int i = 0; i < tot; i++) {
			ptr[i] = data[i];
		}
		return res.to(device);
	}

	Tensor Tensor::reshape(const std::vector<int64_t> &new_shape) const {
		int64_t new_numel = 1;
		for (auto x: new_shape) {
			new_numel *= x;
		}
		if (new_numel != numel()) {
			LOG_FATAL("Size mismatch in Tensor::reshape");
		}
		return Tensor(device, mem_data, new_shape, offset, data_type);
	}

	Tensor Tensor::zeros(const std::vector<int64_t> &shape, const Device &device) {
		//printf("ANA\n");
		Tensor res(shape, device);
		//int tot = res.numel();
		//LOG_DEBUG("tot = %d\n", tot);
		MyTorch::memset(device, res.mem_data.ptr, 0, res.numel() * sizeof(float));
		//printf("JAL\n");
		return res;
	}

	void Tensor::set_pos_data(int pos, float val) {
		if (this->device.device_type == device_t::CPU) {
			void* ptr = (char*)this->mem_data.ptr + pos * sizeof(float);
			*(float*)ptr = val;
		} else {
			void* dst_ptr =  (char*)this->mem_data.ptr + pos * sizeof(float);
			memcpy(this->device, dst_ptr, Device::cpu(), &val, sizeof(float));
		}
	}

	Tensor Tensor::randu(const std::vector<int64_t> &shape, const Device &device, float lo, float hi) {
		srand(time(NULL));
		Tensor res(shape, device);
		for (int i = 0; i < res.numel(); i++) {
			
			float val = lo + (hi - lo) * rand() / RAND_MAX;
			res.set_pos_data(i, val);
		}
		return res;
	}

	int64_t Tensor::get_elem_offset(const std::vector<int64_t> &pos) const {
		// pre-condition: pos.size() == shape.size() 
		int res = offset;
		for (int i = 0; i < (int)pos.size(); i++) {
			res += pos[i] * strides[i];
		}
		return res;
	}

	void* Tensor::data_ptr() const {
		return (char*)mem_data.ptr + offset * sizeof(float);
	}

	void* Tensor::get_elem_ptr(const std::vector<int64_t> &pos) const {
		if (pos.size() != shape.size()) {
			LOG_FATAL("Size mismatch in Tensor::get_elem_ptr");
		}
		return (char*)mem_data.ptr + get_elem_offset(pos) * sizeof(float);
	}

	Tensor Tensor::get_elem(const std::vector<int64_t> &pos) const {
		if (pos.size() != shape.size()) {
			LOG_FATAL("Size mismatch in Tensor::get_elem");
		}
		return Tensor(device, mem_data, {}, get_elem_offset(pos), data_type);
	}

	std::string Tensor::to_string(int lim) const {
		std::string res = "Tensor(";
		
		printf("Tensor(");
		if (shape.empty()) {
			if (this->device.device_type == device_t::CPU) {
				if (this->data_type == data_t::FLOAT32) {
					float x = *(float*)this->data_ptr();
					res += std::to_string(x) + " (scalar)";
				} else {
					int32_t x = *(int32_t*)this->data_ptr();
					res += std::to_string(x) + " (scalar)";
				}
			} else {
				Tensor t = this->to(Device::cpu());
				if (t.data_type == data_t::FLOAT32) {
					float x = *(float*)t.data_ptr();
					res += std::to_string(x) + " (scalar)";
				} else {
					int32_t x = *(int32_t*)t.data_ptr();
					res += std::to_string(x) + " (scalar)";
				}
			}
		} else {
			// Non-scalar tensor
			std::function<void(int, const std::vector<int64_t> &)> print_helper = [&](int cur_dim, const std::vector<int64_t> &pos) {
				//printf("cur_dim=%d, pos.size()=%ld\n", cur_dim, pos.size());
				if (cur_dim == (int)shape.size()) {
					// We have reached the last dimension
					void* ptr = this->get_elem_ptr(pos);
					//LOG_DEBUG("This element might be output");
					if (this->device.device_type == device_t::CPU) {
						if (this->data_type == data_t::FLOAT32) {
							float x = *(float*)ptr;
							res += std::to_string(x);
						} else {
							int32_t x = *(int32_t*)ptr;
							res += std::to_string(x);
						}
					} else {
						Device dst_device = Device::cpu();
						if (this->data_type == data_t::FLOAT32) {
							void* dst_ptr = malloc(sizeof(float));
							memcpy(dst_device, dst_ptr, this->device, ptr, sizeof(float));
							float x = *(float*)dst_ptr;
							res += std::to_string(x);
						} else {
							void* dst_ptr = malloc(sizeof(int32_t));
							memcpy(dst_device, dst_ptr, this->device, ptr, sizeof(int32_t));
							int32_t x = *(int32_t*)dst_ptr;
							res += std::to_string(x);
						}
					} 
					//LOG_DEBUG("Output finished");
				} else {
					// We have not reached the last dimension
					res += "[";
					for (int64_t i = 0; i < shape[cur_dim]; ++i) {
						if (i != 0) {
							res += ", ";
						}
						std::vector<int64_t> new_pos = pos;
						new_pos.push_back(i);
						print_helper(cur_dim + 1, new_pos);
						if (i == lim - 1 && i != shape[cur_dim] - 1) {
							res += ", ...";
							break;
						}
					}
					res += "]";
				}
			};
			print_helper(0, {});
			res += ", shape=[";
			for (int i = 0; i < (int)shape.size(); ++i) {
				res += std::to_string(shape[i]);
				if (i != (int)shape.size() - 1) {
					res += ", ";
				}
			}
			res += "], stride=[";
			for (int i = 0; i < (int)strides.size(); ++i) {
				res += std::to_string(strides[i]);
				if (i != (int)strides.size() - 1) {
					res += ", ";
				}
			}
			res += "]";
		}
		res += ", device=";
		res += device.to_string();
		res += ")";
		return res;
	}

	void Tensor::print(int lim) const {
		printf("%s\n", this->to_string(lim).c_str());
	}

	Tensor Tensor::to(const Device &device) const {
		if (this->device == device) {
			return *this;
		}
		Tensor res = Tensor(shape, device, data_type);
		memcpy(device, res.data_ptr(), this->device, this->data_ptr(), numel() * sizeof(float));
		return res;
	}

	Tensor Tensor::cpu() const {
		return to(Device::cpu());
	}

	Tensor Tensor::cuda(int index) const {
		return to(Device::cuda(index));
	}

	bool Tensor::operator==(const Tensor &other) const {
		return cmp_tensor(*this, other);
	}

	int Tensor::dim() const {
		return shape.size();
	}

}