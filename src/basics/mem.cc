#include "mem.h"
#include "log.h"

#include <string.h>
#include <memory>

namespace MyTorch{

	void* MemData::allocate_data(MyTorch::Device device, size_t length) {
		void* ptr;
		//LOG_DEBUG("Allocating memory on device with size %ld", length);
		if (device.device_type == device_t::CPU) {
			ptr = malloc(length);
			if (ptr == NULL) {
				LOG_FATAL("Failed to allocate memory on CPU");
			}
		} else {
			device.switch_to_this_device();
			cudaError_t err = cudaMalloc(&ptr, length);
			if (err != cudaSuccess) {
				LOG_FATAL("Failed to allocate memory on CUDA device");
			}
		}
		return ptr;
	}

	void MemData::free_data(MyTorch::Device device, void* ptr) {
		if (device.device_type == device_t::CPU) {
			free(ptr);
		} else {
			device.switch_to_this_device();
			cudaError_t err = cudaFree(ptr);
			if (err != cudaSuccess) {
				LOG_FATAL("Failed to free memory on CUDA device");
			}
		}
	}

	MemData::MemData(MyTorch::Device device, size_t length) : device(device), length(length) {
		ptr = allocate_data(device, length);
		refcnt = new size_t(1);
	}

	MemData::MemData(const MemData& other) : device(other.device), ptr(other.ptr), refcnt(other.refcnt), length(other.length) {
		(*refcnt)++;
	}

	MemData::~MemData() {
		(*refcnt)--;
		if (*refcnt == 0) {
			free_data(device, ptr);
			delete refcnt;
		}
	}

	MemData& MemData::operator=(const MemData& other) {
		if (this == &other) {
			return *this;
		}
		(*refcnt)--;
		if (*refcnt == 0) {
			free_data(device, ptr);
			delete refcnt;
		}
		device = other.device;
		ptr = other.ptr;
		refcnt = other.refcnt;
		length = other.length;
		(*refcnt)++;
		return *this;
	}

	void memset(const MyTorch::Device& device, void* ptr, unsigned char value, size_t length) {
		if (device.device_type == device_t::CPU) {
			::memset(ptr, value, length);
		} else {
			//printf("memset on CUDA\n");
			device.switch_to_this_device();
			cudaError_t err = cudaMemset(ptr, value, length);
			if (err != cudaSuccess) {
				LOG_FATAL("Failed to memset memory on CUDA device");
			}
		}
	}

	void memcpy(const MyTorch::Device& dst_device, void* dst_ptr, const MyTorch::Device& src_device, void* src_ptr, size_t length) {
		if (dst_device.device_type == device_t::CPU && src_device.device_type == device_t::CPU) {
			::memcpy(dst_ptr, src_ptr, length);
		} else if (dst_device.device_type == device_t::CPU && src_device.device_type == device_t::CUDA) {
			src_device.switch_to_this_device();
			cudaError_t err = cudaMemcpy(dst_ptr, src_ptr, length, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) {
				LOG_FATAL("Failed to memcpy from CUDA device to CPU");
			}
		} else if (dst_device.device_type == device_t::CUDA && src_device.device_type == device_t::CPU) {
			dst_device.switch_to_this_device();
			cudaError_t err = cudaMemcpy(dst_ptr, src_ptr, length, cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				LOG_FATAL("Failed to memcpy from CPU to CUDA device");
			}
		} else {
			if (dst_device.device_id == src_device.device_id) {
				cudaError_t err = cudaMemcpy(dst_ptr, src_ptr, length, cudaMemcpyDeviceToDevice);
				if (err != cudaSuccess) {
					LOG_FATAL("Failed to memcpy from CUDA device to CUDA device");
				}
			} else {
				LOG_FATAL("Failed to memcpy between different CUDA devices");
			}
		}
	}
}