#include "device.h"
#include "log.h"

namespace MyTorch {

	Device::Device(device_t type, int id) {
		device_type = type;
		device_id = id;
		if (device_type == device_t::CUDA) {
			cudaError_t err = cudaSetDevice(device_id);
			if (err != cudaSuccess) {
				LOG_FATAL("Failed to set device %d", device_id);
			}
			/*err = cudaStreamCreate(&stream);
			if (err != cudaSuccess) {
				LOG_FATAL("Failed to create stream for device %d", device_id);
			}*/
		}
	}

	std::string Device::to_string() const {
		if (device_type == device_t::CPU) {
			return "CPU";
		} else {
			return "CUDA:" + std::to_string(device_id);
		}
	}

	void Device::switch_to_this_device() const {
		if (device_type == device_t::CUDA) {
			cudaError_t err = cudaSetDevice(device_id);
			if (err != cudaSuccess) {
				LOG_FATAL("Failed to set device %d", device_id);
			}
		}
	}

	Device Device::cpu() {
		return Device(device_t::CPU, 0);
	}

	Device Device::cuda(int id) {
		return Device(device_t::CUDA, id);
	}
}