#pragma once

#include <string>

#include <cuda_runtime_api.h>


namespace MyTorch {

	enum class device_t {
		CPU,
		CUDA
	};

	class Device {

	public: 

		device_t device_type;
		int device_id;
		//cudaStream_t stream;

		Device(device_t type, int id);

		std::string to_string() const;

		void switch_to_this_device() const;

		static Device cpu();
		static Device cuda(int id = 0);

		inline bool operator==(const Device &other) const {
			return device_type == other.device_type && device_id == other.device_id;
		}

		inline bool operator!=(const Device &other) const {
			return !(*this == other);
		}

	};
}