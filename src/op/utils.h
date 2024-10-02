#pragma once

#include "src/basics/device.h"
#include "src/basics/log.h"

namespace MyTorch{
	#define DISPATCH_TO_BACKEND(device_type, call) \
	[&]() { \
		switch (device_type) { \
			case device_t::CPU: \
				return MyTorch::Backend::CPU::call; \
			case device_t::CUDA: \
				return MyTorch::Backend::CUDA::call; \
			default: \
				LOG_FATAL("Unknown device."); \
		} \
	}()
}