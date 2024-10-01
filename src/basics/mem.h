#pragma once

#include "src/basics/device.h"

namespace MyTorch {
	class MemData {
	public:
		static void* allocate_data(MyTorch::Device device, size_t length);
		static void free_data(MyTorch::Device device, void* ptr);

		MyTorch::Device device;

		void* ptr;
		size_t* refcnt; // reference count
		size_t length;

		MemData(MyTorch::Device device, size_t length);
		MemData(const MemData& other);
		~MemData();
		
		MemData& operator=(const MemData& other);
	};

	void memset(const MyTorch::Device& device, void* ptr, unsigned char value, size_t length);
	void memcpy(const MyTorch::Device& dst_device, void* dst_ptr, const MyTorch::Device& src_device, void* src_ptr, size_t length);
}