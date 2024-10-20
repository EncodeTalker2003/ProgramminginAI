#pragma once

#include "src/basics/tensor.h"

#include <vector>

namespace MyTorch{
	class OpContext {
	private:
		std::vector<Tensor> back_info;
		int64_t pool_size;
	public:
		OpContext();
		void push_back(const Tensor& t);
		Tensor pop_back();
		int64_t get_pool_size();
		void save_pool_size(int64_t pool_size);
	};
}