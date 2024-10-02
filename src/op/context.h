#pragma once

#include "src/basics/tensor.h"

#include <vector>

namespace MyTorch{
	class OpContext {
	private:
		std::vector<Tensor> back_info;
	public:
		OpContext();
		void push_back(const Tensor& t);
		Tensor pop_back();
	};
}