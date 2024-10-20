#include "context.h"
#include "src/basics/log.h"

namespace MyTorch{
	OpContext::OpContext() {
		back_info.clear();
	}

	void OpContext::push_back(const Tensor& t) {
		back_info.push_back(t);
	}

	Tensor OpContext::pop_back() {
		if (back_info.empty()) {
			LOG_FATAL("Trying to pop from empty OpContext");
		}
		Tensor res = back_info.back();
		back_info.pop_back();
		return res;
	}

	int64_t OpContext::get_pool_size() {
		return pool_size;
	}

	void OpContext::save_pool_size(int64_t pool_size) {
		this->pool_size = pool_size;
	}
}