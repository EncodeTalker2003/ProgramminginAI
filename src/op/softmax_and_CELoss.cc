#include "softmax_and_CELoss.h"
#include "src/backend/cuda/softmax_and_CELoss.h"

namespace MyTorch{
	Tensor softmax_and_CELoss_forward_manual(const std::vector<Tensor> inputs, OpContext &cxt, void *args) {
		Tensor input = inputs[0];
		Tensor truth = inputs[1];
		auto [prob, loss] = MyTorch::Backend::CUDA::softmax_and_CELoss_forward(input, truth);
		cxt.push_back(prob);
		cxt.push_back(truth);
		return loss;
	}

	Tensor softmax_and_CELoss_backward_manual(const Tensor& grad_output, OpContext &cxt) {
		Tensor truth = cxt.pop_back();
		Tensor prob = cxt.pop_back();
		Tensor grad_input = MyTorch::Backend::CUDA::softmax_and_CELoss_backward(grad_output, prob, truth);
		return grad_input;
	}

	Tensor softmax_and_CELoss_forward(const Tensor& input, const Tensor &truth) {
		OpContext cxt;
		return softmax_and_CELoss_forward_manual({input, truth}, cxt, NULL);
	}
}