#include "src/basics/tensor.h"
#include "src/op/mat_mul.h"
#include "src/op/convolution.h"
#include "src/op/pooling.h"
#include "src/op/softmax_and_CELoss.h"

#include <vector>

using MyTorch::Tensor, MyTorch::Device;

void test1() {
	printf("Test 1: test matrix multiplication\n");
	Tensor a = Tensor::from_data({
		1., 2., 3.,
		4., 5., 6.
	}, MyTorch::data_t::FLOAT32, {2, 3}, Device::cuda(0));
	Tensor b = Tensor::from_data({
		1., 2., 
		3., 4., 
		5., 6.
	}, MyTorch::data_t::FLOAT32, {3, 2}, Device::cuda(0));
	Tensor c = Tensor::from_data({
		22., 28.,
		49., 64.
	}, MyTorch::data_t::FLOAT32, {2, 2}, Device::cuda(0));
	MyTorch::OpContext cxt;
	printf("a:\n");
	a.print();
	printf("b:\n");
	b.print();
	Tensor my_output = MyTorch::matmul_forward_manual({a, b}, cxt, NULL);
	printf("a @ b:\n");
	my_output.print();
	Tensor grad_output = Tensor::from_data({
		1., 2.,
		3., 4.
	}, MyTorch::data_t::FLOAT32, {2, 2}, Device::cuda(0));
	auto [grad_a, grad_b] = MyTorch::matmul_backward_manual(grad_output, cxt);
	printf("grad_a:\n");
	grad_a.print();
	printf("grad_b:\n");
	grad_b.print();
}

void test2() {
	printf("Convolution tests\n");
	Tensor input_img = Tensor::from_data({
		0.9022, 0.6055, 0.8799, 0.3339, 0.5612,
		0.2777, 0.2743, 0.7656, 0.0810, 0.5094,
		0.3722, 0.7272, 0.0389, 0.9295, 0.4053,
		0.6124, 0.4697, 0.3102, 0.3041, 0.1790,
		0.5564, 0.6481, 0.7426, 0.2569, 0.6979,

		0.0636, 0.3710, 0.3374, 0.7621, 0.0148,
		0.2665, 0.5692, 0.3018, 0.8880, 0.9180,
		0.8530, 0.1325, 0.0435, 0.3055, 0.4114,
		0.0970, 0.3121, 0.5835, 0.9967, 0.5118,
		0.5507, 0.8006, 0.8614, 0.6140, 0.1617
	}, MyTorch::data_t::FLOAT32, {1, 2, 5, 5}, Device::cuda());
	Tensor kernel = Tensor::from_data({
		0.1666, 0.0232, 0.0465,
		0.1919, 0.3024, 0.0052,
		0.1905, 0.0770, 0.6483,

		0.3007, 0.5991, 0.1295,
		0.4960, 0.7857, 0.2054,
		0.9429, 0.4061, 0.8264,

		0.9828, 0.5343, 0.4301,
		0.7997, 0.7347, 0.6834,
		0.8784, 0.3457, 0.6518,

		0.0270, 0.1614, 0.8181,
		0.6542, 0.5209, 0.2208,
		0.9032, 0.7265, 0.2788
	}, MyTorch::data_t::FLOAT32, {2, 2, 3, 3}, Device::cuda());

	Tensor conv_output_grad = Tensor::from_data({
		0.4389, 0.6895, 0.6377, 0.4156, 0.8253,
		0.0522, 0.6640, 0.5184, 0.6048, 0.4048,
		0.4524, 0.6625, 0.4824, 0.2167, 0.0864,
		0.8571, 0.3407, 0.6353, 0.1914, 0.7777,
		0.4566, 0.2790, 0.9681, 0.7275, 0.9138,

		0.6307, 0.0459, 0.7680, 0.0473, 0.2508,
		0.5143, 0.0483, 0.0309, 0.8686, 0.7356,
		0.8688, 0.2989, 0.4625, 0.6204, 0.2190,
		0.5284, 0.7636, 0.3465, 0.0790, 0.9961,
		0.1740, 0.1176, 0.1962, 0.9571, 0.3933
	}, MyTorch::data_t::FLOAT32, {1, 2, 5, 5}, Device::cuda());

	printf("Input image:\n");
	input_img.print();
	printf("Kernel:\n");
	kernel.print();
	printf("Conv output grad:\n");
	conv_output_grad.print();

	Tensor conv_output_reference = Tensor::from_data({
		1.1800, 2.0554, 2.5465, 2.9281, 1.8882,
		1.5029, 2.3181, 2.6748, 2.7972, 2.3014,
		1.7153, 2.4305, 2.5184, 3.1080, 2.8374,
		2.2469, 3.2901, 3.2335, 3.4352, 2.2588,
		0.9033, 1.8048, 2.2224, 2.0726, 1.3531,

		1.8188, 3.6532, 3.4849, 4.1597, 2.9018,
		2.9715, 4.8931, 4.9541, 4.3069, 3.6805,
		2.6944, 3.9694, 4.6279, 5.0748, 3.5802,
		2.8861, 4.7688, 5.4489, 5.1691, 3.6370,
		2.1155, 3.9127, 4.0236, 3.3183, 1.7081
	}, MyTorch::data_t::FLOAT32, {1, 2, 5, 5}, Device::cuda());

	MyTorch::OpContext cxt;
	Tensor conv_output = MyTorch::conv_forward_manual({input_img, kernel}, cxt, NULL);
	printf("Conv output:\n");
	conv_output.print();
	auto [input_grad, kernel_grad] = MyTorch::conv_backward_manual(conv_output_grad, cxt);
	printf("Input grad:\n");
	input_grad.print();
	printf("Kernel grad:\n");
	kernel_grad.print();
}

void test3() {
	printf("Test3: Pooling tests\n");
	Tensor input_img = Tensor::from_data({
		0.9022, 0.6055, 0.8799, 0.3339, 0.5612, 0.1452,
		0.2777, 0.2743, 0.7656, 0.0810, 0.5094, 0.6192,
		0.3722, 0.7272, 0.0389, 0.9295, 0.4053, 0.1964,
		0.6124, 0.4697, 0.3102, 0.3041, 0.1790, 0.9182,
		0.5564, 0.6481, 0.7426, 0.2569, 0.6979, 0.1145,
		0.7824, 0.1550, 0.9883, 0.0562, 0.8542, 0.5938,

		0.0636, 0.3710, 0.3374, 0.7621, 0.0148, 0.2333,
		0.2665, 0.5692, 0.3018, 0.8880, 0.9180, 0.1565,
		0.8530, 0.1325, 0.0435, 0.3055, 0.4114, 0.4691,
		0.0970, 0.3121, 0.5835, 0.9967, 0.5118, 0.2591,
		0.5507, 0.8006, 0.8614, 0.6140, 0.1617, 0.1953,
		0.7542, 0.8370, 0.7664, 0.3321, 0.1819, 0.4984
	}, MyTorch::data_t::FLOAT32, {1, 2, 6, 6}, Device::cuda());
	Tensor output_grad = Tensor::from_data({
		0.4389, 0.6895, 0.6377,
		0.4156, 0.8253, 0.0522,
		0.6640, 0.5184, 0.6048,

		0.4048, 0.4524, 0.6625,
		0.4824, 0.2167, 0.0864,
		0.8571, 0.3407, 0.6353
	}, MyTorch::data_t::FLOAT32, {1, 2, 3, 3}, Device::cuda());

	Tensor input_grad_reference = Tensor::from_data({
		0.4389, 0.0000, 0.6895, 0.0000, 0.0000, 0.0000,
		0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6377,
		0.0000, 0.4156, 0.0000, 0.8253, 0.0000, 0.0000,
		0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0522,
		0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
		0.6640, 0.0000, 0.5184, 0.0000, 0.6048, 0.0000,

		0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
		0.0000, 0.4048, 0.0000, 0.4524, 0.6625, 0.0000,
		0.4824, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
		0.0000, 0.0000, 0.0000, 0.2167, 0.0864, 0.0000,
		0.0000, 0.0000, 0.3407, 0.0000, 0.0000, 0.0000,
		0.0000, 0.8571, 0.0000, 0.0000, 0.0000, 0.6353,
	}, MyTorch::data_t::FLOAT32, {1, 2, 6, 6}, Device::cuda());
	Tensor output_reference = Tensor::from_data({
		0.9022, 0.8799, 0.6192,
		0.7272, 0.9295, 0.9182,
		0.7824, 0.9883, 0.8542,

		0.5692, 0.8880, 0.9180,
		0.8530, 0.9967, 0.5118,
		0.8370, 0.8614, 0.4984
	}, MyTorch::data_t::FLOAT32, {1, 2, 3, 3}, Device::cuda());

	printf("Input image:\n");
	input_img.print();
	printf("Output grad:\n");
	output_grad.print();

	MyTorch::OpContext ctx;
	MyTorch::Pooling_args args;
	args.pool_size = 2;
	Tensor output = MyTorch::pooling_forward_manual({input_img}, ctx, &args);
	printf("Output:\n");
	output.print();
	Tensor input_grad = MyTorch::pooling_backward_manual(output_grad, ctx);
	printf("Input grad:\n");
	input_grad.print();
}
	
void test4() {
	printf("Test4: Softmax and cross entropy loss tests\n");
	Tensor input = Tensor::from_data({
		0.7815, 0.5220, 0.9146, 0.2101, 0.9586, 0.2019, 0.6881, 0.5623, 0.8250,
		0.1748, 0.0321, 0.5737, 0.5583, 0.0592, 0.9469, 0.2538, 0.9478, 0.7686,
		0.7276, 0.2412, 0.0822, 0.3260, 0.6034, 0.2331, 0.4781, 0.8598, 0.9691,
		0.9732, 0.1825, 0.0688, 0.9939, 0.5160, 0.5248, 0.3011, 0.3472, 0.1683,
		0.8340, 0.9477, 0.1363, 0.2660, 0.4789, 0.0692, 0.9481, 0.3476, 0.9933,
		0.1266, 0.2456, 0.7972, 0.6936, 0.0731, 0.4410, 0.8412, 0.6265, 0.5222,
		0.9257, 0.9962, 0.9305, 0.9696, 0.4161, 0.8328, 0.2192, 0.7043, 0.8733,
		0.6365, 0.0589, 0.8914, 0.0487, 0.5470, 0.7478, 0.6194, 0.9568, 0.9439,
		0.9366, 0.3571, 0.0724, 0.8578, 0.7720, 0.3828, 0.7801, 0.0545, 0.6946,
		0.9337, 0.1366, 0.0075, 0.6953, 0.2300, 0.2982, 0.5964, 0.8812, 0.2708
	}, MyTorch::data_t::FLOAT32, {10, 9}, Device::cuda());
	Tensor ground_truth = Tensor::from_int_data({
		1, 8, 2, 6, 0, 8, 1, 7, 3, 5
	}, MyTorch::data_t::INT32, {10},  Device::cuda());
	Tensor output_grad = Tensor::from_data({
		0.3834, 0.0781, 0.2414, 0.7319, 0.7889, 0.2854, 0.7927, 0.7831, 0.4025, 0.8183
	}, MyTorch::data_t::FLOAT32, {10}, Device::cuda());

	printf("Input:\n");
	input.print();
	printf("Ground truth:\n");
	ground_truth.print();
	printf("Output grad:\n");
	output_grad.print();

	Tensor loss_reference = Tensor::from_data({
		2.3378, 1.9665, 2.6597, 2.4026, 1.9823, 2.1951, 1.9936, 1.8945, 1.9324, 2.3997
	}, MyTorch::data_t::FLOAT32, {10}, Device::cuda());
	Tensor input_grad_reference = Tensor::from_data({
		0.0480, -0.3464,  0.0548,  0.0271,  0.0573,  0.0269,  0.0437,  0.0385, 0.0501,
		0.0060,  0.0052,  0.0090,  0.0089,  0.0054,  0.0131,  0.0065,  0.0131, -0.0672,
		0.0322,  0.0198, -0.2245,  0.0216,  0.0284,  0.0196,  0.0251,  0.0368, 0.0410,
		0.1297,  0.0588,  0.0525,  0.1324,  0.0821,  0.0828, -0.6657,  0.0693, 0.0580,
		-0.6802,  0.1218,  0.0541,  0.0616,  0.0762,  0.0506,  0.1218,  0.0668, 0.1274,
		0.0214,  0.0241,  0.0418,  0.0377,  0.0203,  0.0293,  0.0437,  0.0353, -0.2536,
		0.1006, -0.6847,  0.1011,  0.1051,  0.0604,  0.0917,  0.0496,  0.0806, 0.0955,
		0.0855,  0.0480,  0.1103,  0.0475,  0.0782,  0.0956,  0.0840, -0.6653, 0.1163,
		0.0631,  0.0353,  0.0266, -0.3442,  0.0535,  0.0362,  0.0539,  0.0261, 0.0495,
		0.1402,  0.0632,  0.0555,  0.1105,  0.0694, -0.7440,  0.1001,  0.1330, 0.0722
	}, MyTorch::data_t::FLOAT32, {10, 9}, Device::cuda());

	MyTorch::OpContext cxt;
	Tensor loss = MyTorch::softmax_and_CELoss_forward_manual({input, ground_truth}, cxt, NULL);
	printf("Loss:\n");
	loss.print();

	Tensor input_grad = MyTorch::softmax_and_CELoss_backward_manual(output_grad, cxt);
	printf("Input grad:\n");
	input_grad.print();
}

int main() {
	test1();
	test2();
	test3();
	test4();
	return 0;
}