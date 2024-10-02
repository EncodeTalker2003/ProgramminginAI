#include "src/basics/tensor.h"
#include "src/op/relu.h"
#include "src/op/sigmoid.h"

#include <vector>

using MyTorch::Tensor, MyTorch::Device;

void test1() {
	printf("Test 1: create tensors on CPU\n");
	Tensor tensor1 = Tensor::zeros({2, 3, 4}, Device::cpu());
	Tensor tensor2 = Tensor::randu({4, 3, 2}, Device::cpu(), -1.0, 1.0);
	Tensor tensor3 = Tensor::from_data({
		1., 2., 3., 4., 5.,
		-1., -2., -3., -4., -5.
	}, {2, 5}, Device::cpu());
	tensor1.print();
	tensor2.print();
	tensor3.print();
}

void test2() {
	printf("Test 2: create tensors on CUDA\n");
	Tensor tensor1 = Tensor::zeros({2, 3, 4}, Device::cuda(0));
	//printf("Successfully created tensor1\n");
	Tensor tensor2 = Tensor::randu({4, 3, 2}, Device::cuda(0), -1.0, 1.0);
	Tensor tensor3 = Tensor::from_data({
		1., 2., 3., 4., 5.,
		-1., -2., -3., -4., -5.
	}, {2, 5}, Device::cuda(0));
	tensor1.print();
	tensor2.print();
	tensor3.print();
}

void test3() {
	printf("Test 3: cpu() and cuda() for a tensor\n");
	Tensor tensor1 = Tensor::from_data({
		1., 2., 3., 4., 5.,
		-1., -2., -3., -4., -5.
	}, {2, 5}, Device::cuda(0));
	tensor1.print();
	Tensor tensor2 = tensor1.cpu();
	tensor2.print();
	Tensor tensor3 = tensor1.cuda(0).cpu().cuda(0);
	tensor3.print();
}

void test4() {
	printf("Test 4: ReLU on cpu\n");
	Tensor input = Tensor::from_data({
		1., 2., 3., 4., 5.,
		-1., -2., -3., -4., -5.
	}, {2, 5}, Device::cpu());

	Tensor grad_output = Tensor::from_data({
		-1., 2., -3., 4., -5.,
		1., -2., 3., -4., 5.
	}, {2, 5}, Device::cpu());

	Tensor output = Tensor::from_data({
		1., 2., 3., 4., 5.,
		0., 0., 0., 0., 0.
	}, {2, 5}, Device::cpu());

	Tensor grad_input = Tensor::from_data({
		-1., 2., -3., 4., -5.,
		0., 0., 0., 0., 0.
	}, {2, 5}, Device::cpu());

	MyTorch::OpContext cxt;
	Tensor my_output = MyTorch::relu_forward_manual(input, cxt);
	printf("Input tensor is\n");
	input.print();
	printf("Output tensor is\n");
	my_output.print();
	Tensor my_grad_input = MyTorch::relu_backward_manual(grad_output, cxt);
	printf("Output Gradient is\n");
	grad_output.print();
	printf("Input Gradient is\n");
	my_grad_input.print();
}

void test5() {
	printf("Test 5: ReLU on cuda\n");
	Tensor input = Tensor::from_data({
		1., 2., 3., 4., 5.,
		-1., -2., -3., -4., -5.
	}, {2, 5}, Device::cuda(0));

	Tensor grad_output = Tensor::from_data({
		-1., 2., -3., 4., -5.,
		1., -2., 3., -4., 5.
	}, {2, 5}, Device::cuda(0));

	Tensor output = Tensor::from_data({
		1., 2., 3., 4., 5.,
		0., 0., 0., 0., 0.
	}, {2, 5}, Device::cuda(0));

	Tensor grad_input = Tensor::from_data({
		-1., 2., -3., 4., -5.,
		0., 0., 0., 0., 0.
	}, {2, 5}, Device::cuda(0));

	MyTorch::OpContext cxt;
	Tensor my_output = MyTorch::relu_forward_manual(input, cxt);
	printf("Input tensor is\n");
	input.print();
	printf("Output tensor is\n");
	my_output.print();
	Tensor my_grad_input = MyTorch::relu_backward_manual(grad_output, cxt);
	printf("Output Gradient is\n");
	grad_output.print();
	printf("Input Gradient is\n");
	my_grad_input.print();
}

void test6() {
	printf("Test 6: sigmoid on cpu\n");
	Tensor input = Tensor::from_data({
		1., 2., 3., 4., 5.,
		-1., -2., -3., -4., -5.
	}, {2, 5}, Device::cpu());

	Tensor grad_output = Tensor::from_data({
		-1., 2., -3., 4., -5.,
		1., -2., 3., -4., 5.
	}, {2, 5}, Device::cpu());

	Tensor output = Tensor::from_data({
		0.7310585786300049, 0.8807970779778823, 0.9525741268224334, 0.9820137900379085, 0.9933071490757153,
		0.2689414213699951, 0.11920292202211755, 0.04742587317756678, 0.01798620996209156, 0.0066928509242848554
	}, {2, 5}, Device::cpu());

	Tensor grad_input = Tensor::from_data({
		-0.19661193324148185, 0.20998717080701323, -0.13552997919273602, 0.07065082485316443, -0.03324028335395016, 
		0.19661193324148185, -0.209987170807013, 0.1355299791927364, -0.07065082485316447, 0.033240283353950774
	}, {2, 5}, Device::cpu());

	MyTorch::OpContext cxt;
	Tensor my_output = MyTorch::sigmoid_forward_manual(input, cxt);
	printf("Input tensor is\n");
	input.print();
	printf("Output tensor is\n");
	my_output.print();
	Tensor my_grad_input = MyTorch::sigmoid_backward_manual(grad_output, cxt);
	printf("Output Gradient is\n");
	grad_output.print();
	printf("Input Gradient is\n");
	my_grad_input.print();
}

void test7() {
	printf("Test 6: sigmoid on cpu\n");
	Tensor input = Tensor::from_data({
		1., 2., 3., 4., 5.,
		-1., -2., -3., -4., -5.
	}, {2, 5}, Device::cuda(0));

	Tensor grad_output = Tensor::from_data({
		-1., 2., -3., 4., -5.,
		1., -2., 3., -4., 5.
	}, {2, 5}, Device::cuda(0));

	Tensor output = Tensor::from_data({
		0.7310585786300049, 0.8807970779778823, 0.9525741268224334, 0.9820137900379085, 0.9933071490757153,
		0.2689414213699951, 0.11920292202211755, 0.04742587317756678, 0.01798620996209156, 0.0066928509242848554
	}, {2, 5}, Device::cuda(0));

	Tensor grad_input = Tensor::from_data({
		-0.19661193324148185, 0.20998717080701323, -0.13552997919273602, 0.07065082485316443, -0.03324028335395016, 
		0.19661193324148185, -0.209987170807013, 0.1355299791927364, -0.07065082485316447, 0.033240283353950774
	}, {2, 5}, Device::cuda(0));

	MyTorch::OpContext cxt;
	Tensor my_output = MyTorch::sigmoid_forward_manual(input, cxt);
	printf("Input tensor is\n");
	input.print();
	printf("Output tensor is\n");
	my_output.print();
	Tensor my_grad_input = MyTorch::sigmoid_backward_manual(grad_output, cxt);
	printf("Output Gradient is\n");
	grad_output.print();
	printf("Input Gradient is\n");
	my_grad_input.print();
}

int main() {
	test1();
	test2();
	test3();
	test4();
	test5();
	test6();
	test7();
}