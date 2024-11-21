import sys
import random

sys.path.append("/mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/src/pybind")

import MyTorch
import torch
from torchvision import datasets, transforms

def gen_list(shape):
	return torch.randn(shape).tolist()

def test_mat_mul():
	print("Test for matrix multiplication")
	shape1 = [2, 3]
	shape2 = [3, 2]
	input1_list = gen_list(shape1)
	input2_list = gen_list(shape2)
	print("type of shape1 is ", type(shape1))
	input1 = MyTorch.Tensor(input1_list, MyTorch.data_t.float32, MyTorch.Device.cuda(0))
	input2 = MyTorch.Tensor(input2_list, MyTorch.data_t.float32, MyTorch.Device.cuda(0))
	output = MyTorch.op.matmul_forward(input1, input2)
	input1_torch = torch.tensor(input1_list)
	input2_torch = torch.tensor(input2_list)
	output_torch = torch.matmul(input1_torch, input2_torch)

	input1.print()
	input2.print()
	output.print()
	print("output_torch is ", output_torch)

def test_relu():
	print("Test for ReLU")
	shape = [2, 3]
	input_list = gen_list(shape)
	input = MyTorch.Tensor(input_list, MyTorch.data_t.float32, MyTorch.Device.cuda(0))
	output = MyTorch.op.relu_forward(input)
	input_torch = torch.tensor(input_list)
	output_torch = torch.relu(input_torch)

	input.print()
	output.print()
	print("output_torch is ", output_torch)

def test_sigmoid():
	print("Test for Sigmoid")
	shape = [2, 3]
	input_list = gen_list(shape)
	input = MyTorch.Tensor(input_list, MyTorch.data_t.float32, MyTorch.Device.cuda(0))
	output = MyTorch.op.sigmoid_forward(input)
	input_torch = torch.tensor(input_list)
	output_torch = torch.sigmoid(input_torch)

	input.print()
	output.print()
	print("output_torch is ", output_torch)

def test_conv():
	print("Test for Convolution")
	input_shape = [2, 3, 4, 4]
	kernel_shape = [2, 3, 3, 3]
	input_list = gen_list(input_shape)
	kernel_list = gen_list(kernel_shape)
	input = MyTorch.Tensor(input_list, MyTorch.data_t.float32, MyTorch.Device.cuda(0))
	kernel = MyTorch.Tensor(kernel_list, MyTorch.data_t.float32, MyTorch.Device.cuda(0))
	output = MyTorch.op.conv_forward(input, kernel)
	input_torch = torch.tensor(input_list)
	kernel_torch = torch.tensor(kernel_list)
	output_torch = torch.nn.functional.conv2d(input_torch, kernel_torch, padding='same')

	input.print()
	kernel.print()
	output.print()
	print("output_torch is ", output_torch)
	print("output_torch shape is ", output_torch.shape)

def test_pooling():
	print("Test for Pooling")
	input_shape = [2, 3, 4, 4]
	input_list = gen_list(input_shape)
	input = MyTorch.Tensor(input_list, MyTorch.data_t.float32, MyTorch.Device.cuda(0))
	output = MyTorch.op.pooling_forward(input, 2)
	input_torch = torch.tensor(input_list)
	output_torch = torch.nn.functional.max_pool2d(input_torch, kernel_size=2, stride=2)

	input.print()
	output.print()
	print("output_torch is ", output_torch)
	print("output_torch shape is ", output_torch.shape)

def test_softmax_and_CELoss():
	print("Test for Softmax and CELoss")
	input_shape = [10, 3]
	input_list = gen_list(input_shape)
	label_list = [random.randint(0, 2) for i in range(10)]
	input = MyTorch.Tensor(input_list, MyTorch.data_t.float32, MyTorch.Device.cuda(0))
	label = MyTorch.Tensor(label_list, MyTorch.data_t.int32, MyTorch.Device.cuda(0))
	output = MyTorch.op.softmax_and_CELoss_forward(input, label)
	# sum_output = sum(output.tolist())

	input_torch = torch.tensor(input_list)
	label_torch = torch.tensor(label_list)
	loss = torch.nn.functional.cross_entropy(input_torch, label_torch)
	print("output is ", output)
	# print("sum_output is ", sum_output)
	print("loss is ", loss)

def py2my(py_tensor):
	data = py_tensor.tolist()
	return MyTorch.Tensor(data, MyTorch.data_t.float32, MyTorch.Device.cuda(0))

def read_MNIST_data():
	print("Loading MNIST data...")
	train_set = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	]))
	test_set = datasets.MNIST('./data', train=False, transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	]))
	training_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
	testing_data = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

	train_data_mine, test_data_mine = [], []
	for data, target in training_data:
		train_data_mine.append((py2my(data), py2my(target)))
	
	for data, target in testing_data:
		test_data_mine.append((py2my(data), py2my(target)))
	print(train_data_mine[0][0].shape)
	print("MNIST data loaded.")
	

if __name__ == "__main__":
	print("Hello, MyTorch!")
	test_mat_mul()
	test_relu()
	test_sigmoid()
	test_conv()
	test_pooling()
	test_softmax_and_CELoss()
	read_MNIST_data()

