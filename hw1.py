#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
torch.cuda.is_available()


# ### Hyper-Parameters

# In[19]:


batch_size = 4
lr = 0.001
num_epochs = 10
seed = 42
model_path = './model.ckpt' 


# ### Load Data

# In[11]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[12]:


device=torch.device('cuda:0')


# ### Define Network

# In[21]:


class LeNet(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(3, 6, 5)
		self.pool = torch.nn.MaxPool2d(2, 2)
		self.conv2 = torch.nn.Conv2d(6, 16, 5)
		self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
		self.fc2 = torch.nn.Linear(120, 84)
		self.fc3 = torch.nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
	
net = LeNet().to(device)


# In[22]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)


# ### Train Data

# In[23]:


loss_data = []
acc_data = []

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    train_acc, train_loss = 0.0, 0.0
    total = 0
    net.train()
    for i, data in tqdm(enumerate(trainloader, 0)):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_acc += (predicted == labels).sum().item()
        total += labels.size(0)
        if i % 2500 == 2499:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, train_loss / 1000))
            # train_loss = 0.0
    
    loss_data.append(train_loss)
    acc_data.append(train_acc / total)
    

print('Finished Training')
print('loss_data = ', loss_data)
print('acc_data = ', acc_data)


# In[24]:


x_train_loss = range(num_epochs)
y_train_loss = loss_data
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle='solid', label='train loss')
plt.legend()
plt.show()


# In[25]:


torch.save(net.state_dict(), model_path)


# ### Test Data

# In[26]:


correct, total = 0, 0
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
with torch.no_grad():
	for data in testloader:
		images, labels = data[0].to(device), data[1].to(device)
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
		for label, prediction in zip(labels, predicted):
			if label == prediction:
				correct_pred[classes[label]] += 1
			total_pred[classes[label]] += 1

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

for classname, correct_count in correct_pred.items():
	accuracy = 100 * float(correct_count) / total_pred[classname]
	print("Accuracy for class {:5s} is: {:.1f} %, the total testcases for this class is {:}".format(classname, accuracy, total_pred[classname]))

