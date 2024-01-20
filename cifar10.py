import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 100

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
# 使用torch.utils.data.DataLoader构建训练数据集的DataLoader
trainloader = torch.utils.data.DataLoader(dataset = trainset ,batch_size= batch_size ,shuffle= True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
# 使用torch.utils.data.DataLoader构建测试数据集的DataLoader
testloader = torch.utils.data.DataLoader(dataset = testset ,batch_size= batch_size ,shuffle= True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)




class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2,),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2,),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear( 4*4*64 , 128, bias= True),
            nn.Sigmoid(),
            nn.Linear(128,10,bias=True),
            nn.Softmax()
        )
        

    def forward(self, x):
        x = self.model(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net()
net = net.to(device)
# 请参考 https://pytorch.org/docs/stable/nn.html#loss-functions 选择合适的损失函数完善下行代码
criterion = nn.CrossEntropyLoss()
# 请参考 https://pytorch.org/docs/stable/optim.html#algorithms 选择合适的优化器完善下行代码
optimizer = torch.optim.Adam(net.parameters())

# 请选择合适的epoch次数
n_epoch = 5
for epoch in range(n_epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # 使用net获得网络在输入inputs时的输出
        outputs = net(inputs)
        # 使用criterion获得网络此时的损失并计算误差梯度
        loss = criterion(outputs,labels)

        # 使用optimizer进行一次优化
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = next(dataiter)

net = Net()
net.load_state_dict(torch.load(PATH))


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
