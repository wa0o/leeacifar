import torch
import torchvision.transforms as transforms

import torchvision
    
def load_train():

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset = trainset ,batch_size= batch_size ,shuffle= True)
    train_data = torch.ones([len(train_loader),batch_size,3,32,32])
    train_target = torch.ones([len(train_loader),batch_size])
    i = 0
    for data,target in train_loader:
        if i < len(train_loader)-1:
            train_data[i] = data
            train_target[i] = target
        i+=1
    train_data = train_data.to(device)
    train_target = train_target.to(device)
    return train_data,train_target
    
    
def load_test():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset = testset ,batch_size= batch_size ,shuffle= True)
    test_data = torch.ones([len(test_loader),batch_size,3,32,32])
    test_target = torch.ones([len(test_loader),batch_size])
    i = 0
    for data,target in test_loader:
        if i<len(test_loader)-1:
            test_data[i] = data
            test_target[i] = target
        i += 1
    test_data = test_data.to(device)
    test_target = test_target.to(device)
    return test_data,test_target
