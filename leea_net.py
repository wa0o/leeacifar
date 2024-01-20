import torch
import torch.nn as nn
import torch.nn.functional as F

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fa1 = nn.Conv2d(1,10,5)
#         self.fa2 = nn.Conv2d(10,20,3)
#         self.fb1 = nn.Linear(20*10*10,500) #(((28-5+1)/2)-3+1)^2)
#         self.fb2 = nn.Linear(500,10)
#     def forward(self,x):
#         in_size = x.size(0)
#         out = self.fa1(x)
#         out = F.relu(out)
#         out = F.max_pool2d(out,2,2)
#         out = self.fa2(out)
#         out = F.relu(out)
#         out = out.view(in_size,-1)
#         out = self.fb1(out)
#         out = F.relu(out)
#         out = self.fb2(out)
#         out = F.log_softmax(out,dim=1)
#         return out


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fa1 = nn.Conv2d(3,16,5)
        self.fa2 = nn.Conv2d(16,32,5)
        self.fa3 = nn.Conv2d(32,32,3)
        self.fb1 = nn.Linear(288,128)
        self.fb2 = nn.Linear(128,10)
    def forward(self,x):
        in_size = x.size(0)
        out = self.fa1(x)
        out = F.relu(out)
        out = F.max_pool2d(out,2,2)
        out = self.fa2(out)
        out = F.relu(out)
        out = F.max_pool2d(out,2,2)
        out = self.fa3(out)
        out = F.relu(out)
        out = out.view(in_size,-1)
        out = self.fb1(out)
        out = torch.sigmoid(out)
        out = self.fb2(out)
        out = F.log_softmax(out,dim=1)
        return out

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(7*7,24) #(((28-5+1)/2)-3+1)^2)
#         self.fc2 = nn.Linear(24,10)
#     def forward(self,x):
#         in_size = x.size(0)
#         x = F.max_pool2d(x,4,4)
#         x = x.view(in_size,-1)
#         out = self.fc1(x)
#         out = F.relu(out)
#         out = self.fc2(out)
#         out = F.log_softmax(out,dim=1)
#         return out

class individual():
    def __init__(self):
        self.net = Net()
        self.net.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.fitness = 0
