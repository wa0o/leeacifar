import torch
import numpy as np

def evaluate(species,inputs,labels):
    with torch.no_grad():
        loss = np.zeros(len(species))
        for i in range(len(species)):
            output = species[i].net(inputs)
            loss[i] = torch.nn.functional.nll_loss(output,labels.long())
        if loss.max()-loss.min() == 0:
            print('The species converge to a individual!!')
            loss = loss*0+0.5
        else:
            loss = 1 - (loss-loss.min())/(loss.max()-loss.min())
        for i in range(len(species)):
            species[i].fitness = species[i].fitness*0.6+loss[i]
        return species
    
def get_size(individual):
    with torch.no_grad():
        size = []
        for net in individual.net.parameters():
            sum = 1
            for i in net.size():
                sum *= i
            size.append(sum)
        return size
    
def test(net,data,target):
    correct = 0
    with torch.no_grad():
        x = np.random.randint(len(data),size=(10))
        for i in x:
            output = net(data[i])
            pred = output.max(1,keepdim = True)[1]
            correct += pred.eq(target[i].view_as(pred)).sum().item()
    accuracy = correct/(10*len(data[0]))
    print('accuracy = ',accuracy)
    return accuracy