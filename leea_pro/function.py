import torch
import numpy as np
import copy
import leea_net
import all_function


def getfitness(individual):
    return individual.fitness

def get_parent(species,parent_num):
    species.sort(key=getfitness,reverse=True)
    probabilities = np.zeros(parent_num)+1/parent_num
    index_list = np.arange(0,parent_num,1)
    parent_list = []
    i = 0
    while i < len(species):
        index1 = np.random.choice(index_list, p = probabilities.ravel())
        index2 = np.random.choice(index_list, p = probabilities.ravel())
        if index1 != index2:
            i+=1
            parent_list.append((index1,index2))
    return parent_list


def sexual1(parent1,parent2):
    with torch.no_grad():
        child = copy.deepcopy(parent1)
        child.fitness = (parent1.fitness**2+parent2.fitness**2)/(parent1.fitness+parent2.fitness)
        rate = parent1.fitness/(parent1.fitness+parent2.fitness)-0.5
        for name,weight in child.net.named_parameters():
            tparent = getattr(getattr(parent2.net,name[0:3]),name[4:])
            loc = torch.rand(size=weight.size(),device=0)+rate
            loc = loc.round()
            weight.data = weight.data*loc+tparent*(1-loc)
        return child
    
def sexual2(parent1,parent2):
    with torch.no_grad():
        child = copy.deepcopy(parent1)
        child.fitness = (parent1.fitness+parent2.fitness)/2
        for name,weight in child.net.named_parameters():
            tparent = getattr(getattr(parent2.net,name[0:3]),name[4:])
            weight.data = (weight.data+tparent.data)*(np.random.randn()+1)/2
        return child
        
def ansexual(parent,size,changenum=10):
    child = copy.deepcopy(parent)
    change = np.random.randint(len(size),size=int(changenum))
    for layernum in change:
        layer = 0
        for weight in child.net.parameters():
            if layer == layernum:
                location = ()
                for i in range(len(weight.size())):
                    location += (np.random.randint(weight.size()[i]),)
                weight.data[location] = np.random.randn()
            layer += 1
    return child        

def ansexual1(parent,rate):
    child = copy.deepcopy(parent)
    for weight in child.net.parameters():
        loc = torch.rand(size=weight.size(),device=0)+0.45
        loc = loc.round()
        power = torch.rand(size=weight.size(),device=0)*rate*2-rate
        weight.data += (1-loc)*power
    return child

def ansexual2(parent1):
    with torch.no_grad():
        child = copy.deepcopy(parent1)
        parent2 = leea_net.individual()
        for name,weight in child.net.named_parameters():
            tparent = getattr(getattr(parent2.net,name[0:3]),name[4:])
            loc = torch.randint(2,size=weight.size(),device=0)
            weight.data = weight.data*loc+tparent*(1-loc)
        return child


def produce_offspring(species,parent_list,size,rate):
    new_species = []
    sex_num = np.random.randint(0,100,size=(len(parent_list)))
    for i in range(len(parent_list)):
        if sex_num[i]>90:
            new_species.append(ansexual1(species[parent_list[i][0]],rate))
        elif sex_num[i]>80:
            new_species.append(sexual2(species[parent_list[i][0]],species[parent_list[i][1]]))
        else:
            new_species.append(sexual1(species[parent_list[i][0]],species[parent_list[i][1]]))
    return new_species

def save_sort(species,input,output):
    species.sort(key=getfitness,reverse=True)
    species = all_function.evaluate(species[:int(len(species)*0.2)],input,output)
    return species


def get_next(species1,species2):
    species = species1+species2
    species.sort(key=getfitness,reverse=True)
    return species[:int(len(species1))]
