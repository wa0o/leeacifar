
def leea_pro_run(species_num = 400,gen = 500):
    import leea_pro.function as function
    import leea_net
    import load_sample
    import torch
    import all_function
    import numpy as np
    
    acc = []
    train_data,train_target = load_sample.load_train()
    test_data,test_target = load_sample.load_test()

    with torch.no_grad():
        species = []
        for i in range(species_num):
            species.append(leea_net.individual())
        ones = []
        for i in range(int(gen/100)):
            ones.append(leea_net.individual())
        species = all_function.evaluate(species,train_data[0],train_target[0])
        size = all_function.get_size(species[0])
        index = np.random.randint(len(train_data),size = gen)
        rate = 0.03
        for i in range(gen):
            if i%20 == 0:
                print(i)
                species.sort(key=function.getfitness,reverse=True)
                acc.append(all_function.test(species[0].net,test_data,test_target))
            parent_list = function.get_parent(species,int(len(species)*0.4))
            new_species = function.produce_offspring(species,parent_list,size,rate)
            new_species = all_function.evaluate(new_species,train_data[index[i]],train_target[index[i]])
            species = function.save_sort(species,train_data[index[i]],train_target[index[i]])
            species = function.get_next(new_species,species)
            if i!=0 and i%100==0:
                ones[int(i/100-1)].fitness = species[10].fitness
                species[int(species_num/2)] = ones[int(i/100-1)]
            rate*=0.01**(1/gen)
                
    return acc