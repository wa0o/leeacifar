def adam_run(epoch = 5):
    import all_function
    import leea_net
    import load_sample
    import torch
    import numpy as np
    
    acc = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data,train_target = load_sample.load_train()
    test_data,test_target = load_sample.load_test()

    net = leea_net.Net().to(device)

    optimizer = torch.optim.Adam(net.parameters())
    for i in range(epoch):
        print(i)
        index = np.random.randint(len(train_data),size=len(train_data))
        length = len(index)
        for j in range(length):
            if j%20 == 0:
                acc.append(all_function.test(net,test_data,test_target))
            x ,y= train_data[index[j]],train_target[index[j]]
            optimizer.zero_grad()
            output = net(x)  
            loss = torch.nn.functional.nll_loss(output,y.long())
            loss.backward()
            optimizer.step()
            

    return acc


