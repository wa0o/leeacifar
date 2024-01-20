import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

def help(id):
    print(id)
    if id == 1:
        f = open('result\leea.pkl','rb')
    if id == 2:
        f = open('result\leeapro.pkl','rb')
    if id == 3:
        f = open('result\Adam.pkl','rb')
    if id == 4:
        f = open('result\leeapro1.pkl','rb')
    if id == 5:
        f = open('result\leeapro2.pkl','rb')
    if id == 6:
        f = open('result\leeapro3.pkl','rb')
    if id == 7:
        f = open('result\ea.pkl','rb')
    if id == 8:
        f = open('result\sgd.pkl','rb')
    result = pickle.load(f)
    all = np.zeros(len(result[0]))
    y = []
    for one in result:
        tmp = np.zeros(len(one))
        for i in range(len(one)):
            tmp[i] = one[i]
            all[i] += one[i]
        y.append(tmp)
    all = all/len(result)
    
    # tmp = np.zeros(13)
    # for i in range(13):
    #     tmp[i] = y[i][90]
    # print('middle:',np.around(np.mean(tmp),4),np.around(np.std(tmp),4))
    
    
    # tmp = np.zeros(13)
    # for i in range(13):
    #     tmp[i] = y[i][179]
    # print('result:',np.around(np.mean(tmp),4),np.around(np.std(tmp),4))
    return y,all

def showone(id):
    y,all = help(id)
    x = np.arange(len(y[0]))
    loc = math.ceil(len(y)**(1/2))
    for i in range(len(y)):
        plt.subplot2grid((loc,loc),(int(i/loc),i%loc))
        plt.plot(x,y[i])
        plt.rcParams.update({'font.size':5})
        plt.legend(['test'])
    plt.show()

    plt.plot(x,all)
    plt.show()

    return all

# same caculate
def showall():
    _,y1 = help(1)
    _,y2 = help(2)
    _,y3 = help(3)
    _,y4 = help(4)
    _,y5 = help(5)
    _,y6 = help(6)
    _,y7 = help(7)
    _,y8 = help(8)
    x2 = np.arange(len(y7))*50000
    x = np.arange(len(y1))*20*25
    plt.plot(x2[:3],y7[:3],label = 'GA',linestyle=(0, (3,1,1,1),),color = 'silver')
    plt.plot(x,y8[:len(x)],label = 'SGD',color = 'grey',linestyle=(0, (1,1),))
    plt.plot(x,y3[:len(x)],label = 'Adam',color = 'brown',linestyle=(0, (1,1),))
    plt.plot(x,y1,label = 'LEEA',)
    plt.plot(x,y4,label = 'DeiEA-PO',)
    plt.plot(x,y5,label = 'DeiEA-EP',)
    plt.plot(x,y6,label = 'DeiEA-RI',)
    plt.plot(x,y2,label = 'DeiEA',color = 'orange')
    plt.legend(loc ='lower right')
        
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)


    #标题大小，粗细,位置，边框
    # plt.title('Number of Reported Results',fontsize =12, fontweight="semibold")#loc='left',bbox=dict(facecolor='y', edgecolor='blue', alpha=0.65 )
    plt.xlabel('Training examples',fontweight="semibold")
    plt.ylabel('Accuracy',fontweight="semibold")
    plt.ylim(0.08,0.3)
    plt.show()

# same gen
    
# showone(8)
showall()