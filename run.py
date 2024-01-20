from leea import leea
from leea_pro import leea_pro
from leea_pro1 import leea_pro1
from leea_pro2 import leea_pro2
from leea_pro3 import leea_pro3
from Adam import Adam
from ea import ea
from sgd import sgd
import pickle

def getresult(id,num):
    result = []
    if id == 1:
        for i in range(num):
            print('*****************now is num:',i)
            result.append(leea.leea_run(400,3600))
        with open('result\\leea.pkl','wb') as f:
            pickle.dump(result,f)
        f.close()
    if id == 2:
        for i in range(num):
            print('*****************now is num:',i)
            result.append(leea_pro.leea_pro_run(400,3600))
        with open('result\\leeapro.pkl','wb') as f:
            pickle.dump(result,f)
        f.close()
    if id == 3:
        for i in range(num):
            print('*****************now is num:',i)
            result.append(Adam.adam_run(720))
        with open('result\\Adam.pkl','wb') as f:
            pickle.dump(result,f)
        f.close()
    
    if id == 4:
        for i in range(num):
            print('*****************now is num:',i)
            result.append(leea_pro1.leea_pro_run(400,3600))
        with open('result\\leeapro1.pkl','wb') as f:
            pickle.dump(result,f)
        f.close()
    if id == 5:
        for i in range(num):
            print('*****************now is num:',i)
            result.append(leea_pro2.leea_pro_run(400,3600))
        with open('result\\leeapro2.pkl','wb') as f:
            pickle.dump(result,f)
        f.close()
    if id == 6:
        for i in range(num):
            print('*****************now is num:',i)
            result.append(leea_pro3.leea_pro_run(400,3600))
        with open('result\\leeapro3.pkl','wb') as f:
            pickle.dump(result,f)
        f.close()
    if id == 7:
        for i in range(num):
            print('*****************now is num:',i)
            result.append(ea.ea_run(400,4))
        with open('result\\ea.pkl','wb') as f:
            pickle.dump(result,f)
        f.close()
    if id == 8:
        for i in range(num):
            print('*****************now is num:',i)
            result.append(sgd.sgd_run(2))
        with open('result\\sgd.pkl','wb') as f:
            pickle.dump(result,f)
        f.close()

    return result

def run(id):
    if id == 1:
        leea.leea_run(400,2000)
    if id == 2:
        leea_pro1.leea_pro_run(400,2000)
    if id == 3:
        Adam.adam_run(5)

# run(3)
# getresult(1,13)
getresult(8,13)
# for i in range(1,8):
#     getresult(i,13)
