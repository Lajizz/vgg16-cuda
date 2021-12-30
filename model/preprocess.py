# convert the fortmat of data
import json
import numpy as np
loaddir = "./paras/"
savedir = "./parasbin/"
with open("paras.json","r+") as f:
    paras = json.load(f)
    for key, value in paras.items():
        try:
            # print(value["weight"])
            array = np.load(loaddir+value["weight"],allow_pickle=True, encoding="latin1")  
            array.astype(np.float32).tofile(savedir+value["weight"]) 
            array = np.load(loaddir+value["bias"],allow_pickle=True, encoding="latin1")  
            array.astype(np.float32).tofile(savedir+value["bias"])   
        except:
            print("no key,pass")

