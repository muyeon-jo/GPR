import torch
import numpy as np

freq = np.array([1,2,2,1,3,1,1,1]).repeat((10+1),axis=0)
print(freq)
a = torch.LongTensor(freq).reshape(-1,1)
print(a)