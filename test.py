import torch
import numpy as np
a = list(range(10))
b = list(range(10,20))
c = list(range(20,30))
d = [a,b,c]
idx = [0,2]
col = [[1,3,5,7],[0,1,2,3]]
tt = torch.tensor(d,dtype=torch.float32)
print(tt)
print(tt**2)
print(torch.sum(tt,dim=0))
print(torch.sum(tt,dim=1))
