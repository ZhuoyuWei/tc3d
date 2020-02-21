import torch
import numpy as np


a=np.random.rand(10,5)
b=torch.Tensor(a,device='cuda:0')
print(b)
c=b.topk(k=3,dim=-1,largest=False)

print(c)
