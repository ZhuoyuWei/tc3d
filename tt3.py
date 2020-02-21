import torch
import numpy as np
import time

start=time.time()
a=np.random.rand(200000,3)
b=np.random.rand(2000,3)
#a=torch.Tensor(a)
#b=torch.Tensor(b)
end=time.time()
print('build {}'.format(end-start))
start=time.time()
#a=a.to(device='cuda:0')
#b=b.to(device='cuda:0')

end=time.time()
print('gpu {}'.format(end-start))

start=time.time()
c=np.matmul(a,np.transpose(b))
end=time.time()
print('matmul {}'.format(end-start))

start=time.time()
#c=c.topk(k=3,dim=-1,largest=False)
c=np.min(c,-1)
end=time.time()
print('top3 {}'.format(end-start))

