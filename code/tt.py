from scipy.spatial import KDTree
import  numpy as np
import time

data= np.random.rand(10510,3)
sample= np.random.rand(4350,3)

point=[[0,0,0],[1,1,1]]
print(point)
start=time.time()
kdtree=KDTree(data)
end=time.time()
print('build tree {}'.format(end-start))

start=time.time()
neast=kdtree.query(sample)
end=time.time()
print('searching {}'.format(end-start))


neast=kdtree.query(point,k=5)
print(neast)