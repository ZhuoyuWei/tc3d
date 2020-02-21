from scipy import spatial
import numpy as np
import time

points=np.random.rand(100000,3)

start = time.time()
tree = spatial.KDTree(points)
end = time.time()
print('[IN] build kdtree {} \n'.format(end - start))


querypoints=np.random.rand(1000,3)

start = time.time()
q1=tree.query(querypoints,k=10)
end = time.time()
print('[q1 {} \n'.format(end - start))


querypoints=np.random.rand(1500,3)

start = time.time()
q1=tree.query(querypoints,k=10)
end = time.time()
print('[q2 {} \n'.format(end - start))