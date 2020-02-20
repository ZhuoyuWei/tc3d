import pandas as pd
import numpy as np

a=np.random.rand(10)
b=np.random.rand(10)

df=pd.DataFrame({'aaa':a,'bbb':b})

print(df)


df_max=df.max()
print(df_max)

amax=df_max['aaa']
bmax=df_max['bbb']


print(amax)
print(bmax)

df['aaa']/=amax
df['bbb']/=bmax

print(df)