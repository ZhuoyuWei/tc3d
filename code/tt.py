import pandas as pd
import numpy as np

data1=np.random.rand(10)
data2=np.random.rand(10)
df=pd.DataFrame({'a':data1,'b':data2})
print(df)

for i,row in df.iterrows():
    row['a']=0

print(df)