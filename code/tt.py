import pandas as pd
import numpy as np

data1=np.random.rand(10)
data2=np.random.rand(10)
df=pd.DataFrame({'a':data1,'b':data2})
print(df)

count=0
for i,row in df.iterrows():
    if count==5:
        row['a']=0
    count+=1

print(df)