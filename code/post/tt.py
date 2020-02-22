import pandas as pd
import numpy as np

a=np.random.rand(10)
b=np.random.rand(10)

df=pd.DataFrame({'a':a,'b':b})

print(df)

df.loc[df['a'] > 0.5, 'a'] = -1

print(df)