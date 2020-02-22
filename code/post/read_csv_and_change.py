import sys
import os
import pandas as pd
import time

start=time.time()

inputdir=sys.argv[1]
outptudir=sys.argv[2]

files=os.listdir(inputdir)
for file in files:
    df = pd.read_csv(os.path.join(inputdir,file))
    df.loc[df['max_stress'] <0, 'max_stress'] = 5

    df.to_csv(os.path.join(outptudir,file), index=False)

end=time.time()

print('time: {}'.format(end-start))