import os
import sys

input_dir=sys.argv[1]
files=os.listdir(input_dir)

buffer=[]
for file in files:
    print(file)
    if file.endswith('.csv'):
        file=file.replace('.csv','')
        buffer.append(file)

print(len(buffer))
with open(sys.argv[2],'w') as fout:
    buf_str=','.join(buffer)
    fout.write(buf_str+'\n')
