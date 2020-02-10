import sys
import os
import random
import shutil

random.seed(2020)

input_dir=sys.argv[1]
output_dir=sys.argv[2]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cv_size=5

files=os.listdir(os.path.join(input_dir,'model'))

chunk_size=len(files)//cv_size

ids=[]
for file in files:
    ss=file.split('.')
    if len(ss) == 2:
        ids.append(ss[0])

random.shuffle(ids)

subdatas=[]
for i in range(cv_size-1):
    subdatas.append(ids[i*chunk_size:(i+1)*chunk_size])
subdatas.append(ids[(cv_size-1)*chunk_size:])

for i in range(cv_size):
    train_buffer=[]
    for j in range(cv_size):
        if i == j:
            continue
        train_buffer+=subdatas[j]

    i_output=os.path.join(output_dir,str(i))
    i_train=os.path.join(i_output,'train')
    i_dev=os.path.join(i_output,'dev')
    gt_train_output = os.path.join(i_train, 'gt')
    model_train_output=os.path.join(i_train,'model')
    gt_dev_output = os.path.join(i_dev, 'gt')
    model_dev_output=os.path.join(i_dev,'model')

    os.makedirs(i_output)
    os.makedirs(i_train)
    os.makedirs(i_dev)
    os.makedirs(gt_train_output)
    os.makedirs(model_train_output)
    os.makedirs(gt_dev_output)
    os.makedirs(model_dev_output)



    for id in train_buffer:
        shutil.copy(os.path.join(input_dir,'model',id+'.json'),
                    os.path.join(model_train_output,id+'.json'))
        shutil.copy(os.path.join(input_dir,'gt',id+'.csv'),
                    os.path.join(gt_train_output,id+'.csv'))

    for id in subdatas[i]:
        shutil.copy(os.path.join(input_dir,'model',id+'.json'),
                    os.path.join(model_dev_output,id+'.json'))
        shutil.copy(os.path.join(input_dir,'gt',id+'.csv'),
                    os.path.join(gt_dev_output,id+'.csv'))


