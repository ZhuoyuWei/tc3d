import click
import glob
import joblib
import json
import os
import pandas as pd
import time
import threading
import xgboost

def extract_case_id(fname):
    fname = os.path.basename(fname)
    case_id, _ = os.path.splitext(fname)
    return case_id

def elements_2_nodes(elements,nodes):
    node2count={}
    for i,ele in enumerate(elements):
        if not ele['node_id'] in node2count:
            node2count[ele['node_id']]=0
        node2count[ele['node_id']]+=1
    counts=[0]*len(nodes)
    #for node in nodes:
    for i,node in enumerate(nodes):
        if node['node_id'] in node2count:
            counts[i]=1

    return counts





def read_input_df(fname):
    with open(fname) as inf:
        input_obj = json.load(inf)

    node_size=len(input_obj['nodes'])

    push_counts=elements_2_nodes(input_obj['push_elements'],input_obj['nodes'])
    surf_counts=elements_2_nodes(input_obj['surf_elements'],input_obj['nodes'])
    nset_fix_counts=elements_2_nodes(input_obj['nset_fix'],input_obj['nodes'])
    nset_osibou_counts=elements_2_nodes(input_obj['nset_osibou'],input_obj['nodes'])

    '''
    print('nodes origin: {}'.format(len(input_obj['nodes'])))
    print('nodes push_elements: {}'.format(len(push_counts)))
    print('nodes surf_elements: {}'.format(len(surf_counts)))
    print('nodes nset_fix: {}'.format(len(nset_fix_counts)))
    print('nodes nset_osibou: {}'.format(len(nset_osibou_counts)))
    '''

    push_counts=pd.DataFrame(data=push_counts,dtype=int)
    surf_counts=pd.DataFrame(data=surf_counts,dtype=int)
    nset_fix_counts=pd.DataFrame(data=nset_fix_counts,dtype=int)
    nset_osibou_counts=pd.DataFrame(data=nset_osibou_counts,dtype=int)

    thickness = float(input_obj['config']['thickness'])
    df = pd.DataFrame(input_obj['nodes']).astype({'node_id': int, 'x': float, 'y': float, 'z': float})

    move_id = input_obj['move_node_id']
    move_node = df[df['node_id'] == int(move_id)].iloc[0].to_dict()

    dx = df['x'] - move_node['x']
    dy = df['y'] - move_node['y']
    dz = df['z'] - move_node['z']


    #push_element
    #push_elments=[]
    #for

    #during training, can remove fix nodes

    return df.assign(dx=dx, dy=dy, dz=dz,
                     pcounts=push_counts, scounts=surf_counts,
                     nf_counts=nset_fix_counts, no_counts=nset_osibou_counts,
                     thickness=thickness),input_obj

model_config = {'n_estimators': 100, 'max_depth': 3,
                'n_jobs': 16, 'tree_method': 'gpu_hist'}
# lm_x = LinearRegression()
# lm_x = MLPRegressor(hidden_layer_sizes=(50,20), max_iter=2)
lm_x = xgboost.XGBRegressor(n_estimators=model_config['n_estimators'],
                            max_depth=model_config['max_depth'],
                            n_jobs=model_config['n_jobs'],
                            random_state=42,
                            tree_method=model_config['tree_method'])

print('XGboost is successful on gpu')

all_dfs = []
start = time.time()
for fname in glob.glob(f'/code/model/*.json'):
    input_df, input_obj = read_input_df(fname)
    case_id = extract_case_id(fname)
    output_df = pd.read_csv(f'/code/gt/{case_id}.csv')
    merged_df = input_df.merge(output_df, on='node_id', suffixes=['_in', '_out'])
    all_dfs.append(merged_df)
end = time.time()
print('reading training data cost {} s'.format(end - start))

train_df = pd.concat(all_dfs, ignore_index=True)

fitting_threads = []
feature_in_list = ['x', 'y', 'z', 'dx_in', 'dy_in', 'dz_in', 'thickness',
                   'pcounts', 'scounts', 'nf_counts', 'no_counts']

start = time.time()
lm_x.fit(train_df[feature_in_list], train_df['dx_out'])
end = time.time()
print('train {} model {}'.format('dx_out', end - start))

