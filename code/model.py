#!/usr/bin/env python

import click
import glob
import joblib
import json
import os
import pandas as pd
import time
import threading
import xgboost

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


@click.group()
def cli():
    pass


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

class fit_thread(threading.Thread):

    def __init__(self,lm,train_df,target):
        threading.Thread.__init__(self)
        self.lm=lm
        self.train_df=train_df
        self.target=target

    def run(self):
        print('train {} starts'.format(self.target))
        start = time.time()
        self.lm.fit(self.train_df[['x','y','z','dx_in', 'dy_in', 'dz_in', 'thickness',
                                   'pcounts','scounts','nf_counts','no_counts']],
                      self.train_df[self.target])
        end = time.time()
        print('train {} model {}'.format(self.target,end - start))


@cli.command()
@click.argument('input-dir')
@click.argument('ground-truth-dir')
@click.argument('model-file')
@click.argument('n_estimators')
@click.argument('max_depth')
@click.argument('tree_method')
@click.argument('n_jobs')
@click.argument('sample_rate')
def train(input_dir, ground_truth_dir, model_file, n_estimators, max_depth, tree_method,n_jobs,
          sample_rate):

    n_estimators=int(n_estimators)
    max_depth=int(max_depth)
    n_jobs=int(n_jobs)
    sample_rate=float(sample_rate)

    all_dfs = []
    start=time.time()
    for fname in glob.glob(f'{input_dir}/*.json'):
        input_df,input_obj = read_input_df(fname)
        case_id = extract_case_id(fname)
        output_df = pd.read_csv(f'{ground_truth_dir}/{case_id}.csv')
        merged_df = input_df.merge(output_df, on='node_id', suffixes=['_in', '_out'])
        all_dfs.append(merged_df)
    end=time.time()
    print('reading training data cost {} s'.format(end-start))

    train_df = pd.concat(all_dfs, ignore_index=True)

    if sample_rate < 1:
        train_df = train_df.sample(frac=sample_rate, random_state=42)

    fitting_threads=[]
    feature_in_list=['x','y','z','dx_in', 'dy_in', 'dz_in', 'thickness',
                                   'pcounts','scounts','nf_counts','no_counts']
    model_config={'n_estimators':n_estimators,'max_depth':max_depth,
                  'n_jobs': n_jobs, 'tree_method':tree_method}
    #lm_x = LinearRegression()
    #lm_x = MLPRegressor(hidden_layer_sizes=(50,20), max_iter=2)
    lm_x = xgboost.XGBRegressor(n_estimators=model_config['n_estimators'],
                              max_depth=model_config['max_depth'],
                              n_jobs=model_config['n_jobs'],
                              random_state=42,
                              tree_method=model_config['tree_method'])
    start = time.time()
    lm_x.fit(train_df[feature_in_list],train_df['dx_out'])
    end = time.time()
    print('train {} model {}'.format('dx_out', end - start))
    #fitting_threads.append(fit_thread(lm_x,train_df,'dx_out'))
    joblib.dump(lm_x, model_file + '.x')
    #lm_x.get_booster().set_attr(scikit_learn=None)
    lm_x=None

    #lm_y = LinearRegression()
    #lm_y = MLPRegressor(hidden_layer_sizes=(50,20), max_iter=2)
    lm_y = xgboost.XGBRegressor(n_estimators=model_config['n_estimators'],
                              max_depth=model_config['max_depth'],
                              n_jobs=model_config['n_jobs'],
                              random_state=42,
                              tree_method=model_config['tree_method'])
    start = time.time()
    lm_y.fit(train_df[feature_in_list],train_df['dy_out'])
    end = time.time()
    print('train {} model {}'.format('dy_out', end - start))
    #fitting_threads.append(fit_thread(lm_y, train_df, 'dy_out'))
    joblib.dump(lm_y, model_file + '.y')
    #lm_y.get_booster().set_attr(scikit_learn=None)
    lm_y=None

    #lm_z = LinearRegression()
    #lm_z = MLPRegressor(hidden_layer_sizes=(50,20), max_iter=2)
    lm_z = xgboost.XGBRegressor(n_estimators=model_config['n_estimators'],
                              max_depth=model_config['max_depth'],
                              n_jobs=model_config['n_jobs'],
                              random_state=42,
                              tree_method=model_config['tree_method'])
    #fitting_threads.append(fit_thread(lm_z, train_df, 'dz_out'))
    start = time.time()
    lm_z.fit(train_df[feature_in_list],train_df['dz_out'])
    end = time.time()
    print('train {} model {}'.format('dz_out', end - start))
    joblib.dump(lm_z, model_file + '.z')
    #lm_z.get_booster().set_attr(scikit_learn=None)
    lm_z=None
    #lm_s = LinearRegression()
    #lm_s = MLPRegressor(hidden_layer_sizes=(50,20), max_iter=2)
    lm_s = xgboost.XGBRegressor(n_estimators=model_config['n_estimators'],
                              max_depth=model_config['max_depth'],
                              n_jobs=model_config['n_jobs'],
                              random_state=42,
                              tree_method=model_config['tree_method'])
    start = time.time()
    #lm_s.fit(train_df[feature_in_list],train_df['max_stress'])
    end = time.time()
    print('train {} model {}'.format('ds_out', end - start))
    #fitting_threads.append(fit_thread(lm_s, train_df, 'max_stress'))

    #for i in range(len(fitting_threads)):
    #    fitting_threads[i].start()

    #for i in range(len(fitting_threads)):
    #    fitting_threads[i].join()
    joblib.dump(lm_s, model_file + '.s')
    #lm_s.get_booster().set_attr(scikit_learn=None)
    lm_s=None





def post_procssing(pred_df,input_obj):
    fix_nodes=set()
    for item in input_obj["nset_fix"]:
        fix_nodes.add(int(item['node_id']))
        #print('fixnodeset:\t{}'.format(item['node_id']))

    fix_count=0
    total_count=0
    for index, row in pred_df.iterrows():
        #print('preds:\t{}'.format(row['node_id']))
        if row['node_id'] in fix_nodes:
            pred_df['dx'][index]=0
            pred_df['dy'][index]=0
            pred_df['dz'][index]=0
            fix_count+=1
            #print(str(row).replace('\n','\t'))
        total_count+=1
    #print('Debug Fix count {} == fix set {} in total {}'.format(fix_count,len(fix_nodes),total_count))


    return pred_df


def post_procssing_debug(pred_df, input_obj):
    fix_nodes = set()
    for item in input_obj["nset_fix"]:
        fix_nodes.add(int(item['node_id']))
        # print('fixnodeset:\t{}'.format(item['node_id']))

    fix_count = 0
    total_count = 0
    for index, row in pred_df.iterrows():
        # print('preds:\t{}'.format(row['node_id']))
        if row['node_id'] in fix_nodes:
            print('debug = {}'.format(str(row).replace('\n','\t')))



def _predict(models, input_file, output_file):
    input_df,input_obj = read_input_df(input_file)
    input_df.rename(columns={'dx':'dx_in'}, inplace=True)
    input_df.rename(columns={'dy':'dy_in'}, inplace=True)
    input_df.rename(columns={'dz':'dz_in'}, inplace=True)
    dz_preds=[]
    for i in range(len(models)):
        dz_pred = models[i].predict(input_df[['x','y','z','dx_in', 'dy_in', 'dz_in', 'thickness',
                                   'pcounts','scounts','nf_counts','no_counts']])
        dz_preds.append(dz_pred)
    pred_df = pd.DataFrame([
        {'node_id': i, 'dx': x, 'dy': y, 'dz': z, 'max_stress': s}
        for i, x,y,z,s in zip(input_df['node_id'], dz_preds[0],dz_preds[1],dz_preds[2],dz_preds[3])
    ])
    #pred_df=post_procssing(pred_df,input_obj)
    #post_procssing_debug(pred_df,input_obj)

    pred_df.to_csv(output_file, index=False)


@cli.command()
@click.argument('model-file')
@click.argument('input-file')
@click.argument('output-file')
def predict_one(model_file, input_file, output_file):
    model = joblib.load(model_file)
    _predict(model, input_file, output_file)


@cli.command()
@click.argument('model-file')
@click.argument('input-dir')
@click.argument('output-dir')
def predict_all(model_file, input_dir, output_dir):
    models=[joblib.load(model_file+'.x'),
            joblib.load(model_file + '.y'),
            joblib.load(model_file + '.z'),
            joblib.load(model_file + '.s')]
    #model = joblib.load(model_file)
    start=time.time()
    for input_file in glob.glob(f'{input_dir}/*.json'):
        case_id = extract_case_id(input_file)
        _predict(models,input_file, f'{output_dir}/{case_id}.csv')
    end=time.time()
    print('Predict is finished in {} s'.format(end-start))


if __name__ == '__main__':
    cli()
