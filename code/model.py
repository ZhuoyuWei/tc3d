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
import random

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


@click.group()
def cli():
    pass


def extract_case_id(fname):
    fname = os.path.basename(fname)
    case_id, _ = os.path.splitext(fname)
    return case_id

def read_SPOS(spos):
    element_set=set()
    for sp in spos:
        element_set.add(sp['element_id'])
    return element_set

def elements_2_nodes(elements,nodes,element_set=None):
    node2count={}
    for i,ele in enumerate(elements):
        if element_set is None:
            node2count[ele['node_id']] = 1
        else:
            count=node2count.get(ele['node_id'],0)
            if ele['element_id'] in element_set:
                node2count[ele['node_id']] = 2
            elif count!=2:
                node2count[ele['node_id']] = 1

    counts=[0]*len(nodes)
    #for node in nodes:
    for i,node in enumerate(nodes):
        if node['node_id'] in node2count:
            counts[i]=node2count.get(node['node_id'],0)

    return counts

def elements_2_nodes_mid(elements,nodes,element_set=None):
    node2count={}
    for i,ele in enumerate(elements):
        if element_set is None:
            if int(ele['idx']) < 3:
                node2count[ele['node_id']] = 1
            else:
                node2count[ele['node_id']] = 3
        else:
            count=node2count.get(ele['node_id'],0)
            if ele['element_id'] in element_set:
                if int(ele['idx']) < 3:
                    node2count[ele['node_id']] = 2
                else:
                    node2count[ele['node_id']] = 4
            elif count!=2 and count!=4:
                if int(ele['idx']) < 3:
                    node2count[ele['node_id']] = 1
                else:
                    node2count[ele['node_id']] = 3

    counts=[0]*len(nodes)
    #for node in nodes:
    for i,node in enumerate(nodes):
        if node['node_id'] in node2count:
            counts[i]=node2count.get(node['node_id'],0)

    return counts



def read_input_df(fname):
    with open(fname) as inf:
        input_obj = json.load(inf)

    node_size=len(input_obj['nodes'])

    spos=read_SPOS(input_obj['surf_plate'])

    push_counts=elements_2_nodes(input_obj['push_elements'],input_obj['nodes'],spos)
    surf_counts=elements_2_nodes_mid(input_obj['surf_elements'],input_obj['nodes'],spos)
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

    def __init__(self,lm,train_df,featurelist,target):
        threading.Thread.__init__(self)
        self.lm=lm
        self.train_df=train_df
        self.target=target
        self.featurelist=featurelist

    def run(self):
        print('train {} starts'.format(self.target))
        start = time.time()
        self.lm.fit(self.train_df[self.featurelist],
                      self.train_df[self.target])
        end = time.time()
        print('train {} model {}'.format(self.target,end - start))

class predict_thread(threading.Thread):

    def __init__(self,lm,train_df,ntree_limit=0):
        threading.Thread.__init__(self)
        self.lm=lm
        self.train_df=train_df
        self._return=None
        self.ntree_limit=ntree_limit

    def run(self):
        print('predict start')
        start = time.time()
        if self.ntree_limit == 0:
            self._return=self.lm.predict(self.train_df[['x','y','z','dx_in', 'dy_in', 'dz_in', 'thickness',
                                   'pcounts','scounts','nf_counts','no_counts']])
        else:
            self._return=self.lm.predict(self.train_df[['x','y','z','dx_in', 'dy_in', 'dz_in', 'thickness',
                                   'pcounts','scounts','nf_counts','no_counts']],
                                   ntree_limit=self.ntree_limit)
        end = time.time()
        print('predict model end {}'.format(end - start))

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return

'''
class train_thread(threading.Thread):
    def __init__(self,gpu_id,sample_rate,random_seed):
        threading.Thread.__init__(self)
        self.gpu_id=gpu_id
        self.sample_rate=sample_rate
        self.random_seed=random_seed
    
    def run(self):
        if sample_rate < 1:
            train_df = train_df.sample(frac=sample_rate, random_state=42)
'''
@cli.command()
@click.argument('input-dir')
@click.argument('ground-truth-dir')
@click.argument('model-file')
@click.argument('n_estimators')
@click.argument('max_depth')
@click.argument('tree_method')
@click.argument('n_jobs')
@click.argument('sample_rate')
@click.argument('booster')
@click.argument('objective')
def train(input_dir, ground_truth_dir, model_file, n_estimators, max_depth, tree_method,n_jobs,
          sample_rate,booster,objective):

    n_estimators=int(n_estimators)
    max_depth=int(max_depth)
    n_jobs=int(n_jobs)
    sample_rate=float(sample_rate)

    all_dfs = []
    start=time.time()
    random.seed(42)
    for fname in glob.glob(f'{input_dir}/*.json'):
        #if sample_rate < 1:
        #    rand_v=random.random()
        #    if rand_v > sample_rate:
        #        continue
        input_df,input_obj = read_input_df(fname)
        case_id = extract_case_id(fname)
        output_df = pd.read_csv(f'{ground_truth_dir}/{case_id}.csv')
        merged_df = input_df.merge(output_df, on='node_id', suffixes=['_in', '_out'])
        #if sample_rate < 1:
        #    merged_df=merged_df.sample(frac=sample_rate, random_state=42)
        all_dfs.append(merged_df)
    end=time.time()
    print('reading training data cost {} s'.format(end-start))
    all_df = pd.concat(all_dfs, ignore_index=True)

    random_states=[42,2020,377]
    for MM in range(3):

        if sample_rate < 1:
            train_df = all_df.sample(frac=sample_rate, random_state=random_states[MM])
        else:
            train_df=all_df

        fitting_threads=[]
        feature_in_list=['x','y','z','dx_in', 'dy_in', 'dz_in', 'thickness',
                                   'pcounts','scounts','nf_counts','no_counts']
        model_config={'n_estimators':n_estimators,'max_depth':max_depth,
                    'n_jobs': n_jobs, 'tree_method':tree_method}
        #lm_x = LinearRegression()
        #lm_x = MLPRegressor(hidden_layer_sizes=(50,20), max_iter=2)
        lm_x = xgboost.XGBRegressor(n_estimators=model_config['n_estimators'],
                              max_depth=model_config['max_depth'],
                              #n_jobs=model_config['n_jobs'],
                              random_state=42,
                              tree_method=model_config['tree_method'],gpu_id=0,
                              predictor='gpu_predictor',
                              booster=booster,
                              objective=objective)
        start = time.time()
        #lm_x.fit(train_df[feature_in_list],train_df['dx_out'])
        fitting_threads.append(fit_thread(lm_x,train_df,feature_in_list,'dx_out'))
        end = time.time()
        print('train {} model {}'.format('dx_out', end - start))
        #joblib.dump(lm_x, model_file + '.x')
        #lm_x.get_booster().set_attr(scikit_learn=None)
        #lm_x=None

        #lm_y = LinearRegression()
        #lm_y = MLPRegressor(hidden_layer_sizes=(50,20), max_iter=2)
        lm_y = xgboost.XGBRegressor(n_estimators=model_config['n_estimators'],
                              max_depth=model_config['max_depth'],
                              #n_jobs=model_config['n_jobs'],
                              random_state=42,
                              tree_method=model_config['tree_method'],gpu_id=1,
                              predictor='gpu_predictor',
                              booster=booster,
                              objective=objective)
        start = time.time()
        #lm_y.fit(train_df[feature_in_list],train_df['dy_out'])
        fitting_threads.append(fit_thread(lm_y, train_df,feature_in_list, 'dy_out'))
        end = time.time()
        print('train {} model {}'.format('dy_out', end - start))
        #joblib.dump(lm_y, model_file + '.y')
        #lm_y.get_booster().set_attr(scikit_learn=None)
        #lm_y=None

        for i in range(len(fitting_threads)):
            fitting_threads[i].start()

        for i in range(len(fitting_threads)):
            fitting_threads[i].join()

        joblib.dump(lm_x, model_file + '.x.'+str(MM))
        joblib.dump(lm_y, model_file + '.y.'+str(MM))
        lm_x=None
        lm_y=None
        fitting_threads=[]

        #lm_z = LinearRegression()
        #lm_z = MLPRegressor(hidden_layer_sizes=(50,20), max_iter=2)
        lm_z = xgboost.XGBRegressor(n_estimators=model_config['n_estimators'],
                              max_depth=model_config['max_depth'],
                              #n_jobs=model_config['n_jobs'],
                              random_state=42,
                              tree_method=model_config['tree_method'],gpu_id=0,
                              predictor='gpu_predictor',
                              booster=booster,
                              objective=objective)
        #fitting_threads.append(fit_thread(lm_z, train_df, 'dz_out'))
        start = time.time()
        #lm_z.fit(train_df[feature_in_list],train_df['dz_out'])
        fitting_threads.append(fit_thread(lm_z, train_df, feature_in_list, 'dz_out'))
        end = time.time()
        print('train {} model {}'.format('dz_out', end - start))
        #joblib.dump(lm_z, model_file + '.z')
        #lm_z.get_booster().set_attr(scikit_learn=None)
        #lm_z=None


        #lm_s = LinearRegression()
        #lm_s = MLPRegressor(hidden_layer_sizes=(50,20), max_iter=2)
        lm_s = xgboost.XGBRegressor(n_estimators=model_config['n_estimators'],
                              max_depth=model_config['max_depth'],
                              #n_jobs=model_config['n_jobs'],
                              random_state=42,
                              tree_method=model_config['tree_method'],gpu_id=1,
                              predictor='gpu_predictor',
                              booster=booster,
                              objective=objective)
        start = time.time()
        #lm_s.fit(train_df[feature_in_list],train_df['max_stress'])
        fitting_threads.append(fit_thread(lm_s, train_df, feature_in_list,'max_stress'))
        end = time.time()
        print('train {} model {}'.format('ds_out', end - start))


        for i in range(len(fitting_threads)):
            fitting_threads[i].start()

        for i in range(len(fitting_threads)):
            fitting_threads[i].join()

        joblib.dump(lm_z, model_file + '.z.'+str(MM))
        joblib.dump(lm_s, model_file + '.s.'+str(MM))
        #lm_s.get_booster().set_attr(scikit_learn=None)
        lm_z=None
        lm_s=None
        fitting_threads=[]





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




def _predict(models, input_file, output_file,ntree_limit=0):
    input_df,input_obj = read_input_df(input_file)
    input_df.rename(columns={'dx':'dx_in'}, inplace=True)
    input_df.rename(columns={'dy':'dy_in'}, inplace=True)
    input_df.rename(columns={'dz':'dz_in'}, inplace=True)
    dz_preds=[None,None,None,None,None,None,None,None,None,None,None,None]
    predictThreads = []
    for MM in range(len(models)):

        #models[MM][0].set_params(tree_method='gpu_hist')
        #models[MM][0].set_params(gpu_id=0)
        #models[MM][1].set_params(tree_method='gpu_hist')
        #models[MM][1].set_params(gpu_id=1)
        thread_0=predict_thread(lm=models[MM][0],train_df=input_df,ntree_limit=ntree_limit)
        thread_1=predict_thread(lm=models[MM][1],train_df=input_df,ntree_limit=ntree_limit)
        thread_2=predict_thread(lm=models[MM][2],train_df=input_df,ntree_limit=ntree_limit)
        thread_3=predict_thread(lm=models[MM][3],train_df=input_df,ntree_limit=ntree_limit)
        predictThreads+=[thread_0,thread_1,thread_2,thread_3]




    for i in range(len(predictThreads)):
        predictThreads[i].start()

    for i in range(len(predictThreads)):
        dz_preds[i]=predictThreads[i].join()


    pred_df = pd.DataFrame([
        {'node_id': i, 'dx': (x1+x2+x3)/3, 'dy': (y1+y2+y3)/3, 'dz': (z1+z2+z3)/3, 'max_stress': (s1+s2+s3)/3}
        for i, x1,y1,z1,s1,x2,y2,z2,s2,x3,y3,z3,s3 in zip(input_df['node_id'],
                              dz_preds[0],dz_preds[1],dz_preds[2],dz_preds[3],
                              dz_preds[4], dz_preds[5], dz_preds[6], dz_preds[7],
                              dz_preds[8], dz_preds[9], dz_preds[10], dz_preds[11])
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
@click.argument('ntree_limit')
def predict_all(model_file, input_dir, output_dir,ntree_limit):
    ntree_limit=int(ntree_limit)
    models=[[],[],[]]
    models[0]=[joblib.load(model_file+'.x.0'),
            joblib.load(model_file + '.y.0'),
            joblib.load(model_file + '.z.0'),
            joblib.load(model_file + '.s.0')]
    models[1]=[joblib.load(model_file+'.x.1'),
            joblib.load(model_file + '.y.1'),
            joblib.load(model_file + '.z.1'),
            joblib.load(model_file + '.s.1')]
    models[2]=[joblib.load(model_file+'.x.2'),
            joblib.load(model_file + '.y.2'),
            joblib.load(model_file + '.z.2'),
            joblib.load(model_file + '.s.2')]

    for i in range(3):
        '''
        models[i][0]=models[i][0].set_params(tree_method='gpu_hist')
        models[i][0]=models[i][0].set_params(predictor='gpu_predictor')
        models[i][0]=models[i][0].set_params(gpu_id=0)

        models[i][1]=models[i][1].set_params(tree_method='gpu_hist')
        models[i][1]=models[i][1].set_params(predictor='gpu_predictor')
        models[i][1]=models[i][1].set_params(gpu_id=1)

        models[i][2]=models[i][2].set_params(tree_method='gpu_hist')
        models[i][2]=models[i][2].set_params(predictor='gpu_predictor')
        models[i][2]=models[i][2].set_params(gpu_id=0)

        models[i][3]=models[i][3].set_params(tree_method='gpu_hist')
        models[i][3]=models[i][3].set_params(predictor='gpu_predictor')
        models[i][3]=models[i][3].set_params(gpu_id=1)
        '''
        models[i][0].set_params(predictor='cpu_predictor')
        models[i][1].set_params(predictor='cpu_predictor')
        models[i][2].set_params(predictor='cpu_predictor')
        models[i][3].set_params(predictor='cpu_predictor')


    #model = joblib.load(model_file)
    start=time.time()
    for input_file in glob.glob(f'{input_dir}/*.json'):
        case_id = extract_case_id(input_file)
        _predict(models,input_file, f'{output_dir}/{case_id}.csv',ntree_limit=ntree_limit)
    end=time.time()
    print('Predict is finished in {} s'.format(end-start))


if __name__ == '__main__':
    cli()
