#!/usr/bin/env python

import click
import glob
import joblib
import json
import os
import pandas as pd
import time

from sklearn.linear_model import LinearRegression


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
    counts=[]
    #for node in nodes:



def read_input_df(fname):
    with open(fname) as inf:
        input_obj = json.load(inf)

    node_size=len(input_obj['nodes'])

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

    return df.assign(dx=dx, dy=dy, dz=dz, thickness=thickness),input_obj



@cli.command()
@click.argument('input-dir')
@click.argument('ground-truth-dir')
@click.argument('model-file')
def train(input_dir, ground_truth_dir, model_file):
    all_dfs = []
    for fname in glob.glob(f'{input_dir}/*.json'):
        input_df,input_obj = read_input_df(fname)
        case_id = extract_case_id(fname)
        output_df = pd.read_csv(f'{ground_truth_dir}/{case_id}.csv')
        merged_df = input_df.merge(output_df, on='node_id', suffixes=['_in', '_out'])
        all_dfs.append(merged_df)

    train_df = pd.concat(all_dfs, ignore_index=True)

    start=time.time()
    lm_x = LinearRegression()
    lm_x.fit(train_df[['dx_in', 'dy_in', 'dz_in', 'thickness']], train_df['dx_out'])
    end=time.time()
    print('train 1st model {}'.format(end-start))

    start=time.time()
    lm_y = LinearRegression()
    lm_y.fit(train_df[['dx_in', 'dy_in', 'dz_in', 'thickness']], train_df['dy_out'])
    end=time.time()
    print('train 2nd model {}'.format(end - start))

    start = time.time()
    lm_z = LinearRegression()
    lm_z.fit(train_df[['dx_in', 'dy_in', 'dz_in', 'thickness']], train_df['dz_out'])
    end=time.time()
    print('train 3rd model {}'.format(end - start))

    start = time.time()
    lm_s = LinearRegression()
    lm_s.fit(train_df[['dx_in', 'dy_in', 'dz_in', 'thickness']],train_df['max_stress'])
    end=time.time()
    print('train 4rd model {}'.format(end - start))

    joblib.dump(lm_x, model_file+'.x')
    joblib.dump(lm_y, model_file+'.y')
    joblib.dump(lm_z, model_file+'.z')
    joblib.dump(lm_s, model_file+'.s')

def post_procssing(pred_df,input_obj):
    fix_nodes=set()
    for item in input_obj["nset_fix"]:
        fix_nodes.add(item['node_id'])
        print('fixnodeset:\t'.format(item['node_id']))

    fix_count=0
    total_count=0
    for index, row in pred_df.iterrows():
        print('preds:\t{}'.format(row['node_id']))
        if row['node_id'] in fix_nodes:
            row['dx']=0
            row['dy']=0
            row['dz']=0
            fix_count+=1
        total_count+=1
    print('Debug Fix count {} == fix set {}'.format(fix_count,len(fix_nodes)))






def _predict(models, input_file, output_file):
    input_df,input_obj = read_input_df(input_file)
    dz_preds=[]
    for i in range(len(models)):
        dz_pred = models[i].predict(input_df[['dx', 'dy', 'dz', 'thickness']])
        dz_preds.append(dz_pred)
    pred_df = pd.DataFrame([
        {'node_id': i, 'dx': x, 'dy': y, 'dz': z, 'max_stress': s}
        for i, x,y,z,s in zip(input_df['node_id'], dz_preds[0],dz_preds[1],dz_preds[2],dz_preds[3])
    ])
    post_procssing(pred_df,input_obj)
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
    for input_file in glob.glob(f'{input_dir}/*.json'):
        case_id = extract_case_id(input_file)
        _predict(models,input_file, f'{output_dir}/{case_id}.csv')


if __name__ == '__main__':
    cli()
