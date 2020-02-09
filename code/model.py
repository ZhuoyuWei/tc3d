#!/usr/bin/env python

import click
import glob
import joblib
import json
import os
import pandas as pd

from sklearn.linear_model import LinearRegression


@click.group()
def cli():
    pass


def extract_case_id(fname):
    fname = os.path.basename(fname)
    case_id, _ = os.path.splitext(fname)
    return case_id


def read_input_df(fname):
    with open(fname) as inf:
        input_obj = json.load(inf)

    thickness = float(input_obj['config']['thickness'])
    df = pd.DataFrame(input_obj['nodes']).astype({'node_id': int, 'x': float, 'y': float, 'z': float})

    move_id = input_obj['move_node_id']
    move_node = df[df['node_id'] == int(move_id)].iloc[0].to_dict()

    dx = df['x'] - move_node['x']
    dy = df['y'] - move_node['y']
    dz = df['z'] - move_node['z']
    return df.assign(dx=dx, dy=dy, dz=dz, thickness=thickness)


@cli.command()
@click.argument('input-dir')
@click.argument('ground-truth-dir')
@click.argument('model-file')
def train(input_dir, ground_truth_dir, model_file):
    all_dfs = []
    for fname in glob.glob(f'{input_dir}/*.json'):
        input_df = read_input_df(fname)
        case_id = extract_case_id(fname)
        output_df = pd.read_csv(f'{ground_truth_dir}/{case_id}.csv')
        merged_df = input_df.merge(output_df, on='node_id', suffixes=['_in', '_out'])
        all_dfs.append(merged_df)

    train_df = pd.concat(all_dfs, ignore_index=True)

    lm = LinearRegression()
    lm.fit(train_df[['dx_in', 'dy_in', 'dz_in', 'thickness']], train_df['dz_out'])
    joblib.dump(lm, model_file)


def _predict(model, input_file, output_file):
    input_df = read_input_df(input_file)

    dz_pred = model.predict(input_df[['dx', 'dy', 'dz', 'thickness']])
    pred_df = pd.DataFrame([
        {'node_id': i, 'dx': 0, 'dy': 0, 'dz': o, 'max_stress': 0}
        for i, o in zip(input_df['node_id'], dz_pred)
    ])
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
    model = joblib.load(model_file)
    for input_file in glob.glob(f'{input_dir}/*.json'):
        case_id = extract_case_id(input_file)
        _predict(model,input_file, f'{output_dir}/{case_id}.csv')


if __name__ == '__main__':
    cli()
