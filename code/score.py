#!/usr/bin/env python

import click
import pandas as pd
import sys
import os


def str_list(xs):
    if len(xs) > 6:
        xs = list(xs[:5]) + ['...']
    return ', '.join(map(str, xs))


def id_list(df):
    return str_list(df['node_id'])


@click.group()
def cli():
    pass


@cli.command()
@click.argument('soln_file')
@click.argument('truth_file')
def validate(soln_file, truth_file):
    soln = pd.read_csv(soln_file)
    truth = pd.read_csv(truth_file)

    soln_cols = soln.columns
    truth_cols = truth.columns

    if set(soln_cols) != set(truth_cols):
        missing_cols = [o for o in truth_cols if o not in soln_cols]
        if missing_cols:
            print('Solution is missing required columns:', str_list(missing_cols))

        extra_cols = [o for o in soln_cols if o not in truth_cols]
        if extra_cols:
            print('Solution has some extra columns:', str_list(extra_cols))

        sys.exit(1)

    if len(soln) != soln['node_id'].nunique():
        counts = soln.groupby('node_id').size().reset_index(name='n')
        dupes = id_list(counts[counts['n'] > 1])
        print('Solution contains duplicate node ID:', dupes)
        sys.exit(1)

    merged = truth[['node_id']].merge(soln[['node_id']], on='node_id', indicator=True, how='outer')
    if len(merged) != len(soln) or len(merged) != len(truth):
        missed_ids = merged[merged['_merge'] == 'left_only']
        if not missed_ids.empty:
            print('Solution is missing some node IDs:', id_list(missed_ids))

        extra_ids = merged[merged['_merge'] == 'right_only']
        if not extra_ids.empty:
            print('Solution has some extra node IDs:', id_list(extra_ids))

        sys.exit(1)

    non_numeric_cols = [
        col for col in soln_cols
        if not pd.api.types.is_numeric_dtype(soln[col])
    ]
    if non_numeric_cols:
        print('Solution contains non-numeric columns:', str_list(non_numeric_cols))
        sys.exit(1)

    print('Solution is valid!')


STRESS_WEIGHT = 0.01595615


@cli.command()
@click.argument('soln_file')
@click.argument('truth_file')
@click.option('--verbose/--quiet', default=False)
def score(soln_file, truth_file, verbose):
    soln = pd.read_csv(soln_file)
    truth = pd.read_csv(truth_file)
    if set(soln.columns) != set(truth.columns):
        raise ValueError('columns do not match')

    if len(soln) != soln['node_id'].nunique():
        raise ValueError('solution contains duplicate nodes')

    merged = soln.merge(truth, on='node_id', suffixes=['_soln', '_truth'])
    if len(merged) != len(soln) or len(merged) != len(truth):
        raise ValueError('merge was not perfect')

    dx = merged['dx_soln'] - merged['dx_truth']
    dy = merged['dy_soln'] - merged['dy_truth']
    dz = merged['dz_soln'] - merged['dz_truth']
    d_stress = merged['max_stress_soln'] - merged['max_stress_truth']

    disp2 = dx*dx + dy*dy + dz*dz
    stress2 = d_stress * d_stress
    disp_rmse = disp2.mean()**(1/2)
    stress_rmse = stress2.mean()**(1/2)
    if verbose:
        print(f'RMSE for displacement={disp_rmse}')
        print(f'RMSE for stress={stress_rmse}')

    # Weight stress so that it is on a similar scale to displacement.
    overall_wtd_rmse = disp_rmse + STRESS_WEIGHT * stress_rmse

    # Transform so that lower scores are better and scores are between 0 and 1.
    score = (1 / (1 + overall_wtd_rmse)) * 100
    print(score)

def _score(soln_file, truth_file):
    soln = pd.read_csv(soln_file)
    truth = pd.read_csv(truth_file)
    if set(soln.columns) != set(truth.columns):
        raise ValueError('columns do not match')

    if len(soln) != soln['node_id'].nunique():
        raise ValueError('solution contains duplicate nodes')

    merged = soln.merge(truth, on='node_id', suffixes=['_soln', '_truth'])
    if len(merged) != len(soln) or len(merged) != len(truth):
        raise ValueError('merge was not perfect')

    dx = merged['dx_soln'] - merged['dx_truth']
    dy = merged['dy_soln'] - merged['dy_truth']
    dz = merged['dz_soln'] - merged['dz_truth']
    d_stress = merged['max_stress_soln'] - merged['max_stress_truth']

    disp2 = dx*dx + dy*dy + dz*dz
    stress2 = d_stress * d_stress
    disp_rmse = disp2.mean()**(1/2)
    stress_rmse = stress2.mean()**(1/2)


    # Weight stress so that it is on a similar scale to displacement.
    overall_wtd_rmse = disp_rmse + STRESS_WEIGHT * stress_rmse

    # Transform so that lower scores are better and scores are between 0 and 1.
    score = (1 / (1 + overall_wtd_rmse)) * 100
    print(score)
    return  score


@cli.command()
@click.argument('soln_dir')
@click.argument('truth_dir')
def score_all(soln_dir,truth_dir):
    files=os.listdir(soln_dir)
    sum_score=0
    for file in files:
        one_score=_score(os.path.join(soln_dir,file),
                        os.path.join(truth_dir,file))
        sum_score+=one_score
    sum_score/=len(files)
    print('total score is {}'.format(sum_score))



if __name__ == '__main__':
    cli()
