import json
import pandas as pd
import sys

def read_input_df(fname):
    with open(fname) as inf:
        input_obj = json.load(inf)

    thickness = float(input_obj['config']['thickness'])
    df = pd.DataFrame(input_obj['nodes']).astype({'node_id': int, 'x': float, 'y': float, 'z': float})

    move_id = input_obj['move_node_id']
    move_node = df[df['node_id'] == int(move_id)].iloc[0].to_dict()

    print('Move Node = {}'.format(move_node))

    dx = df['x'] - move_node['x']
    dy = df['y'] - move_node['y']
    dz = df['z'] - move_node['z']
    return df.assign(dx=dx, dy=dy, dz=dz, thickness=thickness)

read_df=read_input_df(sys.argv[1])
print(read_df.count())