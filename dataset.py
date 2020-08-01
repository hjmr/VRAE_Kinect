import json
import argparse


def flatten_skeleton_data(skel_data):
    data_flatten = []
    for a_time_data in skel_data:
        entry = [0] * (len(a_time_data['joints']) * 3)  # num of joints * x, y, z
        for joint in a_time_data['joints']:
            entry[joint['id'] * 3 + 0] = joint['x'] / 100.0
            entry[joint['id'] * 3 + 1] = joint['y'] / 100.0
            entry[joint['id'] * 3 + 2] = joint['z'] / 100.0
        data_flatten.append(entry)
    return data_flatten


def get(file_name):
    data_flatten = []
    with open(file_name, "r") as _f:
        data_flatten = flatten_skeleton_data(json.load(_f))
    return data_flatten


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str, nargs='+', help='file name(s) of skeleton data.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()
    for fn in args.data_file:
        data_flatten = get(fn)
