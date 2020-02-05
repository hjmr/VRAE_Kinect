import argparse
import pickle
import json

import numpy as np
import torch

from VRAE import VRAE


def parse_arg():
    parser = argparse.ArgumentParser(description='Generate movement corresponding to a set of latent variables.')
    parser.add_argument('-m', '--model', type=str, nargs=1,
                        help='load a pre-trained model which is used to generate the movement.')
    parser.add_argument('-l', '--length', type=int, default=500,
                        help='movement length to be generated.')
    parser.add_argument('-o', '--output_file', type=str,
                        help='a file name to which JSON data will be stored.')
    parser.add_argument('latent_data', type=str, nargs=1,
                        help='a set of latent variables.')
    return parser.parse_args()


def generate(args):
    model = torch.load(args.model[0])
    model.eval()

    latent = torch.FloatTensor([[float(x) for x in args.latent_data[0].replace('m', '-').split(',')]])
    ret = None
    with torch.no_grad():
        ret = model.generate(latent, args.length)
    return ret[0]  # first batch


def pack_data_at_a_time(timestamp, joint_loc, uid=0):
    data = {}
    data['time'] = timestamp
    data['user'] = uid
    joints = []
    for i in range(15):
        joint = {}
        joint['id'] = i
        joint['confidence'] = 1.0
        joint['x'] = joint_loc[i * 3 + 0].item()
        joint['y'] = joint_loc[i * 3 + 1].item()
        joint['z'] = joint_loc[i * 3 + 2].item()
        joints.append(joint)
    data['joints'] = joints
    return data


def pack_data(series_data):
    data = []
    timestamp = 0
    for a_time_data in series_data:
        data.append(pack_data_at_a_time(timestamp, a_time_data))
        timestamp += 1 / 30
    return data


def save_data(file_name, data):
    with open(file_name, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=False, separators=(',', ': '))


if __name__ == '__main__':
    args = parse_arg()
    output = generate(args)
    if args.output_file is not None:
        save_data(args.output_file, pack_data(output))
    else:
        print(output)
