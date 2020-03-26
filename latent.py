import os.path
import argparse
import pickle

import numpy as np
import torch

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from VRAE import VRAE
import dataset


def parse_arg():
    parser = argparse.ArgumentParser(description='Check latent variables of VRAE-LSTM for specified data files')
    parser.add_argument('-m', '--model', type=str, nargs=1,
                        help='load a pre-trained model which is used to caluclate latent variables.')
    parser.add_argument('-s', '--save_fig', type=str,
                        help='save the plot to specified file.')
    parser.add_argument('-n', '--no_plot', action='store_true',
                        help='print raw data instead of plotting.')
    parser.add_argument('-p', '--use_pca', action='store_true',
                        help='apply PCA before plotting.')
    parser.add_argument('-3', '--use_3d', action='store_true',
                        help='use 3D data.')
    parser.add_argument('-f', '--use_filename', action='store_true',
                        help='use filename instead of label for legend.')
    parser.add_argument('data_files', type=str, nargs='+',
                        help='file name(s) of skeleton data.')
    return parser.parse_args()


def convert_file_name_to_label(file_name):
    label_group = {'murao': 'A1', 'ryo': 'A2', 'megu': 'B1', 'mishima': 'B2', 'kyo': 'C1', 'riku': 'C2', 'zhang': 'C3'}

    user_name = file_name.split('_')[0]
    return label_group[user_name]


def get_latent(args):
    model = torch.load(args.model[0], map_location=torch.device('cpu'))
    model.eval()

    latents = []
    file_names = []
    for data_file in args.data_files:
        file_names.append(str(os.path.basename(data_file)))
        seq = torch.FloatTensor([dataset.get(data_file)])
        with torch.no_grad():
            mu, ln_var = model.encode(seq)
            latents.append(mu[0].tolist())
    return latents, file_names


def group_data_by_label(data, labels, x_index, y_index):
    tmp_dict = {}
    grouped_labels = []
    for i in range(len(data)):
        l = labels[i]
        if l not in tmp_dict.keys():
            tmp_dict[l] = {'x': [], 'y': []}
            grouped_labels.append(l)
        tmp_dict[l]['x'].append(data[i][x_index])
        tmp_dict[l]['y'].append(data[i][y_index])
    grouped_labels = sorted(grouped_labels)

    grouped_data = []
    for l in grouped_labels:
        grouped_data.append((tmp_dict[l]['x'], tmp_dict[l]['y'], l))

    return grouped_data, grouped_labels


def plot_latent(data, file_names, x_index, y_index, x_label='X', y_label='Y', use_filename=False):
    markers = {'A1': 'o', 'A2': 's', 'B1': 'o', 'B2': 's', 'C1': 'o', 'C2': 's', 'C3': 'D'}
    colors = {'A1': '0.0', 'A2': '0.0', 'B1': '0.5', 'B2': '0.5', 'C1': '1.0', 'C2': '1.0', 'C3': '1.0'}

    fig = pyplot.figure()
    axs = pyplot.axes()

    if use_filename:
        for d, f in zip(data, file_names):
            l = convert_file_name_to_label(f)
            axs.scatter(d[x_index], d[y_index], label=f, c=colors[l], marker=markers[l], linewidth=1, edgecolors='k')
            axs.text(d[x_index], d[y_index], f)
    else:
        labels = []
        for f in file_names:
            labels.append(convert_file_name_to_label(f))

        grouped_data, grouped_labels = group_data_by_label(data, labels, x_index, y_index)
        for x, y, l in grouped_data:
            axs.scatter(x, y, label=l, c=colors[l], marker=markers[l], linewidth=1, edgecolors='k')

    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)
    if not use_filename:
        axs.legend(grouped_labels)
    return fig


def group_3d_data_by_label(data, labels, x_index, y_index, z_index):
    tmp_dict = {}
    grouped_labels = []
    for i in range(len(data)):
        l = labels[i]
        if l not in tmp_dict.keys():
            tmp_dict[l] = {'x': [], 'y': [], 'z': []}
            grouped_labels.append(l)
        tmp_dict[l]['x'].append(data[i][x_index])
        tmp_dict[l]['y'].append(data[i][y_index])
        tmp_dict[l]['z'].append(data[i][y_index])
    grouped_labels = sorted(grouped_labels)

    grouped_data = []
    for l in grouped_labels:
        grouped_data.append((tmp_dict[l]['x'], tmp_dict[l]['y'], tmp_dict[l]['z'], l))

    return grouped_data, grouped_labels


def plot_3d_latent(data, file_names, x_index, y_index, z_index, x_label='X', y_label='Y', z_label='Z', use_filename=False):
    markers = {'A1': 'o', 'A2': 's', 'B1': 'o', 'B2': 's', 'C1': 'o', 'C2': 's', 'C3': 'D'}
    colors = {'A1': '0.0', 'A2': '0.0', 'B1': '0.5', 'B2': '0.5', 'C1': '1.0', 'C2': '1.0', 'C3': '1.0'}

    fig = pyplot.figure()
    axs = pyplot.axes(projection='3d')

    if use_filename:
        for d, f in zip(data, file_names):
            l = convert_file_name_to_label(f)
            axs.scatter(d[x_index], d[y_index], d[z_index],
                        label=f, c=colors[l], marker=markers[l], linewidth=1, edgecolors='k')
            axs.text(d[x_index], d[y_index], d[z_index], f)
    else:
        labels = []
        for f in file_names:
            labels.append(convert_file_name_to_label(f))

        grouped_data, grouped_labels = group_3d_data_by_label(data, labels, x_index, y_index, z_index)
        for x, y, z, l in grouped_data:
            axs.scatter(x, y, z, label=l, c=colors[l], marker=markers[l], linewidth=1, edgecolors='k')

    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)
    axs.set_zlabel(z_label)
    if not use_filename:
        axs.legend(grouped_labels)
    return fig


if __name__ == '__main__':
    args = parse_arg()
    latents, file_names = get_latent(args)

    if args.use_pca:
        pca = PCA(n_components=3)
        pca.fit(latents)
        print('# of components:{}'.format(pca.n_components_))
        print('Explained variance ratio:{}'.format(pca.explained_variance_ratio_))
        latents = pca.transform(latents)

    if args.no_plot:
        for lat, fname in zip(latents, file_names):
            s = []
            if args.use_filename:
                s.append(fname)
            else:
                lab = convert_file_name_to_label(fname)
                s.append(lab)
            for l in lat:
                s.append(str(l))
            print(','.join(s))
    else:
        if args.use_3d:
            fig = plot_3d_latent(latents, file_names, 0, 1, 2, use_filename=args.use_filename)
        else:
            fig = plot_latent(latents, file_names, 0, 1, use_filename=args.use_filename)

        if args.save_fig:
            fig.savefig(args.save_fig)
        else:
            fig.show()
            input('Press Enter')
