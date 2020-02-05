import argparse
import pickle


def parse_arg():
    parser = argparse.ArgumentParser(description='Display arguments')
    parser.add_argument('arg_file', type=str, nargs=1, help='file name of the pickled argument file.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()
    with open(args.arg_file[0], 'rb') as f:
        train_args = pickle.load(f)

    print('               input dims: {}'.format(train_args.input_dims))
    print('     # of encoding layers: {}'.format(train_args.enc_layers))
    print(' hidden states of encoder: {}'.format(train_args.enc_states))
    print('              latent dims: {}'.format(train_args.latent_dims))
    print('     # of decoding layers: {}'.format(train_args.dec_layers))
    print(' hidden states of decoder: {}'.format(train_args.dec_states))
    print('            dropout ratio: {}'.format(train_args.dropout_rate))
    print('          # of train data: {}'.format(len(train_args.data_files)))
    print('             train epochs: {}'.format(train_args.epoch))
    if hasattr(train_args, 'beta'):
        print('                     beta: {}'.format(train_args.beta))
    if hasattr(train_args, 'batch_size'):
        print('               batch size: {}'.format(train_args.batch_size))
