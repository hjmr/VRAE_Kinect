import argparse
import pickle

import torch
import torch.utils.data as tud

from VRAE import VRAE
import dataset


def parse_arg():
    parser = argparse.ArgumentParser(description='Train VRAE-LSTM using specified data files')
    # model parameters
    parser.add_argument('-n', '--enc_states', type=int, default=200,
                        help='the number of encoding states.')
    parser.add_argument('-l', '--enc_layers', type=int, default=1,
                        help='the number of layers in encoding net.')
    parser.add_argument('-d', '--dec_states', type=int, default=200,
                        help='the number of decoding states.')
    parser.add_argument('-k', '--dec_layers', type=int, default=1,
                        help='the number of layers in decoding net.')
    parser.add_argument('-z', '--latent_dims', type=int, default=2,
                        help='the dimension of encoded vector.')
    parser.add_argument('-p', '--dropout_rate', type=float, default=0.5,
                        help='the dropout ratio.')
    # training parameters
    parser.add_argument('-e', '--epoch', type=int, default=100,
                        help='the training epochs.')
    parser.add_argument('-b', '--batch_size', type=int,
                        help='a length of sequence to train at once.')
    parser.add_argument('-B', '--beta', type=float, default=1.0,
                        help='a parameter for KLD.')
    # output parameters
    parser.add_argument('-o', '--output_dir', type=str, default='.',
                        help='all output will be stored in the specified directory.')
    parser.add_argument('-f', '--base_file_name', type=str, default='VRAE_LSTM',
                        help='the base filename of the saved models.')
    parser.add_argument('-i', '--save_interval', type=int, default=100,
                        help='save checkpoints at every specified interval of the epoch')
    # load models from files
    parser.add_argument('-m', '--load_model', type=str,
                        help='initialize the model from given file.')
    parser.add_argument('-r', '--resume_from_checkpoint', type=str,
                        help='resume training from specified checkpoint.')
    # data files
    parser.add_argument('data_files', type=str, nargs='+',
                        help='file name(s) of skeleton data.')
    return parser.parse_args()


def init_data(data_files):
    data_set = []
    for file_name in data_files:
        data_set.append(torch.FloatTensor(dataset.get(file_name)))
    return data_set


def output_log(epoch, train_loss, test_loss):
    print("{}, {}, {}".format(epoch, train_loss, test_loss), flush=True)


def train_model():
    args = parse_arg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_set = init_data(args.data_files)
    n_input = max([d.data.shape[1] for d in data_set])
    num_data = len(data_set)

    args.input_dims = n_input
    with open('{}/args.pickle'.format(args.output_dir), 'wb') as f:
        pickle.dump(args, f)

    print('# GPU: {}'.format(device))
    print('# dataset num: {}'.format(num_data))
    print('# input dimensions: {}'.format(args.input_dims))
    print('# latent dimensions: {}'.format(args.latent_dims))
    print('# minibatch-size: {}'.format(args.batch_size))
    print('# epoch: {}'.format(args.epoch))
    print('')

    if args.load_model is not None:
        model = torch.load(args.load_model)
        model.eval()
    else:
        model = VRAE(args.input_dims, args.enc_states, args.latent_dims, args.dec_states,
                     args.enc_layers, args.dec_layers, args.dropout_rate).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    num_train = int(num_data * 0.8)
    num_test = num_data - num_train
    train_dat, test_dat = tud.random_split(data_set, [num_train, num_test])

    train_iter = tud.BatchSampler(tud.RandomSampler(range(len(train_dat))),
                                  batch_size=args.batch_size,
                                  drop_last=False)
    test_iter = tud.BatchSampler(tud.SequentialSampler(range(len(test_dat))),
                                 batch_size=args.batch_size,
                                 drop_last=False)

    if args.resume_from_checkpoint is not None:
        checkpoint = torch.load(args.resume_from_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(args.epoch):
        train_loss = 0
        for indices in train_iter:
            x_data = [train_dat[idx] for idx in indices]
            model.zero_grad()
            loss = model.loss(x_data, k=1)
            loss.backward()
            optimizer.step()
            train_loss += loss

        # evaluation
        test_loss = 0
        with torch.no_grad():
            for indices in test_iter:
                x_data = [test_dat[idx] for idx in indices]
                test_loss += model.loss(x_data, k=10)

        output_log(epoch, train_loss / len(train_iter), test_loss / len(test_iter))

        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss
            }
            checkpoint_path = '{}/{}_{}.checkpoint'.format(args.output_dir, args.base_file_name, epoch)
            torch.save(checkpoint, checkpoint_path)

    model_path = '{}/{}_final.model'.format(args.output_dir, args.base_file_name)
    torch.save(model, model_path)


if __name__ == '__main__':
    train_model()
