import os
import argparse

import pandas as pd

COLUMN_TAGS = ('epoch', 'train/loss', 'test/loss')


def parse_arg():
    parser = argparse.ArgumentParser(description='Calc average and standard deviation from train.logs.')
    parser.add_argument('-e', '--epoch', type=int, default=10,
                        help='epochs to be converted.')
    parser.add_argument('-t', '--tags', type=str, nargs=1,
                        help='specify columns of the log by comma-seperated-tags. '
                        'tag(s) should be one or more from {}'.format(COLUMN_TAGS[:1]))
    parser.add_argument('-o', '--output', type=str,
                        help='specifyl filename to which save result in CSV.')
    parser.add_argument('dirs', type=str, nargs='+',
                        help='directorie name(s) contain log data.')
    return parser.parse_args()


def main():
    args = parse_arg()
    tags = args.tags[0].split(',')
    usecols = []
    for t in tags:
        if t in COLUMN_TAGS:
            usecols.append(COLUMN_TAGS.index(t))

    data = pd.DataFrame()
    for d in args.dirs:
        file_tag = ''.join(os.path.basename(d).split('_'))
        file_name = '{}/train.log'.format(d)
        df = pd.read_csv(file_name,
                         nrows=args.epoch + 1,
                         comment='#',
                         names=[file_tag],
                         usecols=usecols)
        data = pd.concat([data, df], axis=1)

    data.index.name = 'epoch'
    if args.output is not None:
        data.to_csv(args.output)
    else:
        print(data)


if __name__ == '__main__':
    main()
