import os
import argparse

import pandas as pd

COLUMN_TAGS = ('epoch', 'train/loss', 'test/loss')


def parse_arg():
    parser = argparse.ArgumentParser(description='Calc average and standard deviation from train.logs.')
    parser.add_argument('-e', '--epoch', type=int, default=10,
                        help='epochs to be converted.')
    parser.add_argument('-t', '--tag', type=str, nargs=1,
                        help='specify a column of the log by the tag. '
                        'tag should be one from {}'.format(COLUMN_TAGS))
    parser.add_argument('-x', '--exclude_original', action='store_true',
                        help='to exclude original data.')
    parser.add_argument('-o', '--output', type=str,
                        help='specify filename to which save CSV.')
    parser.add_argument('dirs', type=str, nargs='+',
                        help='directorie name(s) contain log data.')
    return parser.parse_args()


def main():
    args = parse_arg()
    tag = args.tag[0]
    if tag in COLUMN_TAGS:
        data = pd.DataFrame()
        for d in args.dirs:
            file_tag = ''.join(os.path.basename(d).split('_'))
            file_name = '{}/train.log'.format(d)
            df = pd.read_csv(file_name,
                             nrows=args.epoch + 1,
                             comment='#',
                             names=[file_tag],
                             usecols=[COLUMN_TAGS.index(tag)])
            data = pd.concat([data, df], axis=1)
        avg = data.mean(axis='columns', skipna=False, numeric_only=True)
        var = data.var(axis='columns', skipna=False, numeric_only=True)
        std = data.std(axis='columns', skipna=False, numeric_only=True)

        if args.exclude_original:
            data = pd.DataFrame()  # clear data

        data['average'] = avg
        data['variance'] = var
        data['std_dev'] = std

        data.index.name = 'epoch'
        if args.output is not None:
            data.to_csv(args.output)
        else:
            print(data)


if __name__ == '__main__':
    main()
