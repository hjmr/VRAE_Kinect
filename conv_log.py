import os
import argparse

COLUMN_TAGS = ['epoch', 'train/loss', 'test/loss', '3', '4', '5', '6', '7', '8', '9', '10']


def parse_arg():
    parser = argparse.ArgumentParser(description='Convert train.logs suitable to draw graph.')
    parser.add_argument('-e', '--epoch', type=int, default=10,
                        help='epochs to be converted.')
    parser.add_argument('-t', '--tags', type=str,
                        help='specify tag(s) included to log. '
                        'tag(s) should be one or more from {}'.format(COLUMN_TAGS))
    parser.add_argument('dirs', type=str, nargs='+',
                        help='directorie name(s) contain log data.')
    return parser.parse_args()


def register_data(data_array, tag_array, epoch, item, item_tag):
    if item_tag not in tag_array:
        tag_array.append(item_tag)
    data_array[epoch][item_tag] = item


def conv_single_log(data_array, tag_array, file_name, file_tag, max_epoch, tags):
    with open(file_name, 'r') as f:
        for line in f:
            if (not line.startswith('#')) and ',' in line:
                items = line.strip().split(',')
                epoch = int(items[0])
                if epoch <= max_epoch:
                    for idx in range(1, len(items)):
                        if tags is None or COLUMN_TAGS[idx] in tags:
                            item_tag = "{}/{}".format(file_tag, COLUMN_TAGS[idx])
                            register_data(data_array, tag_array, epoch, items[idx], item_tag)


def conv_log(args):
    data_array = [{} for i in range(args.epoch + 1)]
    tag_array = []
    tags = args.tags.split(',') if args.tags is not None else None
    for dn in args.dirs:
        file_tag = ''.join(os.path.basename(dn).split('_'))
        conv_single_log(data_array, tag_array, "{}/train.log".format(dn), file_tag, args.epoch, tags)
    return data_array, tag_array


if __name__ == '__main__':
    args = parse_arg()
    data_array, tag_array = conv_log(args)
    print('epoch,{}'.format(','.join(tag_array)))
    for i in range(1, args.epoch+1):
        line = [str(i)]
        epoch_data = data_array[i]
        for t in tag_array:
            if t in epoch_data:
                line.append(epoch_data[t])
            else:
                line.append('')
        print(','.join(line))
