import torch
from torch.nn.utils.rnn import pad_sequence


def make_ones(seq_len, n_elements):
    seq_list = [torch.FloatTensor([[1.0 for _ in range(n_elements)] for _ in range(v)]) for v in seq_len]
    return pad_sequence(seq_list, batch_first=True)
