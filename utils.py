import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


def make_padded_ones(seq_len, n_elements):
    seq = [torch.ones(s, n_elements) for s in seq_len]
    return pad_sequence(seq, batch_first=True)


def make_padded_sequence(seq):
    seq_len = torch.LongTensor([len(s) for s in seq])
    seq = pad_sequence(seq, batch_first=True)
    seq_len, perm_idx = seq_len.sort(0, descending=True)
    seq = seq[perm_idx]
    return seq, seq_len


def make_unpadded_sequence(seq, seq_len):
    return [seq[i, :l, ...] for i, l in enumerate(seq_len)]
