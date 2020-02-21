import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


def make_ones(seq_len, n_elements):
    seq_list = [torch.FloatTensor([[1.0 for _ in range(n_elements)] for _ in range(v)]) for v in seq_len]
    return pad_sequence(seq_list, batch_first=True)


def ensure_padded_sequence(seq, seq_len):
    return pad_packed_sequence(pack_padded_sequence(seq, seq_len, batch_first=True), batch_first=True)
