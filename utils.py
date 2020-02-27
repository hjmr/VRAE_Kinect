import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


def make_ones(seq_len, n_elements, device):
    seq = [torch.ones(s, n_elements).to(device) for s in seq_len]
    return seq


def unpad_sequence(seq, seq_len):
    return [seq[i, :l, ...] for i, l in enumerate(seq_len)]
