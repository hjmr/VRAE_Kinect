import torch


def make_ones(seq_len, n_elements):
    return [torch.ones(s, n_elements) for s in seq_len]


def make_padded_sequence(seq):
    n_batch = len(seq)
    seq_len = torch.LongTensor([len(s) for s in seq])
    max_len = max(seq_len)
    n_elements = seq[0].size(1)

    out_seq = torch.zeros(n_batch, max_len, n_elements)
    for i, t in enumerate(seq):
        l = t.size(0)
        out_seq[i, max_len-l:, ...] = t
    return out_seq, seq_len


def make_unpadded_sequence(seq, seq_len):
    max_len = max(seq_len)
    return [seq[i, max_len-s:, ...] for i, s in enumerate(seq_len)]
