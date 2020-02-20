import torch


def pre_pad_sequence(seq, device):
    n_batch = len(seq)
    seq_len = [len(s) for s in seq]
    max_len = max(seq_len)
    n_elements = seq[0].size(1)

    out_seq = torch.zeros(n_batch, max_len, n_elements, device=device)
    for i, t in enumerate(seq):
        l = t.size(0)
        out_seq[i, max_len-l:, ...] = t
    return out_seq, seq_len


def make_ones(seq_len, n_elements, device):
    seq_list = [torch.FloatTensor([[1.0 for _ in range(n_elements)] for _ in range(v)], device=device) for v in seq_len]
    return pre_pad_sequence(seq_list, device=device)
