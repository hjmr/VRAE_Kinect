import torch


def pre_pad_sequence(seq, device):
    n_batch = len(seq)
    seq_len = torch.LongTensor([len(s) for s in seq])
    max_len = max(seq_len)
    n_elements = seq[0].size(1)

    out_seq = torch.zeros(n_batch, max_len, n_elements)
    for i, t in enumerate(seq):
        l = t.size(0)
        out_seq[i, max_len-l:, ...] = t
    return out_seq.to(device), seq_len.to(device)


def make_ones(seq_len, n_elements, device):
    seq_list = [torch.FloatTensor([[1.0 for _ in range(n_elements)] for _ in range(v)]) for v in seq_len]
    return pre_pad_sequence(seq_list, device)
