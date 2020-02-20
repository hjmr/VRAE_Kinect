import six

import torch
import torch.nn as nn


def make_ones(seq_len, n_elements, device):
    seq_list = [torch.FloatTensor([[1.0 for _ in range(n_elements)] for _ in range(v)], device=device) for v in seq_len]
    return pre_pad_sequence(seq_list, device=device)


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


class VRAE(nn.Module):
    "Variational Recurrent AUtoEncoder with LSTM"

    def __init__(self, n_input, n_enc_hidden, n_latent, n_dec_hidden, n_enc_layers=1, n_dec_layers=1, dropout_rate=0.5):
        super(VRAE, self).__init__()
        # Encoder
        if 1 < n_enc_layers:
            self.encoder = nn.LSTM(input_size=n_input,
                                   hidden_size=n_enc_hidden,
                                   num_layers=n_enc_layers,
                                   dropout=dropout_rate,
                                   batch_first=True)
        else:
            self.encoder = nn.LSTM(input_size=n_input,
                                   hidden_size=n_enc_hidden,
                                   num_layers=n_enc_layers,
                                   batch_first=True)
        self.enc_mu = nn.Linear(n_enc_layers * n_enc_hidden, n_latent)
        self.enc_ln_var = nn.Linear(n_enc_layers * n_enc_hidden, n_latent)
        # Decoder
        self.gen_z_h = nn.Linear(n_latent, n_dec_layers * n_dec_hidden)
        if 1 < n_dec_layers:
            self.decoder = nn.LSTM(input_size=n_input,
                                   hidden_size=n_dec_hidden,
                                   num_layers=n_dec_layers,
                                   dropout=dropout_rate,
                                   batch_first=True)
        else:
            self.decoder = nn.LSTM(input_size=n_input,
                                   hidden_size=n_dec_hidden,
                                   num_layers=n_dec_layers,
                                   batch_first=True)
        self.dec_out = nn.Linear(n_dec_hidden, n_input)

        # Parameters
        self.n_input = n_input
        self.n_enc_layers = n_enc_layers
        self.n_enc_hidden = n_enc_hidden
        self.n_latent = n_latent
        self.n_dec_layers = n_dec_layers
        self.n_dec_hidden = n_dec_hidden
        self.dropout_rate = dropout_rate

        # Loss
        self.rec_loss = 0

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, enc_inp, dec_inp):
        mu, ln_var = self.encode(enc_inp)
        return self.decode(mu, dec_inp)

    def encode(self, enc_inp):
        n_batch = enc_inp.size(0)
        _, (h_enc, _) = self.encoder(enc_inp)
        mu = self.enc_mu(h_enc.view(n_batch, self.n_enc_layers * self.n_enc_hidden))
        ln_var = self.enc_ln_var(h_enc.view(n_batch, self.n_enc_layers * self.n_enc_hidden))
        return mu, ln_var

    def decode(self, z, dec_inp):
        n_batch = z.size(0)
        h_dec = self.gen_z_h(z)
        dec_c0 = torch.zeros(self.n_dec_layers, n_batch, self.n_dec_hidden,
                             requires_grad=True, device=self.get_device())
        out, _ = self.decoder(dec_inp, (h_dec.view(self.n_dec_layers, n_batch, self.n_dec_hidden), dec_c0))
        return self.dec_out(out)

    def generate(self, z, seq_len):
        with torch.no_grad():
            dec_inp, _ = make_ones(seq_len, self.n_input, self.get_device())
            dec_out = self.decode(z, dec_inp)
        return dec_out

    def loss(self, enc_inp, inp_len, beta=1.0, k=1):
        mu, ln_var = self.encode(enc_inp)
        loss_func = nn.MSELoss(reduction='mean')
        n_batch = enc_inp.size(0)
        dec_inp, _ = make_ones(inp_len, self.n_input, self.get_device())
        rec_loss = 0
        for _ in six.moves.range(k):
            z = torch.normal(mu, ln_var)
            dec_out = self.decode(z, dec_inp)
            rec_loss += loss_func(enc_inp, dec_out)
        rec_loss /= (k * n_batch)
        kld = -0.5 * torch.sum(1 + ln_var - mu.pow(2) - ln_var.exp()) / n_batch
        return rec_loss + beta * kld
