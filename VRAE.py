import six

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

import utils


class VRAE(nn.Module):
    "Variational Recurrent AUtoEncoder with LSTM"

    def __init__(self, n_input, n_enc_hidden, n_latent, n_dec_hidden, n_enc_layers=1, n_dec_layers=1, dropout_rate=0.5):
        super(VRAE, self).__init__()
        # Encoder
        enc_dropout = 0 if n_enc_layers == 1 else dropout_rate
        self.encoder = nn.LSTM(input_size=n_input,
                               hidden_size=n_enc_hidden,
                               num_layers=n_enc_layers,
                               dropout=enc_dropout,
                               batch_first=True)
        self.enc_mu = nn.Linear(n_enc_layers * n_enc_hidden, n_latent)
        self.enc_ln_var = nn.Linear(n_enc_layers * n_enc_hidden, n_latent)
        # Decoder
        self.gen_z_h = nn.Linear(n_latent, n_dec_layers * n_dec_hidden)
        dec_dropout = 0 if n_dec_layers == 1 else dropout_rate
        self.decoder = nn.LSTM(input_size=n_input,
                               hidden_size=n_dec_hidden,
                               num_layers=n_dec_layers,
                               dropout=dec_dropout,
                               batch_first=True)
        self.linear_out = nn.Linear(n_dec_hidden, n_input)

        # Parameters
        self.n_input = n_input
        self.n_enc_layers = n_enc_layers
        self.n_enc_hidden = n_enc_hidden
        self.n_latent = n_latent
        self.n_dec_layers = n_dec_layers
        self.n_dec_hidden = n_dec_hidden
        self.dropout_rate = dropout_rate

        # Loss
        self.loss_func = nn.MSELoss(reduction="mean")

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, enc_inp):
        mu, ln_var = self.encode(enc_inp)
        seq_len = [len(s) for s in enc_inp]
        return self.decode(mu, seq_len)

    def encode(self, enc_inp):
        n_batch = len(enc_inp)
        packed_in = pack_sequence(enc_inp, enforce_sorted=False)
        _, (h_enc, _) = self.encoder(packed_in)
        h_enc = h_enc.view(n_batch, self.n_enc_layers * self.n_enc_hidden)
        mu = self.enc_mu(h_enc)
        ln_var = self.enc_ln_var(h_enc)
        return mu, ln_var

    def decode(self, z, seq_len):
        n_batch = len(seq_len)
        dec_inp = utils.make_zeros(seq_len, self.n_input, self.get_device())
        packed_in = pack_sequence(dec_inp, enforce_sorted=False)
        h_dec = self.gen_z_h(z)
        dec_c0 = torch.zeros(self.n_dec_layers, n_batch, self.n_dec_hidden, requires_grad=True).to(self.get_device())
        packed_out, _ = self.decoder(packed_in, (h_dec.view(self.n_dec_layers, n_batch, self.n_dec_hidden), dec_c0))
        padded_out, out_len = pad_packed_sequence(packed_out, batch_first=True)
        out = utils.unpad_sequence(padded_out, out_len)
        return [self.linear_out(s) for s in out]

    def generate(self, z, seq_len):
        with torch.no_grad():
            dec_out = self.decode(z, seq_len)
        return dec_out

    def loss(self, enc_inp, beta=1.0, k=1):
        mu, ln_var = self.encode(enc_inp)
        n_batch = len(enc_inp)
        inp_len = [len(s) for s in enc_inp]
        rec_loss = 0
        for _ in six.moves.range(k):
            z = torch.normal(mu, ln_var)
            dec_out = self.decode(z, inp_len)
            for o, t in zip(dec_out, enc_inp):
                rec_loss += self.loss_func(o, t)
        # averaged over k * n_batch
        # rec_loss /= (k * n_batch)
        # kld = -0.5 * torch.mean(1 + ln_var - mu.pow(2) - ln_var.exp())
        # averaged over k
        rec_loss /= k
        kld = -0.5 * torch.sum(1 + ln_var - mu.pow(2) - ln_var.exp())
        return rec_loss + beta * kld, rec_loss, kld


if __name__ == "__main__":
    model = VRAE(2, 5, 2, 5, 1, 1, 0)
    for n, p in model.named_parameters():
        print(n)
        print(p)
        print("--------")
