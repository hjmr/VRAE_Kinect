import six

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import make_ones


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

    def forward(self, enc_inp, dec_inp, inp_len):
        mu, ln_var = self.encode(enc_inp, inp_len)
        return self.decode(mu, dec_inp, inp_len)

    def encode(self, enc_inp, inp_len):
        n_batch = enc_inp.size(0)
        packed_in = pack_padded_sequence(enc_inp, inp_len, batch_first=True)
        _, (h_enc, _) = self.encoder(packed_in)
        mu = self.enc_mu(h_enc.view(n_batch, self.n_enc_layers * self.n_enc_hidden))
        ln_var = self.enc_ln_var(h_enc.view(n_batch, self.n_enc_layers * self.n_enc_hidden))
        return mu, ln_var

    def decode(self, z, dec_inp, inp_len):
        n_batch = z.size(0)
        packed_in = pack_padded_sequence(dec_inp, inp_len, batch_first=True)
        h_dec = self.gen_z_h(z)
        dec_c0 = torch.zeros(self.n_dec_layers, n_batch, self.n_dec_hidden,
                             requires_grad=True, device=self.get_device())
        packed_out, _ = self.decoder(packed_in, (h_dec.view(self.n_dec_layers, n_batch, self.n_dec_hidden), dec_c0))
        out, out_len = pad_packed_sequence(packed_out, batch_first=True)
        return self.dec_out(out)

    def generate(self, z, seq_len):
        with torch.no_grad():
            dec_inp = make_ones(seq_len, self.n_input, self.get_device())
            dec_out = self.decode(z, dec_inp, seq_len)
        return dec_out

    def loss(self, enc_inp, inp_len, beta=1.0, k=1):
        mu, ln_var = self.encode(enc_inp, inp_len)
        loss_func = nn.MSELoss(reduction='mean')
        n_batch = enc_inp.size(0)
        dec_inp = make_ones(inp_len, self.n_input, self.get_device())
        rec_loss = 0
        for _ in six.moves.range(k):
            z = torch.normal(mu, ln_var)
            dec_out = self.decode(z, dec_inp, inp_len)
            rec_loss += loss_func(enc_inp, dec_out)
        rec_loss /= (k * n_batch)
        kld = -0.5 * torch.sum(1 + ln_var - mu.pow(2) - ln_var.exp()) / n_batch
        return rec_loss + beta * kld
