import six

import numpy
import torch
import torch.nn as nn


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

    def forward(self, x_inp, y_inp):
        mu, ln_var = self.encode(x_inp)
        return self.decode(mu, y_inp)

    def encode(self, x_inp):
        n_batch = x_inp.size(0)
        _, (h_enc, _) = self.encoder(x_inp)
        mu = self.enc_mu(h_enc.view(n_batch, self.n_enc_layers * self.n_enc_hidden))
        ln_var = self.enc_ln_var(h_enc.view(n_batch, self.n_enc_layers * self.n_enc_hidden))
        return mu, ln_var

    def decode(self, z, y_inp):
        n_batch = z.size(0)
        h_dec = self.gen_z_h(z)
        dec_c0 = torch.zeros(self.n_dec_layers, n_batch, self.n_dec_hidden, requires_grad=True)
        y_dec, _ = self.decoder(y_inp, (h_dec.view(self.n_dec_layers, n_batch, self.n_dec_hidden), dec_c0))
        return self.dec_out(y_dec)

    def generate(self, z, max_length=100):
        n_batch = z.size(0)
        with torch.no_grad():
            y_inp = torch.ones(n_batch, max_length, self.n_input)
            y_out = self.decode(z, y_inp)
        return y_out

    def loss(self, x_inp, beta=1.0, k=1):
        mu, ln_var = self.encode(x_inp)
        loss_func = nn.MSELoss(reduction='mean')
        n_batch = x_inp.size(0)
        seq_len = x_inp.size(1)
        y_inp = torch.ones(n_batch, seq_len, self.n_input)
        rec_loss = 0
        for _ in six.moves.range(k):
            z = torch.normal(mu, ln_var)
            y_out = self.decode(z, y_inp)
            rec_loss += loss_func(x_inp, y_out)
        rec_loss /= (k * n_batch)
        kld = -0.5 * torch.sum(1 + ln_var - mu.pow(2) - ln_var.exp()) / n_batch
        return rec_loss + beta * kld
