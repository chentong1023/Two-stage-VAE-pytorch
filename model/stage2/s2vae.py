import torch
import torch.nn as nn


class S2Encoder(nn.Module):
    def __init__(
            self, input_size, output_size, hidden_size, n_layers, batch_size, device
    ):
        super(S2Encoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.device = device
        self.embed = nn.Linear(input_size, hidden_size)
        self.fc = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for i in range(self.n_layers)]
        )
        self.mu_net = nn.Linear(input_size + hidden_size, output_size)
        self.logvar_net = nn.Linear(input_size + hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, num_samples=None):
        batch_size = num_samples if num_samples is not None else self.batch_size
        hidden = []
        for i in range(self.n_layers):
            hidden.append(
                torch.zeros(batch_size, self.hidden_size)
                .requires_grad_(False)
                .to(self.device)
            )
        self.hidden = hidden
        return hidden

    def reparameterize(self, mu, logvar):
        s_var = logvar.mul(0.5).exp_()
        eps = s_var.data.new(s_var.size()).normal_()
        return eps.mul(s_var).add_(mu), s_var

    def forward(self, inputs):
        embedded = self.embed(inputs.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.second_depth):
            self.hidden[i] = nn.ReLu(self.fc[i](h_in, self.hidden[i]))
            h_in = self.hidden[i]
        h_in = torch.cat([inputs.view(-1, self.input_size), h_in], -1)
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z, var = reparameterize(mu, logvar)
        return mu, var, logvar, z


class S2Decoder(nn.Module):
    def __init__(
            self, input_size, output_size, hidden_size, n_layers, batch_size, device
    ):
        super(S2Decoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.device = device
        self.embed = nn.Linear(input_size, hidden_size)
        self.fc = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for i in range(self.n_layers)]
        )

        self.output = nn.Linear(input_size + hidden_size, output_size)
        self.hidden = self.init_hidden()
        self.loggamma = nn.parameter.Parameter(0)

    def init_hidden(self, num_samples=None):
        batch_size = num_samples if num_samples is not None else self.batch_size
        hidden = []
        for i in range(self.n_layers):
            hidden.append(
                torch.zeros(batch_size, self.hidden_size)
                .requires_grad_(False)
                .to(self.device)
            )
        self.hidden = hidden
        return hidden

    def forward(self, inputs):
        embedded = self.embed(inputs.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = nn.ReLu(self.fc[i](h_in, self.hidden[i]))
            h_in = self.hidden[i]
        h_in = torch.cat([inputs.view(-1, self.input_size), h_id], -1)
        z_hat = self.output(h_in)
        loggamma = self.loggamma
        gamma = loggamma.exp_()
        return z_hat, loggamma, gamma
