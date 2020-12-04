import torch
import torch.nn as nn


class WaeEncoder(nn.Module):
    def __init__(self, latent_dim, side_length, channel):
        super(WaeEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.channel = channel
        self.side_length = side_length
        self.main = nn.Sequential(
            nn.Conv2d(self.channel, self.side_length, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.side_length, self.side_length * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.side_length * 2),
            nn.ReLU(True),
            nn.Conv2d(self.side_length * 2, self.side_length * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.side_length * 4),
            nn.ReLU(True),
            nn.Conv2d(self.side_length * 4, self.side_length * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.side_length * 8),
            nn.ReLU(True),
        )
        self.conv_size = self.side_length * 8
        self.mu_net = nn.Linear(self.conv_size, self.latent_dim)
        self.logvar_net = nn.Linear(self.conv_size, self.latent_dim)

    def reparameterize(self, mu, logvar):
        s_var = logvar.mul(0.5).exp_()
        eps = s_var.data.new(s_var.size()).normal_()
        return eps.mul(s_var).add_(mu), s_var

    def forward(self, inputs):
        y = self.main(inputs)
        y = y.squeeze()

        mu = self.mu_net(y)
        logvar = self.logvar_net(y)
        z, var = self.reparameterize(mu, logvar)
        return mu, var, logvar, z


class WaeDecoder(nn.Module):
    def __init__(self, latent_dim, side_length, channel):
        super(WaeDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.channel = channel
        self.side_length = side_length

        self.proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.side_length * 8 * 7 * 7),
            nn.ReLU()
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.side_length * 8, self.side_length * 4, 4),
            nn.BatchNorm2d(self.side_length * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.side_length * 4, self.side_length * 2, 4),
            nn.BatchNorm2d(self.side_length * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.side_length * 2, 1, 4, stride=2),
            nn.Sigmoid()
        )
        self.loggamma = nn.parameter.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, inputs):
        y = self.proj(inputs)
        y = y.view(-1, self.side_length * 8, 7, 7)
        x_hat = self.main(y)
        loggamma_x = self.loggamma
        gamma_x = torch.exp(loggamma_x)

        return x_hat, loggamma_x, gamma_x
