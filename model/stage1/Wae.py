import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class WaeEncoder(nn.Module):
    def __init__(self, latent_dim, side_length, channel):
        super(WaeEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.channel = channel
        self.side_length = side_length
        self.main = nn.Sequential(
            nn.Conv2d(channel, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),            # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024*4*4)),                                 # B, 1024*4*4
        )
        self.conv_size = 1024 * 4 * 4
        self.mu_net = nn.Linear(self.conv_size, self.latent_dim)
        self.logvar_net = nn.Linear(self.conv_size, self.latent_dim)

    def reparameterize(self, mu, logvar):
        s_var = logvar.mul(0.5).exp_()
        eps = s_var.data.new(s_var.size()).normal_()
        return eps.mul(s_var).add_(mu), s_var

    def forward(self, inputs):
        y = self.main(inputs)
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

        self.main = nn.Sequential(
            nn.Linear(latent_dim, 1024*8*8),                           # B, 1024*8*8
            View((-1, 1024, 8, 8)),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, channel, 1),                       # B,   nc, 64, 64
            nn.Sigmoid(),
        )
        self.loggamma = nn.parameter.Parameter(torch.zeros([channel, 64, 64]), requires_grad=True)

    def forward(self, inputs):
        x_hat = self.main(inputs)
        loggamma_x = self.loggamma
        gamma_x = torch.exp(loggamma_x)

        return x_hat, loggamma_x, gamma_x
