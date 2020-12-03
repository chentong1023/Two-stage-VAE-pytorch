import torch
import torch.nn as nn


class WaeEncoder(nn.Module):
    def __init__(self, latent_dim, side_length, channel):
        super(WaeEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.channel = channel
        self.side_length = side_length
        self.cn1 = nn.Conv2d(self.channel, self.channel * 128, 5, 1, 2)
        self.cn2 = nn.Conv2d(self.channel * 128, self.channel * 256, 5, 2, 2)
        self.cn3 = nn.Conv2d(self.channel * 256, self.channel * 512, 5, 2, 2)
        self.cn4 = nn.Conv2d(self.channel * 512, self.channel * 1024, 5, 2, 2)
        self.bn1 = nn.BatchNorm2d(self.channel * 128)
        self.bn2 = nn.BatchNorm2d(self.channel * 256)
        self.bn3 = nn.BatchNorm2d(self.channel * 512)
        self.bn4 = nn.BatchNorm2d(self.channel * 1024)
        self.relu = nn.ReLU()
        self.conv_size = self.channel * 1024 * 4 * 4
        self.mu_net = nn.Linear(self.conv_size, self.latent_dim)
        self.logvar_net = nn.Linear(self.conv_size, self.latent_dim)

    def reparameterize(self, mu, logvar):
        s_var = logvar.mul(0.5).exp_()
        eps = s_var.data.new(s_var.size()).normal_()
        return eps.mul(s_var).add_(mu), s_var

    def forward(self, inputs):
        y = inputs
        y = self.relu(self.bn1(self.cn1(y)))
        y = self.relu(self.bn2(self.cn2(y)))
        y = self.relu(self.bn3(self.cn3(y)))
        y = self.relu(self.bn4(self.cn4(y)))

        y = y.view(y.size()[0], -1)

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
        self.fc = nn.Linear(self.latent_dim, 8 * 8 * 1024)
        self.bn1 = nn.ConvTranspose2d(self.channel * 1024, self.channel * 512, 5, 2)
        self.bn2 = nn.ConvTranspose2d(self.channel * 512, self.channel * 256, 5, 2)
        self.bn3 = nn.ConvTranspose2d(self.channel * 256, self.channel * 128, 5, 2)
        self.bn4 = nn.ConvTranspose2d(self.channel * 128, self.channel, 5, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.loggamma = nn.parameter.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, inputs):
        y = inputs
        y = self.relu(self.fc(y))
        print(y.shape)
        y = y.reshape([-1, 1024, 8, 8])
        y = self.relu(self.bn1(y))
        print(y.shape)
        y = self.relu(self.bn2(y))
        print(y.shape)
        y = self.relu(self.bn3(y))
        print(y.shape)
        y = self.bn4(y)
        print(y.shape)

        x_hat = self.sigmoid(y)
        loggamma_x = self.loggamma
        gamma_x = loggamma_x.exp_()

        return x_hat, loggamma_x, gamma_x
