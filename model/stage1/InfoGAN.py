import torch
import torch.nn as nn
# from network.util import conv2d

class InfoGANEncoder(nn.Module):
	def __init__(self, g_h, g_w, channels, latent_dim, batch_size):
		super(InfoGANEncoder, self).__init__()
		self.g_h = g_h
		self.g_w = g_w
		self.channels = channels
		self.latent_dim = latent_dim
		self.batch_size = batch_size
		self.net1 = nn.Sequential(
			nn.utils.spectral_norm(nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1)),
			nn.LeakyReLU(negative_slope=0.2),
			nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(negative_slope=0.2)
		)
		self.net2 = nn.Sequential(
			nn.utils.spectral_norm(nn.Linear(128 * (g_h // 4) * (g_w // 4), 1024)),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(negative_slope=0.2),
			nn.utils.spectral_norm(nn.Linear(1024, 2 * self.latent_dim))
		)

	def reparameterize(self, mu, log_var):
		sd = torch.exp(log_var)
		z = mu + sd.sqrt() * torch.randn_like(sd).detach()
		return z, sd

	def forward(self, inputs):
		y = inputs
		y = self.net1(y)
		y = torch.reshape(y, [inputs.size(0), -1])
		gaussian_params = self.net2(y)
		mu_z = gaussian_params[:,:self.latent_dim]
		log_sd_z = gaussian_params[:, self.latent_dim:]
		z, sd = self.reparameterize(mu_z,log_sd_z)
		return mu_z, sd, log_sd_z, z


class InfoGANDecoder(nn.Module):
	def __init__(self, g_h, g_w, channels, latent_dim, batch_size):
		super(InfoGANDecoder, self).__init__()
		self.g_h = g_h
		self.g_w = g_w
		self.channels = channels
		self.latent_dim = latent_dim
		self.batch_size = batch_size
		self.net1 = nn.Sequential(
			nn.Linear(self.latent_dim, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 128 * (g_h // 4) * (g_w // 4)),
			nn.BatchNorm1d(128 * (g_h // 4) * (g_w // 4)),
			nn.ReLU()
		)
		self.net2 = nn.Sequential(
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),
			nn.Sigmoid()
		)

	def forward(self, input):
		x_hat = self.net1(input)
		x_hat = torch.reshape(x_hat, [self.batch_size, 128, self.g_h // 4, self.g_w // 4])
		x_hat = self.net2(x_hat)
		log_gamma_x = torch.zeros(size=x_hat.shape, dtype=torch.float32)
		gamma_x = torch.exp(log_gamma_x)
		return x_hat, log_gamma_x, gamma_x
