import torch
import torch.nn as nn

class WaeEncoder(nn.Module):
	def __init__(self, latent_dim):
		super(WaeEncoder, self).__init__()
		self.latent_dim = latent_dim
		self.conv1 = nn.Conv2d()
	
	def reparameterize(self, mu, logvar):
        s_var = logvar.mul(0.5).exp_()
        eps = s_var.data.new(s_var.size()).normal_()
        return eps.mul(s_var).add_(mu), s_var
	
	def forward(self, inputs):
		y = inputs
		y = nn.ReLu(nn.BatchNorm2d(self.conv1(y)))
		y = nn.ReLu(nn.BatchNorm2d(self.conv2(y)))
		y = nn.ReLu(nn.BatchNorm2d(self.conv3(y)))
		y = nn.ReLu(nn.BatchNorm2d(self.conv4(y)))
		
		y = y.view(y.size()[0], -1)
		
		mu = self.mu_net(y)
		logvar = self.logvar_net(y)
		z, var = reparameterize(mu, logvar)
		return mu, var, logvar, z

class WaeDecoder(nn.Module):
	def __init__(self):
		