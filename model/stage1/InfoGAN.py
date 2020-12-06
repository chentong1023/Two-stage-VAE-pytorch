import torch
import torch.nn as nn

class InfoganEncoder(nn.Module):
	def __init__(self, latent_dim):
		super(InfoganEncoder, self).__init__()
		self.latent_dim = latent_dim
		self.conv1 = 
	
	def forward(self, inputs):
		y = inputs
		y = nn.LeakyReLu()