import torch

def kl_loss(mu, sd, logsd):
	kld = (mu ** 2 + sd ** 2 - 2 * logsd - 1) / 2.0
	return kld.mean()
	
def gen_loss1(x, x_hat, gamma_x, loggamma_x):
	gen = x * torch.log(torch.max(x_hat, 1e-8)) + (1-x) * torch.log(torch.max(1-x_hat, 1e-8))
	return -gen.mean()

def gen_loss2(x, x_hat, gamma_x, loggamma_x):
	HALF_LOG_TWO_PI = 0.91893
	gen = ((x - x_hat) / gamma_x) ** 2 / 2.0 + loggamma_x + HALF_LOG_TWO_PI
	return gen.mean()