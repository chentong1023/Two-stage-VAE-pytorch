import torch

def kl_loss(mu, logsd):
	kld = (torch.pow(mu, 2) + torch.pow(torch.exp(logsd), 2) - 2.0 * logsd - 1) / 2.0
	return kld.sum()
	
def gen_loss1(x, x_hat, loggamma_x):
	relu = torch.nn.ReLU()
	gen = x * torch.log(relu(x_hat)) + (1-x) * torch.log(relu(1-x_hat))
	return -gen.sum()

def gen_loss2(x, x_hat, loggamma_x):
	HALF_LOG_TWO_PI = 0.91893
	mse = torch.nn.MSELoss()
	gen = torch.pow((x - x_hat) / torch.exp(loggamma_x), 2) / 2.0 + loggamma_x + HALF_LOG_TWO_PI
	return gen.sum()