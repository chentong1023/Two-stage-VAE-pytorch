import torch

def kl_loss(mu, logvar):
    kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean()
    return kld
	
def gen_loss1(x, x_recon, loggamma_x):
    n = x.size(0)
    loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(n)
    return loss

def gen_loss2(x, x_hat, loggamma_x):
	HALF_LOG_TWO_PI = 0.91893
	gen = torch.pow((x - x_hat) / torch.exp(loggamma_x), 2) / 2.0 + loggamma_x + HALF_LOG_TWO_PI
	return gen.sum()

def gen_loss3(x, x_hat, loggamma_x):
	mse = torch.nn.MSELoss()
	return mse(x, x_hat)
