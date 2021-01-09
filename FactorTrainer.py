import torch
import math
import os
import numpy as np
import torch.optim as optim
from collections import OrderedDict
from model.losses import *
from utils.utils_ import print_current_loss
import time
from VaeTrainer import VaeTrainer

class FactorTrainer(VaeTrainer):
	def __init__(self, args, batch_sampler, device, stage, cross_entropy_loss=False):
		super(FactorTrainer, self).__init__(args, batch_sampler, device, stage, cross_entropy_loss, None)
		self.gen_loss = gen_loss3
		self.dis_cri = torch.nn.BCELoss()
	
	def sample_batch(self):
		if self.batch_enumerator is None:
			self.batch_enumerator = enumerate(self.batch_sampler)
		
		batch_idx, batch = next(self.batch_enumerator)
		
		if batch_idx == len(self.batch_sampler) - 1:
			self.batch_enumerator = enumerate(self.batch_sampler)
		# self.real_batch = batch # ?? have not used
		return batch
	
	def permute_dims(self, z):
		assert z.dim() == 2

		B, _ = z.size()
		perm_z = []
		for z_j in z.split(1, 1):
			perm = torch.randperm(B).to(z.device)
			perm_z_j = z_j[perm]
			perm_z.append(perm_z_j)

		return torch.cat(perm_z, 1)
	
	def train(self, encoder, decoder, disc, opt_encoder, opt_decoder, opt_disc, sample_true):
		opt_encoder.zero_grad()
		opt_decoder.zero_grad()
		opt_disc.zero_grad()
		
		data1 = sample_true()
		data1 = torch.clone(data1).float().detach_().to(self.device)
		
		log_dict = OrderedDict({"g_loss": 0})
		
		kld_loss = 0
		gen_loss = 0
		shuffle_loss = 0
		gan_loss = 0
		
		mu_z, sd_z, logsd_z, z = encoder(data1)
		z_sh = self.permute_dims(z).detach()
		x_sh_hat, _, _ = decoder(z_sh)
		x_sh_hat = x_sh_hat.detach()
		
		disc_gt = disc(data1)
		labelv = torch.FloatTensor(self.args.batch_size).to(self.device)
		labelv.resize_(self.args.batch_size).fill_(1)
		errD_real = self.dis_cri(disc_gt, labelv)
		
		disc_fake = disc(x_sh_hat)
		labelv = torch.FloatTensor(self.args.batch_size).to(self.device)
		labelv.resize_(self.args.batch_size).fill_(0)
		errD_fake = self.dis_cri(disc_fake, labelv)
		
		gan_loss += -errD_real + errD_fake
		
		gan_loss.backward(retain_graph=True)
		opt_disc.step()
		data1 = sample_true()
		data1 = torch.clone(data1).float().detach_().to(self.device)

		mu_z, sd_z, logsd_z, z = encoder(data1)
		x_hat, loggamma_x, gamma_x = decoder(z)
		z_sh = self.permute_dims(z).detach()
		x_sh_hat, _, _ = decoder(z_sh)
		
		# print(self.args.batch_size)
		kld_loss += self.kld_loss(mu_z, logsd_z)  
		gen_loss += self.gen_loss(data1, x_hat, loggamma_x)
		
		x_sh_hat = x_sh_hat.detach()
		disc_fake = disc(x_sh_hat)
		labelv = torch.FloatTensor(self.args.batch_size).to(self.device)
		labelv.resize_(self.args.batch_size).fill_(0)
		shuffle_loss += self.dis_cri(disc_fake, labelv)
		
		# disc_real = disc(data1)
		# labelv = torch.FloatTensor(self.args.batch_size).to(self.device)
		# labelv.resize_(self.args.batch_size).fill_(1)
		# temp_loss = self.dis_cri(disc_fake, labelv)
		
		losses = (kld_loss + gen_loss) / self.args.batch_size - self.args.alpha_gan * shuffle_loss
		losses.backward()
		opt_encoder.step()
		opt_decoder.step()
		opt_disc.step()

		avg_loss = losses.item()
		log_dict["g_kld_loss"] = kld_loss.item()
		log_dict["g_gen_loss"] = gen_loss.item()
		log_dict["g_gan_loss"] = gan_loss.item()
		log_dict["g_shuffle_loss"] = shuffle_loss.item() * 1000000000
		log_dict["g_loss"] = avg_loss
		
		return log_dict
	
	def trainIters(self, encoder, decoder, discriminator):
		self.opt_encoder = optim.Adam(
			encoder.parameters(),
			lr = self.args.lr,
			betas = (0.9, 0.999),
			weight_decay = 0.00001,
		)
		self.opt_decoder = optim.Adam(
			decoder.parameters(),
			lr = self.args.lr,
			betas = (0.9, 0.999),
			weight_decay = 0.00001,
		)
		self.opt_disc = optim.Adam(
			discriminator.parameters(),
			lr = self.args.lr,
			betas = (0.9, 0.999),
			weight_decay = 0.00001,
		)
		
		encoder.to(self.device)
		decoder.to(self.device)
		discriminator.to(self.device)
		
		def save_model(file_name):
			state = {
				"encoder": encoder.state_dict(),
				"decoder": decoder.state_dict(),
				"discriminator": discriminator.state_dict(),
				"opt_encoder": self.opt_encoder.state_dict(),
				"opt_decoder": self.opt_decoder.state_dict(),
				"opt_discriminator": self.opt_disc.state_dict(),
				"iterations": iter_num,
			}
			filename = "s" + str(self.stage) + "_" + file_name
			
			torch.save(state, os.path.join(self.args.model_path, filename) + ".tar")
		
		def load_model(file_name):
			filename = "s" + str(self.stage) + "_" + file_name
			model_dict = torch.load(os.path.join(self.args.model_path, filename + ".tar"))
			
			encoder.load_state_dict(model_dict["encoder"])
			decoder.load_state_dict(model_dict["decoder"])
			discriminator.load_state_dict(model_dict["discriminator"])
			self.opt_encoder.load_state_dict(model_dict["opt_encoder"])
			self.opt_decoder.load_state_dict(model_dict["opt_decoder"])
			self.opt_disc.load_state_dict(model_dict["opt_discriminator"])
			return model_dict
		
		if self.args.is_continue:
			model_dict = load_model("latest")
			iter_num = model_dict["iterations"]
		
		iter_num = 0
		logs = OrderedDict()
		start_time = time.time()
		
		while True:
			encoder.train()
			decoder.train()
			discriminator.train()
			
			if self.stage == 1:
				sample_true = self.sample_batch1
			else:
				sample_true = self.sample_batch2

			gen_log_dict = self.train(
				encoder,
				decoder,
				discriminator,
				self.opt_encoder,
				self.opt_decoder,
				self.opt_disc,
				sample_true
			)
			
			for k, v in gen_log_dict.items():
				if k not in logs:
					logs[k] = [v]
				else:
					logs[k].append(v)
			
			iter_num += 1
			
			if iter_num % self.args.print_every == 0:
				mean_loss = OrderedDict()
				for k, v in logs.items():
					mean_loss[k] = (
						sum(logs[k][-1 * self.args.print_every:]) / self.args.print_every
					)
				print_current_loss(start_time, iter_num, self.args.epochs, mean_loss)
			
			if iter_num % self.args.save_every == 0:
				save_model(str(iter_num))
			
			if iter_num % self.args.save_latest == 0:
				save_model("latest")
			
			if iter_num >= self.args.epochs:
				break
		return logs
		