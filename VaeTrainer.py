import torch
import math
import os
import numpy as np
import torch.optim as optim
from collections import OrderedDict
from model.losses import *
from utils.utils_ import print_current_loss
import time

class VaeTrainer(object):
	def __init__(self, args, batch_sampler, device, stage, cross_entropy_loss=True):
		self.args = args
		self.device = device
		self.batch_sampler = batch_sampler
		self.batch_enumerator = None
		self.stage = stage
		self.kld_loss = kl_loss
		
		if cross_entropy_loss or stage == 2:
			self.gen_loss = gen_loss2
		else:
			self.gen_loss = gen_loss1
	
	def sample_batch(self):
		if self.batch_enumerator is None:
			self.batch_enumerator = enumerate(self.batch_sampler)
		
		batch_idx, batch = next(self.batch_enumerator)
		
		if batch_idx == len(self.batch_sampler) - 1:
			self.batch_enumerator = enumerate(self.batch_sampler)
		# self.real_batch = batch # ?? have not used
		return batch
	
	def train(self, encoder, decoder, opt_encoder, opt_decoder, sample_true):
		opt_encoder.zero_grad()
		opt_decoder.zero_grad()
		
		data = sample_true()
		data = torch.clone(data).float().detach_().to(self.device)
		
		log_dict = OrderedDict({"g_loss": 0})
		
		kld_loss = 0
		gen_loss = 0
		
		mu_z, sd_z, logsd_z, z = encoder(data)
		x_hat, loggamma_x, gamma_x = decoder(z)
		
		kld_loss += self.kld_loss(mu_z, sd_z, logsd_z)
		gen_loss += self.gen_loss(data, x_hat, gamma_x, loggamma_x)
		
		log_dict["g_kld_loss"] = kld_loss.item()
		log_dict["g_gen_loss"] = gen_loss.item()
		
		losses = (kld_loss + gen_loss)
		
		avg_loss = losses.item()
		
		losses.backward()
		opt_encoder.step()
		opt_decoder.step()
		
		log_dict["g_loss"] = avg_loss
		
		return log_dict
	
	def trainIters(self, encoder, decoder):
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
		
		encoder.to(self.device)
		decoder.to(self.device)
		
		def save_model(file_name):
			state = {
				"encoder": encoder.state_dict(),
				"decoder": decoder.state_dict(),
				"opt_encoder": self.opt_encoder.state_dict(),
				"opt_decoder": self.opt_decoder.state_dict(),
				"iterations": iter_num,
			}
			filename = "s" + str(self.stage) + "_" + file_name
			
			torch.save(state, os.path.join(self.args.model_path, filename) + ".tar")
		
		def load_model(file_name):
			filename = "s" + str(self.stage) + "_" + file_name
			model_dict = torch.load(os.path.join(self.args.model_path, filename + ".tar"))
			
			encoder.load_state_dict(model_dict["encoder"])
			decoder.load_state_dict(model_dict["decoder"])
			self.opt_encoder.load_state_dict(model_dict["opt_encoder"])
			self.opt_decoder.load_state_dict(model_dict["opt_decoder"])
		
		if self.args.is_continue and self.args.is_train:
			load_model("lastest")
		
		iter_num = 0
		logs = OrderedDict()
		start_time = time.time()
		
		while True:
			encoder.train()
			decoder.train()
			
			gen_log_dict = self.train(
				encoder,
				decoder,
				self.opt_encoder,
				self.opt_decoder,
				self.sample_batch,
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
		