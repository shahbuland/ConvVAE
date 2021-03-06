# Torch stuff
import torch
from torch	import nn
import torch.nn.functional as F

# Models
from models import Encoder, Decoder, VAEmodel

# Some ops
from ops import weights_init_normal, randint, KL_Loss 

import matplotlib.pyplot as plt
import numpy as np

# Wrapper for VAE

# USE GUIDE:

# Constructor:
# Vals: 
#	CHANNELS - Number of colour channels in samples
#	H_SIZE - Size of data before being flattened for latent vector
#	kl_loss_weight - Weight of KL Loss, dictates how hard the model
# 					 focuses on objective of making latent space continuous
#	dp_prob - Probability of dropout layers dropping actiation values
# Flags:
#	use_upsampling - Use upsampling (as opposed to transposed convolution)
#	use_bn - Use batch normalization
#	use_dropout - Use dropout in the decoder (can encourage generalization)

# Load:
#	Given path, loads weights

# Save:
#	Saves weights as "VAEparams.pt"

# Drawsamples:
#	Draws samples when given tensors X and Y
#	Saves figure as file with "[title].png"

# Encode:
#	Encodes a pytorch tensor and returns latent vector

# Decode:
#	Decodes a pytorch tensor (latent vector) and returns constructed image

# Encode_np/Decode_np:
#	Same as above but work on numpy arrays

# Train:
#	Data - Input data to train on
#	Iterations - Iterations for which to train
#	Batch size - Size of batches
#	Sample_interval, Log_interval, Save_interval, do what they say	
class VAE:
	def __init__(self,input_size,latent_dim,
				 CHANNELS=1, H_SIZE=4, kl_loss_weight=1,
				 use_upsampling=False, use_bn=False, use_dropout=False, dp_prob=0.25,
				 CUDA=False):
		
		# constants
		self.CHANNELS = CHANNELS
		self.kl_loss_weight = kl_loss_weight
		self.CUDA = CUDA

		self.model = VAEmodel(input_size, latent_dim, CHANNELS,
					 H_SIZE, use_upsampling, use_bn, use_dropout, dp_prob)
		if self.CUDA: self.model.cuda()
		self.model.apply(weights_init_normal)

		self.opt = torch.optim.Adam(self.model.parameters(), lr=2e-4, betas=(0.5,0.999))
		
	def load(self,path):
		try: 
			self.model.load_state_dict(torch.load(path))
			print("Weights Loaded")
		except: print("No pt file found") 

	def save(self):
		try: torch.save(self.model.state_dict(),"VAEparams.pt")
		except: print("Couldn't save pt file")
	
	def drawsamples(self, title, X, Y):
		# show 5 samples
		fig,axs = plt.subplots(list(X.size())[0],2)
		X = 0.5*X + 0.5
		Y = 0.5*Y + 0.5
		X = X.cpu().detach().numpy()
		Y = Y.cpu().detach().numpy()
		X = np.moveaxis(X,1,3)
		Y = np.moveaxis(Y,1,3)
		if self.CHANNELS == 1: 
			cmap = 'gray'
			X = np.squeeze(X)
			Y = np.squeeze(Y)
		else: cmap = None
		for i in range((X.shape)[0]):
			axs[i][0].imshow(X[i],cmap=cmap)
			axs[i][1].imshow(Y[i],cmap=cmap)
		plt.show()	
		#plt.savefig(title+".png")
		plt.close()

	def encode(self, x):
		return self.model.E(x)[0]
	
	def decode(self, x):
		return self.model.D(x)
	
	def decode_np(self,x):
		# TODO: Figure out if x is array or single
		# Just two numbers
		t = torch.Tensor([[x[0],x[1]]])
		t = t.float()
		if self.CUDA: t = t.cuda()
		decoded_t = self.decode(t) # [1, 64, 64, 1]
		decoded_x = decoded_t.detach().cpu().numpy()
		decoded_x = np.squeeze(decoded_x)
		decoded_x = 0.5*decoded_x + 1
		return decoded_x

	def train(self, data, iterations, batch_size,
			  sample_interval = 50, log_interval = 10, save_interval = 250):
		self.model.train()
		print("Training...")

		total_size = list(data.size())[0]
		print("Loaded dataset with", total_size, "elements")

		for i in range(iterations):
			# get data
			inds = randint(batch_size, 0, total_size - 1)
			batch = data[inds]
			if self.CUDA: batch = batch.cuda()
			# input and target are both batch, run it through the VAE
			rec_batch, mus, logvars = self.model(batch)
			# reset optimizer
			self.opt.zero_grad()
			# get losses
			rec_loss = nn.MSELoss()(batch, rec_batch)
			kl_loss = self.kl_loss_weight*KL_Loss(mus,logvars)
			loss = rec_loss + kl_loss
			# backprop
			loss.backward()
			# updata params
			self.opt.step()
		
			# Logging and stuff
			if (i+1) % sample_interval == 0:
				print("Drawing samples")
				self.drawsamples(str(i), batch, rec_batch)
			if (i+1) % log_interval == 0:
				print("[",i+1,"/",iterations,"]", "Rec Loss:", rec_loss.item(), "KL Loss:", kl_loss.item())
			if (i+1) % save_interval == 0:
				print("Saving parameters...")
				self.save()
					


