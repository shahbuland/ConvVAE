import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Gets list of channels required to get size (assumed int[3], and square)
# separate for encoding (CHANNELSxHxW -> 512xH_SIZExH_SIZE)
def get_ch_list_encoding(channels, size, h_size):
	f = channels # Keep track of current filter count
	ch = [f] # List of num filters (channels)
	while size != h_size and size != 1:
		size = size // 2
		if f == channels: f = 64
		else: f *= 2
		ch.append(f)
	return ch

def get_ch_list_decoding(channels, h_size, h_filters, size):
	f = h_filters
	ch = [f]
	while h_size != size:
		h_size *= 2
		if f == 64: f = channels
		else: f = f // 2
		ch.append(f)
	return ch

# ENCODER MODEL

class Encoder(nn.Module):
	# The first two are required
	# CHANNELS is 1 if gray, 3 if RGB
	# The rest are optionial 
	def __init__(self, input_size, latent_dim, CHANNELS=1, H_SIZE=4,
				 use_bn=False, use_dropout=False, dp_prob=0.25):
		super(Encoder, self).__init__()
		
		# init some constants for model
		self.use_bn = use_bn
		self.use_dropout = use_dropout
		self.CHANNELS = CHANNELS
		self.H_SIZE = H_SIZE
		self.dp_prob = dp_prob
		
		# Need to figure out required channels to get from
		# [CHANNELS,input_size,input_size] -> [512,H_SIZE,H_SIZE]
		ch = get_ch_list_encoding(CHANNELS, input_size, H_SIZE)
		self.n_layers = len(ch) - 1
		# Module lists for layers
		self.conv = nn.ModuleList()
		self.bn = nn.ModuleList() #if batch norm is enabled
		self.dp = nn.ModuleList() #if dropout is enabled
		# fc layers (one for mu, one for logvar)
		self.mu_fc = nn.Linear(512*H_SIZE*H_SIZE, latent_dim)
		self.logvar_fc = nn.Linear(512*H_SIZE*H_SIZE, latent_dim)
	
		# Make conv layers
		for i in range(self.n_layers):
			self.conv.append(self.myconv(ch[i],ch[i+1]))

		# Make batch norm layers
		if use_bn:
			for i in range(self.n_layers):
				self.bn.append(nn.BatchNorm2d(ch[i+1]))

		# Make dropout layers
		if use_dropout:
			for i in range(self.n_layers):
				self.dp.append(nn.Dropout2d(dp_prob))


	# func to make conv layers quickly
	def myconv(self, fi, fo, k=4,s=2,p=1):
		return nn.Conv2d(fi,fo,k,s,p)

	# get a vector from mus and logvars
	def sample(self,mu,logvar):
		sigma = torch.exp(0.5*logvar)
		eps = torch.randn(*mu.size()) # ~ N(0,1)
		z = mu + sigma*eps # ~ N(mu, std)
		return z

	def forward(self, x):
		for i in range(self.n_layers):
			x = self.conv[i](x)
			x = F.relu(x)
			if self.use_bn: x = self.bn[i](x)
			if self.use_dropout: x= self.dp[i](x)
		x = x.view(-1,512*self.H_SIZE*self.H_SIZE)
		mu = self.mu_fc(x)
		logvar = self.logvar_fc(x)
		z = self.sample(mu, logvar)
		return z, mu, logvar

# DECODER MODEL

class Decoder(nn.Module):
	# The first two are required
	# CHANNELS is 1 if gray, 3 if RGB
	# The rest are optionial 
	def __init__(self, input_size, latent_dim, CHANNELS=1, H_SIZE=4,
				 use_upsampling=False, use_bn=False, use_dropout=False, dp_prob=0.25):
		super(Decoder, self).__init__()
		
		# init some constants for model
		self.use_bn = use_bn
		self.use_dropout = use_dropout
		self.use_upsampling = use_upsampling
		self.CHANNELS = CHANNELS
		self.H_SIZE = H_SIZE
		self.dp_prob = dp_prob
		
		# Need to figure out required channels to get from
		# [CHANNELS,input_size,input_size] -> [512,H_SIZE,H_SIZE]
		ch = get_ch_list_decoding(CHANNELS, H_SIZE, 512, input_size)
		self.n_layers = len(ch) - 1
		# Module lists for layers
		self.conv = nn.ModuleList()
		self.bn = nn.ModuleList() #if batch norm is enabled
		self.dp = nn.ModuleList() #if dropout is enabled
		
		self.fc = nn.Linear(latent_dim,512*H_SIZE*H_SIZE)

		# Make conv layers
		for i in range(self.n_layers):
			self.conv.append(self.myconv(ch[i],ch[i+1]))

		# Make batch norm layers
		if use_bn:
			for i in range(self.n_layers):
				self.bn.append(nn.BatchNorm2d(ch[i+1]))

		# Make dropout layers
		if use_dropout:
			for i in range(self.n_layers):
				self.dp.append(nn.Dropout2d(dp_prob))

		# upsampling layer
		self.up = nn.Upsample(scale_factor=2)

	def myconv(self, fi, fo, k=4,s=2,p=1):
		if self.use_upsampling: return nn.Conv2d(fi,fo,k,1,p)
		else: return nn.ConvTranspose2d(fi,fo,k,s,p)

	def forward(self, x):
		x = self.fc(x)
		x = x.view(-1,512,self.H_SIZE,self.H_SIZE)
		for i in range(self.n_layers):
			if self.use_upsampling: x = self.up(x)
			x = self.conv[i](x)
			if i != self.n_layers - 1: x = F.relu(x)
			if self.use_bn and i != self.n_layers - 1: x = self.bn[i](x) # Don't want to BN output
			if self.use_dropout and i != self.n_layers - 1: x = self.dp[i](x) # Don't want to DP output
		return torch.tanh(x)

# VAE model

class VAEmodel(nn.Module):
	def __init__(self,input_size,latent_dim,
				 CHANNELS=1, H_SIZE=4,
				 use_upsampling=False, use_bn=False, use_dropout=False, dp_prob=0.25):
		super(VAEmodel, self).__init__()
		
		self.E = Encoder(input_size, latent_dim, CHANNELS, H_SIZE, use_bn, use_dropout, dp_prob)
		self.D = Decoder(input_size, latent_dim, CHANNELS, H_SIZE, use_upsampling, use_bn, use_dropout, dp_prob)
		
	def forward(self,x):
		z, mu, logvar = self.E(x)
		rec_x = self.D(z)
		return rec_x, mu, logvar
